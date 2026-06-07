"""
collect_big_data.py — Collecte MASSIVE de données texte (plusieurs Go).

POURQUOI un autre script que fetch_data.py ?
  fetch_data.py scrape Wikipédia article par article : parfait pour quelques Mo,
  mais impossible de monter à 2-10 Go (trop de requêtes, rate-limit, trop lent).

  À l'échelle du Go, on change de méthode : on télécharge des datasets
  pré-construits en STREAMING via la librairie HuggingFace `datasets`. On lit le
  flux à la volée, on garde le texte, on l'écrit dans des fichiers "shards"
  jusqu'à atteindre la taille cible. Rien n'est chargé entièrement en RAM, et on
  ne stocke que ce qu'on garde.

INSTALLATION (une fois) :
    python -m pip install datasets zstandard

EXEMPLES :
    # 2 Go de Wikipédia anglais (encyclopédique, propre)
    python src/collect_big_data.py --target_gb 2 --source wikipedia

    # 2 Go de web haute qualité (FineWeb), filtré sur le vocabulaire SaaS/business
    python src/collect_big_data.py --target_gb 2 --source fineweb --filter_saas

    # 5 Go web généraliste (C4), sans filtre = le plus rapide
    python src/collect_big_data.py --target_gb 5 --source c4

Les fichiers sont écrits dans data/big/shard_XXXX.txt (≈100 Mo chacun).
Ensuite : prepare_data devra être adapté au streaming (voir README, section
"Passer à l'échelle du Go").

NOTE DISQUE : prévois ~1,5× la taille du texte. 10 Go de texte ≈ ~2 milliards de
tokens ≈ ~4 Go de fichiers .bin tokenisés. Donc ~15 Go libres pour viser 10 Go.
"""

import os
import re
import time
import argparse

# --- Presets de sources (datasets publics HuggingFace) ---
# text_key = nom du champ texte dans chaque exemple du dataset.
SOURCES = {
    # Encyclopédique, très propre. ~20 Go au total ; on en prend un sous-ensemble.
    "wikipedia": dict(path="wikimedia/wikipedia", name="20231101.en",
                      split="train", text_key="text"),
    # Web nettoyé (Common Crawl C4). Énorme. Bon volume, qualité correcte.
    "c4": dict(path="allenai/c4", name="en", split="train", text_key="text"),
    # Web haute qualité filtré (FineWeb). Échantillon 10B tokens, idéal en streaming.
    "fineweb": dict(path="HuggingFaceFW/fineweb", name="sample-10BT",
                    split="train", text_key="text"),
    # OpenWebText (~38 Go), style "articles partagés sur Reddit".
    "openwebtext": dict(path="Skylion007/openwebtext", name=None,
                        split="train", text_key="text"),
}

# Mots-clés pour garder uniquement le contenu SaaS/business (filtre optionnel).
SAAS_KEYWORDS = [
    "saas", "software as a service", "startup", "mvp", "product-market fit",
    "product market fit", "churn", "mrr", "arr", "subscription", "b2b",
    "onboarding", "pricing", "go-to-market", "go to market", "customer acquisition",
    "retention", "freemium", "product-led", "product led", "venture capital",
    "growth", "conversion rate", "cloud", "api", "dashboard", "landing page",
    "lifetime value", "ltv", "cac", "funnel", "saas metrics", "recurring revenue",
]
KEYWORDS_RE = re.compile("|".join(re.escape(k) for k in SAAS_KEYWORDS), re.IGNORECASE)


def keep_doc(text, filter_saas, min_chars, min_hits):
    """Décide si on garde un document."""
    if len(text) < min_chars:
        return False
    if filter_saas:
        # garde si au moins `min_hits` mots-clés distincts apparaissent.
        hits = len(set(m.lower() for m in KEYWORDS_RE.findall(text)))
        if hits < min_hits:
            return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="wikipedia", choices=list(SOURCES),
                   help="Dataset source (voir SOURCES).")
    p.add_argument("--target_gb", type=float, default=2.0,
                   help="Taille de texte à collecter, en Go.")
    p.add_argument("--out_dir", default="data/big",
                   help="Dossier des shards de sortie.")
    p.add_argument("--shard_mb", type=int, default=100,
                   help="Taille max d'un fichier shard (Mo).")
    p.add_argument("--filter_saas", action="store_true",
                   help="Ne garder que les documents contenant du vocabulaire SaaS.")
    p.add_argument("--min_hits", type=int, default=2,
                   help="Nb min de mots-clés distincts si --filter_saas.")
    p.add_argument("--min_chars", type=int, default=400,
                   help="Ignore les documents plus courts que ça.")
    p.add_argument("--skip_docs", type=int, default=0,
                   help="Sauter les N premiers documents du flux. Le streaming étant "
                        "DÉTERMINISTE, mets ici le total 'lus' d'un run précédent pour "
                        "récupérer du contenu NEUF (sinon tu retombes sur les mêmes docs).")
    args = parser_or_exit(p)

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "La librairie 'datasets' n'est pas installée.\n"
            "Installe-la avec :  python -m pip install datasets zstandard"
        )

    src = SOURCES[args.source]
    os.makedirs(args.out_dir, exist_ok=True)
    target_bytes = int(args.target_gb * 1024**3)
    shard_bytes = args.shard_mb * 1024**2
    sep = b"\n<|endoftext|>\n"

    print(f"Source        : {args.source} ({src['path']})")
    print(f"Cible         : {args.target_gb} Go")
    print(f"Filtre SaaS   : {'oui' if args.filter_saas else 'non'}")
    print(f"Sortie        : {args.out_dir}/shard_XXXX.txt ({args.shard_mb} Mo/shard)\n")
    print("Ouverture du flux (streaming)...")

    ds = load_dataset(src["path"], name=src["name"], split=src["split"],
                      streaming=True)
    text_key = src["text_key"]

    written = 0          # octets de texte gardés au total
    kept = 0             # documents gardés
    seen = 0             # documents lus
    shard_idx = 0
    shard_written = 0
    t0 = time.time()
    f = open(os.path.join(args.out_dir, f"shard_{shard_idx:04d}.txt"), "wb")

    try:
        for ex in ds:
            seen += 1
            if seen <= args.skip_docs:   # saute le contenu déjà vu lors d'un run précédent
                continue
            text = ex.get(text_key) or ""
            if not keep_doc(text, args.filter_saas, args.min_chars, args.min_hits):
                if seen % 50000 == 0:
                    print(f"  lus {seen:,} | gardés {kept:,} | "
                          f"{written / 1024**2:.0f} Mo")
                continue

            data = text.encode("utf-8") + sep
            f.write(data)
            written += len(data)
            shard_written += len(data)
            kept += 1

            # nouveau shard si le courant est plein.
            if shard_written >= shard_bytes:
                f.close()
                shard_idx += 1
                shard_written = 0
                f = open(os.path.join(args.out_dir, f"shard_{shard_idx:04d}.txt"), "wb")

            if kept % 2000 == 0:
                speed = written / 1024**2 / max(time.time() - t0, 1)
                print(f"  gardés {kept:,} docs | {written / 1024**2:.0f} Mo "
                      f"/ {args.target_gb * 1024:.0f} Mo | {speed:.1f} Mo/s")

            if written >= target_bytes:
                print("\nCible atteinte.")
                break
    finally:
        f.close()

    print(f"\nTerminé : {kept:,} documents gardés sur {seen:,} lus, "
          f"{written / 1024**3:.2f} Go -> {args.out_dir}/ "
          f"({shard_idx + 1} shards)")
    print("\nProchaine étape : python src/prepare_data.py --retrain_tokenizer")


def parser_or_exit(p):
    """parse_args isolé pour garder main() lisible."""
    return p.parse_args()


if __name__ == "__main__":
    main()
