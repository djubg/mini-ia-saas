"""
filter_corpus.py — Filtre un gros corpus brut pour n'en garder que le SaaS/business.

Utile quand on a téléchargé du web GÉNÉRIQUE (avec collect_big_data.py sans
--filter_saas) et qu'on veut un corpus SPÉCIALISÉ SaaS sans tout re-télécharger.

Critère STRICT (haute précision) :
  on garde un document s'il contient
    - au moins `--min_strong` mot(s)-clé(s) FORT(s) (très spécifiques SaaS), ET
    - au moins `--min_total` mots-clés DISTINCTS au total (forts + contexte).
  Frontières de mots (\\b) pour éviter les faux positifs (ex. 'arr' ≠ 'arrange').

Streaming : lit un fichier à la fois, écrit des shards filtrés au fur et à mesure.

Usage :
    python src/filter_corpus.py --in_dir data/big --out_dir data/big_saas
    python src/filter_corpus.py --in_dir data/big --out_dir data/big_saas --min_total 5
"""

import os
import re
import sys
import glob

# Console Windows / pipe = cp1252 -> force UTF-8 (sinon UnicodeEncodeError sur le
# texte accentué). line_buffering : progression visible en pipe.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass
import time
import argparse

# --- Mots-clés FORTS : presque toujours dans un contexte SaaS/startup ---
STRONG = [
    "saas", "software as a service", "b2b saas", "saas company", "saas platform",
    "saas product", "saas startup", "saas metrics", "product-market fit",
    "product market fit", "minimum viable product", "mvp", "churn rate", "churn",
    "monthly recurring revenue", "mrr", "annual recurring revenue", "arr",
    "freemium", "product-led", "product led growth", "go-to-market",
    "recurring revenue", "customer acquisition cost", "customer lifetime value",
    "subscription business", "free trial", "self-serve", "self-service software",
]

# --- Mots-clés de CONTEXTE : business/produit/growth, plus génériques ---
GENERAL = [
    "startup", "subscription", "pricing", "b2b", "onboarding", "retention",
    "conversion rate", "venture capital", "customer acquisition", "growth",
    "funnel", "dashboard", "api", "lifetime value", "ltv", "cac",
    "cloud software", "web application", "enterprise software", "software company",
    "monetization", "customer success", "go to market", "scale", "product manager",
    "user acquisition", "landing page", "value proposition",
]


def build_regex(words):
    """Regex insensible à la casse avec frontières de mots autour de chaque terme."""
    parts = [r"\b" + re.escape(w) + r"\b" for w in words]
    return re.compile("|".join(parts), re.IGNORECASE)


STRONG_RE = build_regex(STRONG)
ALL_RE = build_regex(STRONG + GENERAL)


def keep(text, min_strong, min_total):
    """Filtre strict avec court-circuit rapide sur les mots-clés forts."""
    # 1) rejet rapide : pas de mot-clé fort => on jette (cas le plus fréquent).
    strong_hits = set(m.lower() for m in STRONG_RE.findall(text))
    if len(strong_hits) < min_strong:
        return False
    # 2) assez de mots-clés distincts au total ?
    all_hits = set(m.lower() for m in ALL_RE.findall(text))
    return len(all_hits) >= min_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="data/big", help="Dossier des shards bruts.")
    p.add_argument("--out_dir", default="data/big_saas", help="Dossier de sortie filtré.")
    p.add_argument("--min_strong", type=int, default=1,
                   help="Nb min de mots-clés FORTS distincts requis.")
    p.add_argument("--min_total", type=int, default=4,
                   help="Nb min de mots-clés DISTINCTS au total requis.")
    p.add_argument("--min_chars", type=int, default=400,
                   help="Ignore les documents plus courts que ça.")
    p.add_argument("--shard_mb", type=int, default=100, help="Taille max d'un shard (Mo).")
    p.add_argument("--no_dedup", action="store_true",
                   help="Désactiver la déduplication exacte des documents.")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.txt")))
    if not files:
        raise SystemExit(f"Aucun .txt dans '{args.in_dir}'.")
    os.makedirs(args.out_dir, exist_ok=True)

    sep = "\n<|endoftext|>\n"
    total = kept = dups = 0
    kept_bytes = 0
    shard_idx = 0
    shard_bytes = 0
    seen = set()  # hash des docs déjà gardés (déduplication exacte)
    out = open(os.path.join(args.out_dir, f"shard_{shard_idx:04d}.txt"), "w", encoding="utf-8")
    t0 = time.time()

    print(f"Filtrage strict : min_strong={args.min_strong}, min_total={args.min_total}")
    print(f"{len(files)} fichiers à traiter depuis {args.in_dir}\n")

    try:
        for fi, path in enumerate(files, 1):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            for doc in content.split("<|endoftext|>"):
                doc = doc.strip()
                if not doc:
                    continue
                total += 1
                if len(doc) < args.min_chars or not keep(doc, args.min_strong, args.min_total):
                    continue
                if not args.no_dedup:
                    h = hash(doc)
                    if h in seen:
                        dups += 1
                        continue
                    seen.add(h)
                out.write(doc + sep)
                kept += 1
                kept_bytes += len(doc)
                shard_bytes += len(doc)
                if shard_bytes >= args.shard_mb * 1024**2:
                    out.close()
                    shard_idx += 1
                    shard_bytes = 0
                    out = open(os.path.join(args.out_dir, f"shard_{shard_idx:04d}.txt"),
                               "w", encoding="utf-8")
            print(f"  [{fi}/{len(files)}] {os.path.basename(path)} | "
                  f"gardés {kept:,} docs / {kept_bytes/1024**2:.0f} Mo "
                  f"(sur {total:,} lus) | {time.time()-t0:.0f}s", flush=True)
    finally:
        out.close()

    print(f"\nTerminé : {kept:,} docs gardés sur {total:,} "
          f"({100*kept/max(total,1):.1f}%), {dups:,} doublons écartés, "
          f"{kept_bytes/1024**2:.0f} Mo -> {args.out_dir}/ ({shard_idx+1} shards)")


if __name__ == "__main__":
    main()
