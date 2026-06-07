"""
fetch_data.py — Récupère un corpus SaaS depuis Wikipédia (anglais).

Pourquoi Wikipédia ?
  - Contenu librement réutilisable (licence CC BY-SA).
  - API publique, texte propre, gros volume disponible.
  - Couvre tout le champ : SaaS, cloud, startups, produit, pricing,
    marketing, growth, ventes, métriques...

Le script :
  1. part d'une liste de titres pertinents (SEED_TITLES) ;
  2. l'élargit avec les membres de catégories pertinentes (CATEGORIES) ;
  3. télécharge le texte brut de chaque article (un par requête) ;
  4. nettoie (retire les sections References / See also / External links...) ;
  5. écrit le tout dans data/wikipedia_saas.txt, articles séparés par <|endoftext|>.

Usage :
    python src/fetch_data.py
    python src/fetch_data.py --max_articles 400 --out data/wikipedia_saas.txt

Utilisation responsable : sources publiques librement réutilisables, requêtes
espacées, User-Agent descriptif. Ne pas augmenter le débit de façon abusive.
"""

import os
import re
import time
import json
import threading
import argparse
import urllib.parse
import urllib.request
import concurrent.futures

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "MiniSaaSLLM/0.1 (educational from-scratch LLM project; contact: local user)"

# --- Titres "graines" : concepts coeur du domaine SaaS ---
SEED_TITLES = [
    "Software as a service", "Platform as a service", "Infrastructure as a service",
    "Cloud computing", "Multitenancy", "Web application", "Application programming interface",
    "Microservices", "DevOps", "Continuous integration", "Continuous delivery",
    "Service-level agreement", "Enterprise software", "Software industry",
    "Startup company", "Lean startup", "Minimum viable product", "Product–market fit",
    "Business model", "Subscription business model", "Freemium", "Business-to-business",
    "Pricing", "Pricing strategies", "Value proposition", "Go to market",
    "Customer relationship management", "Customer success", "Customer retention",
    "Churn rate", "Customer lifetime value", "Customer acquisition cost",
    "Growth hacking", "Network effect", "Viral marketing",
    "Content marketing", "Inbound marketing", "Digital marketing", "Marketing automation",
    "Email marketing", "Affiliate marketing", "Search engine optimization", "Marketing",
    "Sales", "Conversion rate optimization", "A/B testing", "Landing page",
    "Key performance indicator", "Net promoter score", "Performance indicator",
    "Venture capital", "Seed money", "Angel investor", "Bootstrapping",
    "Product management", "Agile software development", "Scrum (software development)",
    "User experience", "Usability", "Customer experience",
    "Software as a service business model", "Recurring billing", "Onboarding",
    "Y Combinator", "Unicorn (finance)", "Entrepreneurship",
]

# --- Catégories : pages membres (concepts ciblés, pas d'entreprises) ---
# On évite les catégories trop larges (ex: "Marketing") qui dérivent hors sujet.
CATEGORIES = [
    "Software as a service",
    "Cloud computing",
    "Cloud platforms",
    "Web applications",
    "Software distribution",
    "Business models",
    "Business software",
    "Pricing",
    "Product management",
    "Project management software",
    "Online marketing",
    "Digital marketing",
    "Marketing techniques",
    "Sales",
    "E-commerce",
    "Entrepreneurship",
    "Customer experience",
    "Management",
    "Software development process",
    "Online services",
]

# Sections de fin à couper (pas de la prose utile).
STOP_SECTIONS = {
    "see also", "references", "notes", "external links", "further reading",
    "bibliography", "citations", "sources", "footnotes", "works cited",
}


def api_get(params, max_retries=5):
    """Appel API avec retry + backoff exponentiel (gère le HTTP 429)."""
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = 3.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                # respecte Retry-After si présent, sinon backoff exponentiel.
                wait = e.headers.get("Retry-After")
                wait = float(wait) if (wait and wait.isdigit()) else delay
                print(f"    429 (trop de requêtes) — pause {wait:.0f}s puis retry...")
                time.sleep(wait)
                delay *= 2
                continue
            raise
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                print(f"    réseau ({e}) — pause {delay:.0f}s puis retry...")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("api_get: échec après plusieurs tentatives")


def get_category_members(category, limit):
    """Renvoie les titres des PAGES (cmtype=page) d'une catégorie."""
    titles = []
    cont = None
    while len(titles) < limit:
        params = {
            "action": "query", "list": "categorymembers",
            "cmtitle": f"Category:{category}", "cmlimit": "500", "cmtype": "page",
        }
        if cont:
            params["cmcontinue"] = cont
        try:
            d = api_get(params)
        except Exception as e:
            print(f"  ! catégorie '{category}' : {e}")
            break
        for m in d.get("query", {}).get("categorymembers", []):
            titles.append(m["title"])
        cont = d.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        time.sleep(0.1)
    return titles[:limit]


def fetch_extract(title):
    """Texte brut intégral d'un article (None si page manquante)."""
    d = api_get({
        "action": "query", "prop": "extracts", "explaintext": "1",
        "redirects": "1", "titles": title,
    })
    pages = d.get("query", {}).get("pages", {})
    if not pages:
        return None
    p = next(iter(pages.values()))
    if "missing" in p:
        return None
    return p.get("extract", "")


def clean_extract(text):
    """Coupe les sections de fin, retire le balisage '==', normalise."""
    if not text:
        return ""
    lines = text.split("\n")
    out = []
    for line in lines:
        header = re.match(r"^\s*(=+)\s*(.*?)\s*\1\s*$", line)
        if header:
            name = header.group(2).strip().lower()
            if name in STOP_SECTIONS:
                break  # on arrête : tout ce qui suit est du boilerplate
            out.append(header.group(2).strip())  # garde le titre comme ligne de texte
        else:
            out.append(line)
    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # max 2 sauts de ligne
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/wikipedia_saas.txt")
    parser.add_argument("--max_articles", type=int, default=600,
                        help="Nombre maximum d'articles à récupérer.")
    parser.add_argument("--per_category", type=int, default=100,
                        help="Nombre max de pages par catégorie.")
    parser.add_argument("--min_chars", type=int, default=300,
                        help="Ignore les articles plus courts que ça (stubs).")
    parser.add_argument("--workers", type=int, default=3,
                        help="Requêtes simultanées. Garde bas (2-3) : au-delà, "
                             "Wikipédia rate-limite et impose 59s de pause par 429.")
    parser.add_argument("--full", action="store_true",
                        help="Texte INTÉGRAL au lieu des intros (plus complet mais "
                             "beaucoup plus lent : 1 requête par article).")
    args = parser.parse_args()

    # 1) construit la liste de titres (graines + catégories), dédupliquée.
    print("Collecte des titres...")
    titles = list(SEED_TITLES)
    seen = set(t.lower() for t in titles)
    for cat in CATEGORIES:
        members = get_category_members(cat, args.per_category)
        added = 0
        for t in members:
            if t.lower() not in seen:
                seen.add(t.lower())
                titles.append(t)
                added += 1
        print(f"  catégorie '{cat}': +{added} titres")
        if len(titles) >= args.max_articles:
            break
    titles = titles[:args.max_articles]
    print(f"{len(titles)} titres à télécharger.\n")

    # 2) télécharge + nettoie les articles.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lock = threading.Lock()
    progress = {"done": 0}
    docs = []

    if args.full:
        # --- mode LENT : texte intégral, 1 requête par article ---
        def fetch_full(title):
            try:
                cleaned = clean_extract(fetch_extract(title))
                ok = len(cleaned) >= args.min_chars
            except Exception:
                return None
            with lock:
                progress["done"] += 1
                print(f"  [{progress['done']}/{len(titles)}] {title!r} : "
                      f"{('%d car.' % len(cleaned)) if ok else 'ignoré'}", flush=True)
            return cleaned if ok else None

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for doc in ex.map(fetch_full, titles):
                if doc:
                    docs.append(doc)
    else:
        # --- mode RAPIDE : intros par lots de 20 articles / requête ---
        chunks = [titles[i:i + 20] for i in range(0, len(titles), 20)]
        n_chunks = len(chunks)

        def fetch_chunk(chunk):
            try:
                d = api_get({
                    "action": "query", "prop": "extracts", "explaintext": "1",
                    "exintro": "1", "exlimit": "20", "redirects": "1",
                    "titles": "|".join(chunk),
                })
                pages = d.get("query", {}).get("pages", {})
                extracts = [p.get("extract", "") for p in pages.values()
                            if "missing" not in p]
            except Exception as e:
                extracts = []
                with lock:
                    print(f"    erreur sur un lot : {e}", flush=True)
            with lock:
                progress["done"] += 1
                print(f"  lot {progress['done']}/{n_chunks} "
                      f"(+{len(extracts)} articles)", flush=True)
            return extracts

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for extracts in ex.map(fetch_chunk, chunks):
                for raw in extracts:
                    cleaned = clean_extract(raw)
                    if len(cleaned) >= args.min_chars:
                        docs.append(cleaned)

    total_chars = sum(len(d) for d in docs)

    # 3) écrit le corpus.
    sep = "\n<|endoftext|>\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(sep.join(docs))

    print(f"\nTerminé : {len(docs)} articles, {total_chars:,} caractères "
          f"(~{total_chars / 1_000_000:.2f} Mo) -> {args.out}")
    print("\nProchaines étapes :")
    print("  python src/prepare_data.py --retrain_tokenizer")
    print("  python src/train.py")


if __name__ == "__main__":
    main()
