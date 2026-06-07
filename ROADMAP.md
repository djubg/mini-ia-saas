# Feuille de route — Mini IA SaaS

État au 2026-06-07 : modèle 13,8M entraîné sur 6,29M tokens SaaS propres (FineWeb
filtré). Le modèle « parle SaaS » mais reste peu cohérent et sur-utilise le mot
« SaaS ». Voir [README.md](README.md) pour le détail du pipeline.

Checkpoints sauvegardés : `ckpt.pt` (actuel 13,8M), `ckpt_v2_2M.pt`, `ckpt_v1_255k.pt`.

---

## ✅ IMPLÉMENTÉ le 2026-06-07 (améliorations faites & testées)

- [x] **Génération** : `--repetition_penalty`, `--top_p` (nucleus), `--stop` (arrêt sur
  `<|endoftext|>`). → corrige le « SaaS SaaS SaaS ».
- [x] **filter_corpus** : déduplication exacte des documents (`--no_dedup` pour désactiver).
- [x] **collect_big_data** : `--skip_docs N` (contenu neuf malgré le flux déterministe).
- [x] **train** : sauve `ckpt.pt` (meilleur val) **+** `ckpt_last.pt` (dernier état), et
  **embarque le tokenizer** dans le `.pt` (checkpoint autonome).
- [x] **`src/eval.py`** : perplexité du val set + génération **greedy déterministe** sur
  prompts fixes (comparaison objective entre versions). Tourne sur DML.
- [x] **`run_all.ps1`** : pipeline filtre→tokenise→entraîne→évalue en une commande.
- [x] **`CLAUDE.md`** : contexte projet pour reprise rapide.
- *DML par défaut* : déjà le cas (`config.device = "auto"` détecte DirectML).

Reste (volontairement non fait) : **tokenizer Rust** (tiktoken/HF) — contredit le "from
scratch" et lourd ; à n'envisager que pour des dizaines de Go. Shuffle avant split : peu
utile car l'entraînement échantillonne déjà des fenêtres aléatoires.

---

## 🎯 À FAIRE LA PROCHAINE FOIS (expériences à lancer)

Par ordre d'impact. Chaque ligne = une expérience concrète.

- [ ] **1. Modèle plus gros + contexte plus long** (le plus gros gain de cohérence)
  ```powershell
  python src/prepare_data.py --no_big   # ré-encode avec le contexte voulu si besoin
  python src/train.py --device dml --n_layer 8 --n_head 8 --n_embd 512 --block_size 256 --max_iters 12000
  ```
  → plus lent sur DML (~500-800 ms/iter), mais nettement plus cohérent. Lancer le soir.

- [ ] **2. Entraînement plus long** sur le modèle actuel ou le gros
  `--max_iters 15000` à `20000`. Le val loss descendait encore un peu à la fin.

- [ ] **3. Réduire la répétition « SaaS »** (deux options, cumulables)
  - À la génération : ajouter un *repetition penalty* / *top-p* (voir Améliorations §Génération).
  - Dans les données : diluer avec ~20-30 % de texte business NON filtré (collecte
    `c4`/`fineweb` sans `--filter_saas`) pour casser le biais de sélection.

- [ ] **4. Plus de données SaaS FRAÎCHES**
  ⚠️ FineWeb se streame dans un ordre **déterministe** : relancer `--target_gb 2`
  redonne les **mêmes** docs. Pour du nouveau, soit `--target_gb 5` (les Go 2→5 sont
  neufs), soit ajouter un `--skip_docs` (voir Améliorations §Données). Puis :
  ```powershell
  python src/filter_corpus.py --in_dir data/big --out_dir data/big_saas
  python src/prepare_data.py --retrain_tokenizer --no_big
  python src/train.py --device dml --max_iters 10000
  ```

- [ ] **5. Tester un filtre moins strict** pour plus de volume
  `python src/filter_corpus.py --in_dir data/big --out_dir data/big_saas --min_total 3`
  → ~2-3× plus de docs, encore acceptables (≥1 mot-clé SaaS fort requis).

- [ ] **6. Mesurer vraiment les progrès** : se fixer un petit jeu de prompts SaaS de
  référence et comparer les générations entre versions (pas seulement le val loss).

---

## 🛠️ AMÉLIORATIONS / AJOUTS (code & outillage)

### Génération (`src/generate.py`)
- [ ] **Repetition penalty** : pénaliser les tokens déjà générés (corrige « SaaS SaaS… »).
- [ ] **Top-p (nucleus sampling)** en plus du top-k, pour un meilleur équilibre.
- [ ] Option **`--stop`** sur `<|endoftext|>` pour couper proprement à la fin d'un doc.

### Tokenizer (`src/tokenizer.py`)
- [ ] **Tokenizer rapide (Rust)** : passer à `tiktoken` ou HuggingFace `tokenizers`
  pour l'entraînement BPE. Le BPE Python pur coûte ~1-2 min/Mo → bloquant au-delà de
  ~10-20 Mo d'échantillon. Indispensable pour tokeniser des dizaines de Go.

### Données (`collect_big_data.py`, `filter_corpus.py`)
- [ ] **`--skip_docs N`** dans `collect_big_data` : sauter les N premiers docs du flux
  pour récupérer du contenu NEUF à chaque run (contourne le déterminisme FineWeb).
- [ ] **Déduplication** des documents (hash de chunk) : éviter les quasi-doublons qui
  font mémoriser au lieu de généraliser.
- [ ] **Mélange de sources** pondéré (ex. 70 % FineWeb-SaaS + 30 % Wikipédia business).
- [ ] **Shuffle** des documents avant le split train/val (actuellement = ordre des shards).

### Entraînement (`src/train.py`)
- [ ] **Batch effectif plus grand** via `--grad_accum 4` (gradients plus stables sur
  gros modèle) — déjà supporté, juste à utiliser.
- [ ] **Gestion des checkpoints** : nommer automatiquement par version + garder le
  meilleur val loss séparément du dernier.
- [ ] **Sauver le tokenizer avec le checkpoint** (pour qu'un `ckpt.pt` soit autonome).
- [ ] **Reprise propre** : tester `--resume` après le passage au gros modèle.

### Pipeline & confort
- [ ] **Script tout-en-un** (`run_all.ps1` ou Makefile) : collect → filter → prepare →
  train → generate, en une commande.
- [ ] **`--device dml` par défaut** dans la config si DirectML est détecté (au lieu de CPU).
- [ ] **Petit harnais d'éval** : perplexité sur le val set + génération sur prompts fixes,
  loggés à chaque version.

---

## 🧭 Cap à plus long terme (rappel de la vision)
Une fois un modèle de base correct : **fine-tuning** (style/instructions), puis **RAG**
(base de connaissances SaaS), puis **agent SaaS** avec outils. Voir README §"Pour aller
plus loin".
