# CLAUDE.md — Contexte projet (à lire en premier)

Mini-GPT **from scratch** (PyTorch) spécialisé **SaaS**, à but pédagogique : comprendre
comment un LLM apprend (tokenisation → attention → entraînement → génération). Pas de
prétention à rivaliser avec ChatGPT. Tout le code est commenté en **français**.

> Pour les détails d'usage → [README.md](README.md). Pour les prochaines étapes →
> [ROADMAP.md](ROADMAP.md). Ce fichier = le strict nécessaire pour reprendre vite.

## Environnement (IMPORTANT)
- **`python` = scoop Python 3.11** (a torch). `pip` système pointe vers un autre Python →
  toujours `python -m pip`.
- GPU = **AMD RX 6600**. Pas de ROCm sous Windows. **`torch-directml` est installé** →
  entraîner avec `--device dml` (~3× plus rapide que CPU). `import torch_directml` charge
  **torch 2.4.1** (le CPU pur est en 2.12.0).
- **Pièges DirectML** (déjà gérés dans le code, ne pas régresser) :
  - charger les checkpoints avec `map_location="cpu"` puis `.to(device)`.
  - `model.to(dml)` **casse le weight tying** → toujours `model.tie_weights()` après.
  - `torch.multinomial` non fiable sur DML → **génération sur CPU** (`generate.py --device cpu`).
    `eval.py` contourne via décodage greedy (argmax), donc tourne sur DML.
  - AdamW `lerp` retombe sur CPU (warning bénin).

## Pipeline & commandes
```
collect_big_data.py  →  filter_corpus.py  →  prepare_data.py  →  train.py  →  generate.py / eval.py
```
```powershell
# tout-en-un (filtre data/big, réentraîne le tokenizer, entraîne sur DML, évalue)
.\run_all.ps1 -Filter -Retrain -Device dml -MaxIters 8000

# étapes manuelles
python src/collect_big_data.py --target_gb 2 --source fineweb --filter_saas   # data/big (web SaaS-ish, BRUIT)
python src/filter_corpus.py --in_dir data/big --out_dir data/big_saas         # SaaS propre (strict)
python src/prepare_data.py --retrain_tokenizer --no_big                        # tokenise (ignore data/big brut)
python src/train.py --device dml --max_iters 8000                              # entraîne (défaut 13,8M params)
python src/eval.py --device dml                                                # perplexité + greedy
python src/generate.py --prompt "..." --device cpu --top_p 0.9 --repetition_penalty 1.3 --stop
```

## Conventions de données
- `data/*.txt` + `data/big_saas/**` = corpus PROPRE (toujours tokenisé).
- `data/big/**` = brut non filtré (`--no_big` pour l'ignorer ; supprimable après filtrage).
- Documents séparés par `<|endoftext|>`. Split train/val au niveau document.
- `.bin` = tokens uint16 (memmap). Tokenizer BPE **from scratch** dans `tokenizer.py`
  (avec cache d'encodage + entraînement sur échantillon).

## État actuel (2026-06-07)
- Modèle courant `checkpoints/ckpt.pt` : **13,8M params** (n_layer 6, n_head 6, n_embd 384,
  block 128), entraîné sur **6,29M tokens** SaaS (FineWeb filtré strict, 23,5 Mo).
  best val ≈ 4.89, **perplexité ≈ 122**.
- Backups : `ckpt_v2_2M.pt` (5,3M / 2,24M tokens), `ckpt_v1_255k.pt`. `ckpt_last.pt` = dernier état.
- Le modèle "parle SaaS" mais **sur-utilise « SaaS »** (biais du filtrage par mots-clés) →
  atténué par `--repetition_penalty` à la génération.

## Gotchas qui font perdre du temps
- **BPE Python pur ≈ 1-2 min/Mo** à l'entraînement → garder `--tokenizer_sample_mb` ≤ ~10.
  (Optimisation future : tokenizer Rust tiktoken/HF — volontairement NON fait pour rester
  "from scratch".)
- **FineWeb se streame en ordre déterministe** → relancer la même collecte redonne les mêmes
  docs. Utiliser `collect_big_data.py --skip_docs N` (N = total "lus" du run précédent) pour
  du contenu neuf.
- Le **val loss n'est pas comparable** entre deux tokenizers différents → comparer via
  `eval.py` (perplexité + greedy sur prompts fixes) sur le MÊME tokenizer.
- Sorties console : forcer l'UTF-8 (`sys.stdout.reconfigure`) sinon `UnicodeEncodeError` (cp1252).

## Mémoire persistante
Contexte additionnel dans `~/.claude/projects/.../memory/` (environment.md, project.md).
