# CLAUDE.md — Contexte projet (à lire en premier)

Mini-GPT **from scratch** (PyTorch) spécialisé **SaaS**, à but pédagogique : comprendre
comment un LLM apprend (tokenisation → attention → entraînement → génération). Code
commenté en **français**. Projet **open-source communautaire** (MIT).

> Détails d'usage → [README.md](README.md) · prochaines étapes → [ROADMAP.md](ROADMAP.md)
> · contribuer → [CONTRIBUTING.md](CONTRIBUTING.md). **Ce fichier est la source de contexte
> portable** (la mémoire `~/.claude/` ne suit PAS si le dossier est déplacé).

## Environnement
- **`python` = scoop Python 3.11** (a torch). Toujours `python -m pip`.
- GPU = **AMD RX 6600**, pas de ROCm sous Windows. **`torch-directml` installé** →
  `--device dml` (~3× plus rapide que CPU). `import torch_directml` charge torch 2.4.1.
- **Meilleur si dispo : GPU NVIDIA + CUDA** (`--device cuda`, flash attention). Voir README.
- **Pièges DirectML** (déjà gérés, ne pas régresser) :
  - checkpoints : `map_location="cpu"` puis `.to(device)`.
  - `model.to(dml)` casse le weight tying → `model.tie_weights()` après (fait dans le code).
  - `torch.multinomial` non fiable sur DML → **génération sur CPU** (`generate.py --device cpu`).
    `eval.py` utilise du greedy (argmax) donc tourne sur DML.
  - AdamW `lerp` retombe sur CPU (warning bénin).

## Pipeline & commandes
```
collect_big_data → filter_corpus → prepare_data → train → generate / eval
```
```powershell
# Collecte 10 Go : BRUT (sans --filter_saas) = ~10× plus rapide, on filtre APRÈS.
python src/collect_big_data.py --target_gb 10 --source fineweb
python src/filter_corpus.py --in_dir data/big --out_dir data/big_saas --min_total 3
python src/prepare_data.py --retrain_tokenizer --no_big      # ignore le brut data/big
python src/train.py --device dml --max_iters 8000            # défaut ~13,8M params
python src/eval.py --device dml                              # perplexité + greedy
python src/generate.py --prompt "..." --device cpu --top_p 0.9 --repetition_penalty 1.3 --stop

# Scripts tout-en-un :
.\run_all.ps1 -Filter -Retrain -Device dml -MaxIters 8000
.\train_journee.ps1     # entraîne ~8h PUIS éval + génère, tout logué (pour run sans surveillance)
```

## Conventions de données
- `data/*.txt` + `data/big_saas/**` = corpus PROPRE (toujours tokenisé).
- `data/big/**` = brut non filtré (`--no_big` pour l'ignorer ; supprimable après filtrage).
- Docs séparés par `<|endoftext|>`, split train/val au niveau document. `.bin` = uint16 (memmap).
- **Seul `data/sample_saas.txt` est versionné** (le reste est ignoré : trop lourd / reproductible).

## État actuel (2026-06-08)
- Corpus prêt : **56 Mo SaaS propre** (10 Go FineWeb brut → filtre strict min_total=3) →
  **15,3M tokens** (train 13,8M / val 1,46M), vocab 8193 (tokenizer réentraîné).
- **Run de 8h FAIT** (`train_journee.ps1`, 40000 iters, ~655 ms/iter sur DML) → modèle
  **29,5M** (n_layer 8, n_head 8, n_embd 512, block 256). `checkpoints/ckpt.pt` est
  désormais **le bon modèle** : entraîné AVEC le tokenizer courant + tokenizer embarqué
  (l'orphelin est réglé). `ckpt_last.pt` = dernier état. Backups : `ckpt_v2_2M.pt`, `ckpt_v1_255k.pt`.
- **Résultats** : meilleure val loss **3,94** · perplexité val **79,1** (`eval.py`) · génération
  cohérente et sur-thème (greedy répète un peu ; échantillonné `top_p 0.9` + `repetition_penalty 1.3` = propre).
- Progression val loss : 98k→6,17 · 255k→5,44 · 2,24M→4,84 · **15,3M (40000 it)→3,94**.

## Outils ajoutés (session du 2026-06-07)
- Génération : `--repetition_penalty`, `--top_p`, `--stop` (corrige la répétition « SaaS »).
- `filter_corpus.py` : filtre SaaS strict (≥1 mot-clé fort + contexte, frontières de mots) + dédup.
- `collect_big_data.py` : streaming HF datasets, `--filter_saas`, `--skip_docs N` (contenu neuf).
- `prepare_data.py` : **streaming** (RAM bornée), tokenizer sur échantillon, cache d'encodage, `--no_big`.
- `train.py` : sauve `ckpt.pt` (best) + `ckpt_last.pt`, **embarque le tokenizer** dans le `.pt`.
- `src/eval.py` : perplexité + greedy déterministe (comparaison objective des versions).
- Scripts `run_all.ps1`, `train_journee.ps1`.

## Open-source / GitHub
- Dépôt git **initialisé** (branche `main`, 1ᵉʳ commit, 28 fichiers ; data/checkpoints exclus).
- Fichiers communautaires : `LICENSE` (MIT), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `.github/` (templates issues/PR), `ROADMAP.md`.
- **Pas encore poussé sur GitHub** : à faire par l'utilisateur (`gh repo create ... --push`
  ou créer le repo sur github.com puis `git remote add origin ... && git push -u origin main`).
  ⚠️ Claude n'a PAS accès au GitHub de l'utilisateur.
- ⚠️ Projet sous OneDrive → conseillé de le déplacer hors OneDrive (conflits `.git`).

## Gotchas qui font perdre du temps
- **BPE Python pur ≈ 1-2 min/Mo** à l'entraînement → `--tokenizer_sample_mb` ≤ ~10.
- **FineWeb se streame en ordre déterministe** → même collecte = mêmes docs. `--skip_docs N`
  (N = total « lus » du run précédent) pour du neuf.
- **val loss non comparable** entre deux tokenizers → comparer via `eval.py`.
- Sorties console : forcer UTF-8 (`sys.stdout.reconfigure`) sinon `UnicodeEncodeError` (cp1252).
  ⚠️ **Un pipe (`| Tee-Object`) force AUSSI cp1252** → un `print` accentué plante le run. `train.py`,
  `generate.py`, `eval.py` font tous le `reconfigure(utf-8, line_buffering=True)` (le `line_buffering`
  garde le log vivant dans un pipe, sinon stdout est bloc-bufferisé et n'affiche rien pendant des heures).
- Désactiver la **veille Windows** avant un long entraînement (sinon le GPU gèle).
