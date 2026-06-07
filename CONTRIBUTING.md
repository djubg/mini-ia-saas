# Contribuer à Mini IA SaaS 🤝

Merci de l'intérêt ! Ce projet est **éducatif et communautaire** : un mini-GPT
entraîné *from scratch* pour comprendre comment un LLM fonctionne, spécialisé SaaS.
Toutes les contributions sont les bienvenues — code, données, docs, idées.

## Philosophie (à respecter)
- **From scratch & lisible** : le but est de *comprendre*. On garde le code simple et
  bien commenté plutôt que clever. On évite les grosses dépendances qui cachent la magie
  (ex. : on n'utilise pas un tokenizer tout-fait à la place du BPE maison).
- **Tourne sur une machine modeste** : CPU, ou GPU grand public (DirectML/CUDA). Pas besoin
  d'un cluster.
- **Commentaires en français OU anglais** acceptés (le code existant est en français ;
  les traductions sont les bienvenues).

## Mise en place (dev)
```bash
python -m pip install -r requirements.txt
# GPU optionnel : torch-directml (AMD/Windows) ou torch CUDA (NVIDIA)
python src/prepare_data.py        # tokenise le corpus d'exemple
python src/train.py --max_iters 500   # mini entraînement de test
python src/eval.py                # vérifie que rien n'est cassé
```

## Par où commencer
- **[ROADMAP.md](ROADMAP.md)** liste les améliorations prévues (idéal pour une 1ʳᵉ PR).
- Cherche les issues étiquetées **`good first issue`**.
- Petites idées rapides : nouveaux mots-clés pour `filter_corpus`, nouvelles sources dans
  `collect_big_data`, options de génération, tests, docs.

## Process de contribution
1. **Fork** le repo, crée une branche : `git checkout -b feature/ma-feature`.
2. Fais des changements **focalisés** (une idée par PR).
3. Vérifie que ça tourne : `python src/eval.py` (pas de régression de perplexité grossière).
4. Décris clairement ta PR (le *pourquoi*, pas que le *quoi*).
5. Ouvre la **Pull Request** vers `main`.

## Domaines où aider
| Domaine | Exemples |
|---------|----------|
| **Données** | meilleurs filtres SaaS, nouvelles sources, déduplication avancée, datasets partagés |
| **Tokenizer** | option tokenizer rapide, BPE multilingue |
| **Modèle** | variantes d'attention, RoPE, normalisation, tailles |
| **Entraînement** | schedulers, mixed precision, reprise, logging (W&B/TensorBoard) |
| **Génération** | beam search, contraintes, meilleurs réglages |
| **Hardware** | support CUDA / ROCm / Apple MPS, benchmarks |
| **Docs / i18n** | traductions EN, tutoriels, schémas |
| **Modèles partagés** | publie ton checkpoint + sa perplexité (voir template d'issue) |

## Ce qu'on NE committe pas
Les **données** (`data/`, sauf `sample_saas.txt`) et les **checkpoints** (`*.pt`) sont
trop lourds et reproductibles → ils sont dans `.gitignore`. Partage plutôt les **scripts**
qui les génèrent (ou un lien externe pour un modèle).

## Code de conduite
Sois respectueux et bienveillant — voir [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
