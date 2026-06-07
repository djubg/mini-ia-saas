# Mini IA SaaS — un mini-GPT entraîné *from scratch*

Un petit modèle de langage (type GPT) entraîné **à partir de zéro** en PyTorch,
spécialisé sur le domaine **SaaS** (startup, produit, pricing, growth, marketing).

> Le but n'est **pas** de rivaliser avec ChatGPT. Le but est de **comprendre
> comment un LLM apprend réellement** : tokenisation, attention, entraînement,
> génération — en lisant et en exécutant un code simple et commenté.

---

## Ce que fait le projet

Le modèle apprend **une seule chose** : prédire le prochain token.

```
Input :  "How to build a SaaS product"
Output : "you need to validate your idea, define your MVP, ..."
```

De cette tâche répétée des milliers de fois émergent les régularités du
langage SaaS présent dans les données.

---

## Architecture du code

```
IA FOR SaaS/
├── README.md
├── requirements.txt
├── data/
│   └── sample_saas.txt      # corpus SaaS de démarrage (modifiable)
├── src/
│   ├── config.py            # TOUS les hyperparamètres (à éditer ici)
│   ├── tokenizer.py         # tokenizer BPE écrit from scratch
│   ├── model.py             # le mini-GPT (Transformer) from scratch
│   ├── dataset.py           # chargement des tokens + batchs
│   ├── prepare_data.py      # texte -> tokens (train.bin / val.bin)
│   ├── train.py             # boucle d'entraînement
│   ├── generate.py          # génération de texte (inférence)
│   └── utils.py             # détection du device (CPU / DirectML)
└── checkpoints/             # modèles sauvegardés (créé à l'entraînement)
```

---

## Installation

Tu as plusieurs versions de Python ; ce projet utilise **`python` = Python 3.11**
(scoop). Vérifie avec `python --version`.

```powershell
python -m pip install -r requirements.txt
```

### GPU AMD (RX 6600) sous Windows — à lire

⚠️ **Important** : ROCm (le CUDA d'AMD) **ne supporte pas Windows** pour la
RX 6600. Les seules options réalistes sous Windows sont :

| Option        | Comment                                          | Vitesse |
|---------------|--------------------------------------------------|---------|
| **CPU**       | par défaut, rien à faire                          | lente mais OK pour apprendre |
| **DirectML**  | `python -m pip install torch-directml`           | accélère via le GPU AMD |

Le code **détecte automatiquement** DirectML s'il est installé
(voir `src/utils.py`). Sinon il tourne sur CPU. Pour de meilleures performances
GPU, l'alternative serait un dual-boot Linux + ROCm, mais ce n'est pas
nécessaire pour ce projet d'apprentissage.

### Sur une autre machine — GPU NVIDIA (ex. RTX 5060) ✅ recommandé si dispo

Une carte **NVIDIA est nettement meilleure** que DirectML : CUDA est le backend
natif de PyTorch → plus rapide, plus stable, et **flash attention** disponible
(gros gain à `block_size 256`). Le code est déjà compatible (il tente CUDA en
premier ; les bidouilles DirectML sont inoffensives sur CUDA).

Mise en place sur la machine NVIDIA :
```bash
# 1) PyTorch CUDA. RTX 50xx (Blackwell) exige torch récent (>=2.7) + CUDA 12.8 :
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy regex tqdm datasets

# 2) vérifier que le GPU est vu :
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
- **Copier les données** déjà préparées (rapide, ~30 Mo) : `data/train.bin`,
  `data/val.bin`, `data/meta.pkl`, `data/tokenizer.json`. (Ou re-collecter sur place.)
- **Activer flash attention** : dans `src/config.py`, mettre `use_flash_attention = True`.
- **Entraîner** (CUDA gère tout, y compris l'échantillonnage) :
  ```bash
  python src/train.py --device cuda --n_layer 8 --n_head 8 --n_embd 512 --block_size 256 --max_iters 40000
  python src/generate.py --prompt "How to build a SaaS"   # pas besoin de --device cpu sur CUDA
  ```
- Avec un 5060, on peut viser **plus gros** (n_layer 10-12, n_embd 640-768) et/ou
  un `block_size 512` — l'entraînement sera plus rapide qu'ici.

---

## Utilisation — 3 étapes

### 1. Préparer les données (texte → tokens)

```powershell
python src/prepare_data.py
```

Cela entraîne le tokenizer BPE, encode le corpus et crée `data/train.bin`,
`data/val.bin`, `data/meta.pkl`, `data/tokenizer.json`.

Options :
```powershell
python src/prepare_data.py --vocab_size 4096   # vocabulaire plus petit
```

### 2. Entraîner le modèle

```powershell
python src/train.py
```

Affiche la loss qui doit **descendre** au fil des itérations. Le modèle est
sauvegardé dans `checkpoints/ckpt.pt`.

Pour un test rapide sur CPU :
```powershell
python src/train.py --max_iters 500 --device cpu
```

Reprendre un entraînement interrompu :
```powershell
python src/train.py --resume
```

### 3. Générer du texte

```powershell
python src/generate.py --prompt "How to build a SaaS product"
```

Options utiles :
```powershell
python src/generate.py --prompt "SaaS pricing strategy:" `
    --max_new_tokens 300 --temperature 0.8 --top_k 40 --num_samples 3
```

- `temperature` : `<1` = prudent/cohérent, `>1` = créatif/aléatoire.
- `top_k` : ne tire que parmi les *k* tokens les plus probables.

---

## Régler la taille du modèle

Tout se passe dans [`src/config.py`](src/config.py). Valeurs par défaut
(~15M paramètres, adaptées à une machine modeste) :

| Paramètre     | Défaut | Effet |
|---------------|--------|-------|
| `n_layer`     | 6      | profondeur (nombre de blocs Transformer) |
| `n_head`      | 6      | nombre de têtes d'attention |
| `n_embd`      | 384    | largeur du modèle |
| `block_size`  | 128    | longueur de contexte (tokens vus à la fois) |
| `vocab_size`  | auto   | défini par le tokenizer |

**Si l'entraînement est trop lent ou manque de RAM**, réduis :
`--n_layer 4 --n_embd 256 --block_size 64 --batch_size 8`.

**Modèle minuscule pour tester en 2 minutes sur CPU** :
```powershell
python src/train.py --n_layer 2 --n_head 2 --n_embd 128 --block_size 64 --max_iters 500
```

---

## Améliorer les résultats : plus de données

Avec le petit corpus de démarrage, le modèle va surtout **mémoriser**
(overfit). C'est normal et pédagogique. Pour de vrais résultats, ajoute du
texte SaaS dans `data/` (un ou plusieurs fichiers `.txt`) :

- articles de blog SaaS, guides startup / product management ;
- documentation produit, pages pricing, posts de fondateurs ;
- vise plusieurs Mo de texte propre (plus = mieux).

Place tes `.txt` dans `data/`, puis relance :
```powershell
python src/prepare_data.py --retrain_tokenizer
python src/train.py
```

> Collecte les données de manière responsable : utilise des sources dont tu as
> le droit d'usage (contenu public, tes propres écrits, datasets ouverts).

### Petite collecte (quelques Mo) — Wikipédia

[`src/fetch_data.py`](src/fetch_data.py) récupère des articles Wikipédia sur le
domaine SaaS/business :
```powershell
python src/fetch_data.py --max_articles 1500   # mode rapide (intros par lots de 20)
python src/fetch_data.py --full                 # texte intégral (plus lent, rate-limité)
```

---

## Passer à l'échelle du Go (2 Go → 10 Go)

À partir de ~100 Mo, **on ne scrape plus article par article** (trop lent,
rate-limit). On télécharge des **datasets pré-construits en streaming** avec la
librairie HuggingFace `datasets`. Script fourni :
[`src/collect_big_data.py`](src/collect_big_data.py).

```powershell
python -m pip install datasets zstandard
python src/collect_big_data.py --target_gb 2 --source wikipedia
python src/collect_big_data.py --target_gb 2 --source fineweb --filter_saas
```

Sources dispo : `wikipedia` (propre), `c4` / `fineweb` / `openwebtext` (web,
volume énorme). `--filter_saas` ne garde que les documents au vocabulaire
SaaS/business. La sortie va dans `data/big/shard_XXXX.txt`.

### ⚠️ Disque

| Texte | ≈ Tokens | Fichiers `.bin` (uint16) | Espace libre conseillé |
|-------|----------|--------------------------|------------------------|
| 2 Go  | ~0,4 Md  | ~0,8 Go                  | ~5 Go                  |
| 10 Go | ~2 Md    | ~4 Go                    | ~20 Go                 |

### Le pipeline est prêt pour le Go ✅

`prepare_data.py` fonctionne désormais **en streaming** : après `collect_big_data.py`,
lance simplement :
```powershell
python src/prepare_data.py --retrain_tokenizer
```
Il détecte automatiquement les shards `data/big/**/*.txt`. Ce qui est en place :

1. **Streaming** — encodage document par document, écriture des tokens dans le
   `.bin` au fil de l'eau. La mémoire reste bornée (~taille d'un shard), donc 2-10 Go
   passent sans saturer la RAM.
2. **Tokenizer sur échantillon** — le BPE s'entraîne sur les premiers
   `--tokenizer_sample_mb` Mo (défaut **5**) puis s'applique à tout le corpus.
   ⚠️ Le BPE en Python pur coûte **~1-2 min/Mo** : garde l'échantillon petit
   (5-10 Mo suffisent pour un bon vocabulaire). C'est un coût unique.
3. **Cache d'encodage** — `tokenizer.py` mémoïse l'encodage par chunk ; comme le
   texte réel répète beaucoup les mêmes mots, l'encodage est quasi-linéaire.
4. **`dataset.py`** lit le `.bin` en `memmap` : le corpus reste sur le disque
   pendant l'entraînement. ✅

Options utiles : `--tokenizer_sample_mb` (taille de l'échantillon BPE),
`--val_every` (1 doc sur N en validation).

> **Limite restante** : pour des dizaines de Go, le BPE Python pur (même mémoïsé)
> finira par être lent. À ce stade, basculer sur un tokenizer Rust
> (`tiktoken` ou HuggingFace `tokenizers`) sera la prochaine optimisation.

### ⚠️ Réalité matérielle (RX 6600)

Collecter 2-10 Go est faisable, mais **les *utiliser* pleinement demande un plus
gros modèle et beaucoup de temps de calcul**. Sur la RX 6600 via DirectML
(~100 ms/iter pour le modèle 5,3M), un vrai entraînement sur plusieurs Go se
compte en heures/jours. Stratégie réaliste : collecter le corpus maintenant, et
n'en tokeniser/entraîner qu'un **sous-ensemble** (ex. 200-500 Mo) à la fois,
en augmentant progressivement la taille du modèle.

---

## Comment ça marche (les concepts clés)

1. **Tokenisation (BPE)** — le texte est découpé en tokens (morceaux de mots)
   convertis en entiers. Voir [`src/tokenizer.py`](src/tokenizer.py).
2. **Embeddings** — chaque token devient un vecteur ; on y ajoute un embedding
   de position pour encoder l'ordre.
3. **Self-attention causale** — chaque token regarde les tokens précédents pour
   décider de ce qui compte. Voir `CausalSelfAttention` dans
   [`src/model.py`](src/model.py).
4. **Blocs Transformer** — attention + réseau feed-forward, empilés N fois.
5. **Entraînement** — on minimise la *cross-entropy* entre la prédiction et le
   vrai token suivant, avec AdamW. Voir [`src/train.py`](src/train.py).
6. **Génération** — on prédit un token, on l'ajoute au contexte, on recommence.

---

## Pour aller plus loin

Ce projet est une **base réelle** pour évoluer vers :

- **Fine-tuning** : partir d'un modèle pré-entraîné et le spécialiser.
- **RAG** (Retrieval-Augmented Generation) : brancher une base de connaissances.
- **Agent SaaS** : ajouter des outils et de la mémoire.

---

## 🤝 Contribuer / Communauté

Projet **open-source et communautaire** (licence MIT) — débutants bienvenus !
- Comment contribuer → [CONTRIBUTING.md](CONTRIBUTING.md)
- Idées & améliorations prévues → [ROADMAP.md](ROADMAP.md)
- Partage un modèle que tu as entraîné → ouvre une issue « 🚀 Partage ton modèle »

Quelques façons d'aider : nouveaux filtres/sources de données, support CUDA/ROCm/Mac,
meilleures options de génération, traductions, tutoriels, ou simplement entraîner un
modèle et partager tes résultats (perplexité + exemples).

## Crédits

Architecture du modèle inspirée de **nanoGPT** (Andrej Karpathy) et du
tokenizer **minbpe**, réécrits ici en version pédagogique et commentée en
français. Projet communautaire — voir les [contributeurs](../../graphs/contributors).
