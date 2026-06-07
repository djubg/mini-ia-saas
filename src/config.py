"""
config.py — Tous les hyperparamètres du projet au même endroit.

Deux dataclasses :
  - GPTConfig    : l'architecture du modèle (taille, profondeur...).
  - TrainConfig  : tout ce qui concerne l'entraînement (lr, batch, etc.).

Pour faire varier ton modèle, tu modifies UNIQUEMENT ce fichier.
Les valeurs par défaut sont calibrées pour une machine modeste
(CPU + 16 Go RAM) et donnent un modèle d'environ ~15M de paramètres.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- Taille du vocabulaire ---
    # Renseignée automatiquement par prepare_data.py à partir du tokenizer.
    # On met une valeur par défaut ici ; elle sera écrasée au chargement.
    vocab_size: int = 8192

    # --- Contexte ---
    # block_size = nombre de tokens que le modèle "voit" en même temps.
    # Plus c'est grand, plus il capte de contexte, mais plus c'est lourd.
    block_size: int = 128

    # --- Profondeur et largeur du Transformer ---
    n_layer: int = 6     # nombre de blocs Transformer empilés
    n_head: int = 6      # nombre de têtes d'attention (doit diviser n_embd)
    n_embd: int = 384    # dimension des embeddings (la "largeur" du modèle)

    # --- Régularisation ---
    dropout: float = 0.1   # 0.0 si peu de données et beaucoup d'epochs
    bias: bool = False     # biais dans les Linear/LayerNorm ? False = plus léger

    # --- Attention ---
    # scaled_dot_product_attention (flash) est rapide mais pas toujours
    # supporté par DirectML. False => implémentation manuelle, lisible et
    # portable (idéale pour COMPRENDRE l'attention). Passe à True sur CUDA.
    use_flash_attention: bool = False


@dataclass
class TrainConfig:
    # --- Données ---
    data_dir: str = "data"           # contient train.bin / val.bin / meta.pkl
    out_dir: str = "checkpoints"     # où sauvegarder le modèle

    # --- Boucle d'entraînement ---
    batch_size: int = 16             # séquences par batch (baisse si OOM)
    gradient_accumulation_steps: int = 4  # batch effectif = batch_size * cela
    max_iters: int = 5000            # nombre total d'itérations
    eval_interval: int = 250         # fréquence d'évaluation
    eval_iters: int = 100            # nb de batchs pour estimer la loss
    log_interval: int = 10           # fréquence d'affichage de la loss train

    # --- Optimiseur (AdamW) ---
    learning_rate: float = 3e-4      # lr max
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0           # clipping du gradient (0 = désactivé)

    # --- Scheduler (cosine decay avec warmup) ---
    decay_lr: bool = True
    warmup_iters: int = 200
    lr_decay_iters: int = 5000       # ~= max_iters
    min_lr: float = 3e-5             # ~= learning_rate / 10

    # --- Divers ---
    seed: int = 1337
    device: str = "auto"             # "auto" | "cpu" | "cuda" | "dml"
    compile: bool = False            # torch.compile (laisse False sur Windows/AMD)
    always_save_checkpoint: bool = False  # sauver même si val loss n'a pas baissé
