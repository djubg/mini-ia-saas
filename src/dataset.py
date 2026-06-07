"""
dataset.py — Chargement efficace des tokens et échantillonnage de batchs.

On stocke les tokens dans des fichiers .bin et on les lit via np.memmap :
le fichier reste sur le disque et seules les fenêtres nécessaires sont
chargées en RAM. Indispensable quand le corpus devient gros.
"""

import os
import numpy as np
import torch


def _load_bin(path):
    """Ouvre un .bin en lecture mémoire-mappée (uint16)."""
    return np.memmap(path, dtype=np.uint16, mode="r")


class DataLoader:
    """
    Donne des batchs aléatoires (x, y) où :
      x = fenêtre de `block_size` tokens.
      y = la même fenêtre décalée d'un token (la cible à prédire).
    """

    def __init__(self, data_dir, block_size, batch_size, device):
        train_path = os.path.join(data_dir, "train.bin")
        val_path = os.path.join(data_dir, "val.bin")
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"{train_path} introuvable. Lance d'abord : python src/prepare_data.py"
            )
        self.train_data = _load_bin(train_path)
        self.val_data = _load_bin(val_path) if os.path.exists(val_path) else self.train_data
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self, split: str):
        data = self.train_data if split == "train" else self.val_data
        # positions de départ aléatoires.
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([
            torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix
        ])

        # transfert vers le device. pin_memory + non_blocking aide sur CUDA.
        if str(self.device).startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
