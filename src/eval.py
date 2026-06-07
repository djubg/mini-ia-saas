"""
eval.py — Évalue un checkpoint de façon REPRODUCTIBLE.

Deux mesures, pour comparer objectivement les versions du modèle :
  1. Perplexité sur le jeu de validation (= exp(loss moyenne)). Plus bas = mieux.
     C'est le "score" numérique du modèle, indépendant du hasard.
  2. Génération GREEDY (argmax, donc déterministe) sur une liste de prompts fixes.
     Pas de hasard => deux versions sont directement comparables.

Lance :
    python src/eval.py
    python src/eval.py --device dml --max_blocks 1000
"""

import os
import sys
import math
import pickle
import argparse

import numpy as np
import torch

# La console Windows est en cp1252 par défaut -> force l'UTF-8 pour éviter
# les UnicodeEncodeError sur le texte généré.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from config import GPTConfig
from model import GPT
from tokenizer import BPETokenizer, END_OF_TEXT
from utils import get_device, device_name

# Prompts de référence (garde-les stables pour comparer les versions dans le temps).
FIXED_PROMPTS = [
    "How to build a SaaS product",
    "Our pricing strategy is",
    "The key to reducing churn is",
    "To validate a startup idea, you should",
]


@torch.no_grad()
def perplexity(model, val_path, block_size, device, max_blocks):
    """Perplexité sur des blocs NON chevauchants du val set (déterministe)."""
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    n = len(data)
    total_loss, total = 0.0, 0
    for b, start in enumerate(range(0, n - block_size - 1, block_size)):
        if b >= max_blocks:
            break
        x = torch.from_numpy(data[start:start + block_size].astype(np.int64))[None].to(device)
        y = torch.from_numpy(data[start + 1:start + 1 + block_size].astype(np.int64))[None].to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        total += 1
    mean = total_loss / max(total, 1)
    return mean, math.exp(mean), total


@torch.no_grad()
def greedy_generate(model, tok, prompt, max_new, block_size, device, eot, rep_penalty=1.3):
    """
    Décodage glouton (argmax) : DÉTERMINISTE (donc comparable entre versions) et
    fonctionne sur tout device (DML inclus, pas de multinomial). Un petit
    repetition penalty évite les boucles tout en restant déterministe.
    """
    ids = tok.encode(prompt, allowed_special="all") or tok.encode("\n")
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None]
    for _ in range(max_new):
        cond = idx[:, -block_size:]
        logits, _ = model(cond)
        logits = logits[:, -1, :]
        if rep_penalty != 1.0:
            seen = torch.unique(idx[0])
            v = logits[0, seen]
            logits[0, seen] = torch.where(v > 0, v / rep_penalty, v * rep_penalty)
        nxt = logits.argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        if eot is not None and nxt.item() == eot:
            break
    text = tok.decode(idx[0].tolist())
    return text.split(END_OF_TEXT)[0].rstrip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--ckpt", default="ckpt.pt", help="Nom du checkpoint dans out_dir.")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--device", default="auto")
    p.add_argument("--max_blocks", type=int, default=500,
                   help="Nb de blocs du val set pour la perplexité (plus = plus précis).")
    p.add_argument("--max_new_tokens", type=int, default=80)
    args = p.parse_args()

    device = get_device(args.device)
    print(f"Device : {device_name(device)}")

    ckpt_path = os.path.join(args.out_dir, args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gcfg = GPTConfig(**checkpoint["model_args"])
    model = GPT(gcfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.tie_weights()
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Checkpoint : {args.ckpt} | {n_params/1e6:.1f}M params | "
          f"iter {checkpoint.get('iter_num','?')} | "
          f"best_val {checkpoint.get('best_val_loss', float('nan')):.4f}")

    # --- tokenizer ---
    tok = None
    meta_path = os.path.join(args.data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if os.path.exists(meta.get("tokenizer_path", "")):
            tok = BPETokenizer.load(meta["tokenizer_path"])
    if tok is None and checkpoint.get("tokenizer_json"):
        tmp = os.path.join(args.out_dir, "_tok_from_ckpt.json")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(checkpoint["tokenizer_json"])
        tok = BPETokenizer.load(tmp)
    eot = tok.special_tokens.get(END_OF_TEXT) if tok else None

    # --- 1) Perplexité ---
    val_path = os.path.join(args.data_dir, "val.bin")
    if os.path.exists(val_path):
        loss, ppl, nb = perplexity(model, val_path, gcfg.block_size, device, args.max_blocks)
        print(f"\n=== Perplexité (val, {nb} blocs) ===")
        print(f"  loss moyenne : {loss:.4f}")
        print(f"  perplexité   : {ppl:.2f}   (plus bas = mieux)")
    else:
        print("\n(val.bin introuvable — perplexité ignorée)")

    # --- 2) Générations greedy déterministes ---
    if tok is not None:
        print("\n=== Générations greedy (déterministes) ===")
        for prompt in FIXED_PROMPTS:
            out = greedy_generate(model, tok, prompt, args.max_new_tokens,
                                  gcfg.block_size, device, eot)
            print(f"\n> {prompt!r}\n  {out}")
    print()


if __name__ == "__main__":
    main()
