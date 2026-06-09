"""
finetune.py — Fine-tuning conversationnel : transforme le modèle BASE (ckpt.pt,
qui ne fait que CONTINUER du texte) en ASSISTANT qui RÉPOND.

C'est exactement l'étape "base -> chat" des vrais LLM (en miniature). Différences
avec train.py (le pré-entraînement) :
  - on NE part PAS de zéro : on charge ckpt.pt (tout le savoir déjà appris).
  - learning rate BAS : on AJUSTE le comportement, on ne ré-apprend pas la langue.
  - peu d'itérations : les données de chat sont petites (~5M tokens).
  - données = chat_train.bin / chat_val.bin (format `User:` / `Assistant:`).
  - on sauvegarde dans ckpt_chat.pt : le modèle de base (ckpt.pt) reste intact.

Pré-requis :  python src/prepare_chat_data.py   (génère chat_train.bin/chat_val.bin)

Lance :
    python src/finetune.py --device dml
"""

import os
import sys
import math
import time
import argparse

import torch

# Console Windows / pipe = cp1252 -> force UTF-8 (sinon UnicodeEncodeError).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

from config import GPTConfig
from model import GPT
from dataset import DataLoader
from utils import get_device, device_name, count_parameters, human_count


def get_lr(it, warmup, max_iters, lr, min_lr):
    """Warmup linéaire court puis décroissance cosinus jusqu'à min_lr."""
    if it < warmup:
        return lr * (it + 1) / (warmup + 1)
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup) / max(1, max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model, loader, eval_iters):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = loader.get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="checkpoints/ckpt.pt",
                   help="Modèle de base à fine-tuner (le pré-entraîné).")
    p.add_argument("--out", default="checkpoints/ckpt_chat.pt",
                   help="Où sauvegarder le modèle conversationnel.")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--device", default="dml")
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-4,
                   help="Bas (fine-tuning). Le pré-entraînement utilisait 3e-4.")
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_iters", type=int, default=80)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    print(f"Device : {device_name(device)}")

    # --- charge le modèle de base ---
    if not os.path.exists(args.init):
        raise FileNotFoundError(
            f"{args.init} introuvable. Entraîne d'abord le modèle de base (train.py)."
        )
    print(f"Modèle de base : {args.init}")
    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    gcfg = GPTConfig(**ckpt["model_args"])
    model = GPT(gcfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.tie_weights()  # rétablit le weight tying cassé par .to() (DirectML)
    print(f"Modèle : {human_count(count_parameters(model))} paramètres "
          f"(block_size={gcfg.block_size})")

    # tokenizer embarqué -> ckpt_chat.pt autonome (réutilise celui du base).
    tok_json = ckpt.get("tokenizer_json")
    if tok_json is None:
        tpath = os.path.join(args.data_dir, "tokenizer.json")
        if os.path.exists(tpath):
            with open(tpath, "r", encoding="utf-8") as f:
                tok_json = f.read()

    # --- données de chat ---
    loader = DataLoader(args.data_dir, gcfg.block_size, args.batch_size, device,
                        train_name="chat_train.bin", val_name="chat_val.bin")

    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (0.9, 0.95)
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_val = float("inf")
    saved = False
    print(f"\nFine-tuning : {args.max_iters} itérations "
          f"(batch effectif = {args.batch_size * args.grad_accum})\n")

    t0 = time.time()
    running = None
    for it in range(args.max_iters + 1):
        lr = get_lr(it, args.warmup_iters, args.max_iters, args.learning_rate, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0:
            losses = estimate_loss(model, loader, args.eval_iters)
            print(f"[iter {it:>5}] train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | lr {lr:.2e}")
            if it > 0 and losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "model_args": vars(gcfg),
                    "iter_num": it,
                    "best_val_loss": best_val,
                    "tokenizer_json": tok_json,
                }, args.out)
                saved = True
                print(f"         -> sauvegarde ({args.out}, val {best_val:.4f})")
        if it == args.max_iters:
            break

        # --- une étape d'optimisation (accumulation de gradient) ---
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = loader.get_batch("train")
            _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        lossf = loss.item()
        running = lossf if running is None else 0.9 * running + 0.1 * lossf
        if it % args.log_interval == 0 and it > 0:
            dt = time.time() - t0
            print(f"  iter {it:>5} | loss {lossf:.4f} (moy {running:.4f}) "
                  f"| {dt * 1000 / args.log_interval:.0f} ms/iter")
            t0 = time.time()

    # filet de sécurité : garantit un ckpt_chat.pt même si la val n'a jamais baissé.
    if not saved:
        torch.save({
            "model": model.state_dict(), "model_args": vars(gcfg),
            "iter_num": args.max_iters, "best_val_loss": best_val,
            "tokenizer_json": tok_json,
        }, args.out)

    print(f"\nFine-tuning terminé. Meilleure val : {best_val:.4f}")
    print(f"Checkpoint : {args.out}")
    print("\nDiscute avec :  python src/chat.py")


if __name__ == "__main__":
    main()
