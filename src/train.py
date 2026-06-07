"""
train.py — Entraîne le mini-GPT sur les tokens préparés.

Boucle d'entraînement classique :
  pour chaque itération :
    1. tire un batch (x, y)
    2. forward : calcule les prédictions et la loss (cross-entropy)
    3. backward : calcule les gradients
    4. step : met à jour les poids (AdamW)
  périodiquement : évalue sur train+val et sauvegarde le meilleur modèle.

Lance :
    python src/train.py
    python src/train.py --max_iters 2000 --device cpu
"""

import os
import time
import math
import pickle
import argparse
from contextlib import nullcontext

import torch

from config import GPTConfig, TrainConfig
from model import GPT
from dataset import DataLoader
from utils import get_device, device_name, count_parameters, human_count


def get_lr(it, cfg: TrainConfig):
    """Learning rate avec warmup linéaire puis décroissance cosinus."""
    if not cfg.decay_lr:
        return cfg.learning_rate
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, loader, eval_iters):
    """Estime la loss moyenne sur train et val (mode eval = pas de dropout)."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = loader.get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def parse_args():
    p = argparse.ArgumentParser()
    # surcharges courantes en ligne de commande (sinon valeurs de config.py)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--max_iters", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--n_embd", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--grad_accum", type=int, default=None,
                   help="étapes d'accumulation de gradient (1 = plus réactif).")
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--device", default=None, help="auto | cpu | cuda | dml")
    p.add_argument("--resume", action="store_true", help="reprendre depuis un checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    gcfg = GPTConfig()
    tcfg = TrainConfig()

    # applique les surcharges CLI
    for name in ["block_size", "n_layer", "n_head", "n_embd"]:
        if getattr(args, name) is not None:
            setattr(gcfg, name, getattr(args, name))
    for name in ["data_dir", "out_dir", "max_iters", "batch_size",
                 "learning_rate", "eval_interval", "device"]:
        if getattr(args, name) is not None:
            setattr(tcfg, name, getattr(args, name))
    if args.grad_accum is not None:
        tcfg.gradient_accumulation_steps = args.grad_accum

    # aligne la décroissance du learning rate sur la durée réelle d'entraînement.
    tcfg.lr_decay_iters = tcfg.max_iters
    tcfg.warmup_iters = min(tcfg.warmup_iters, max(1, tcfg.max_iters // 20))

    torch.manual_seed(tcfg.seed)

    # --- device ---
    device = get_device(tcfg.device)
    print(f"Device : {device_name(device)}")

    # --- vocab_size depuis meta.pkl ---
    meta_path = os.path.join(tcfg.data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            "meta.pkl introuvable. Lance d'abord : python src/prepare_data.py"
        )
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    gcfg.vocab_size = meta["vocab_size"]
    print(f"vocab_size = {gcfg.vocab_size}")

    # --- données ---
    loader = DataLoader(tcfg.data_dir, gcfg.block_size, tcfg.batch_size, device)

    # --- modèle ---
    os.makedirs(tcfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(tcfg.out_dir, "ckpt.pt")        # meilleur val loss
    ckpt_last_path = os.path.join(tcfg.out_dir, "ckpt_last.pt")  # dernier état
    # on embarque le tokenizer dans le checkpoint -> .pt autonome.
    tok_json_path = os.path.join(tcfg.data_dir, "tokenizer.json")
    tokenizer_json = None
    if os.path.exists(tok_json_path):
        with open(tok_json_path, "r", encoding="utf-8") as f:
            tokenizer_json = f.read()
    iter_num = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(ckpt_path):
        print(f"Reprise depuis {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        gcfg = GPTConfig(**checkpoint["model_args"])
        model = GPT(gcfg)
        model.load_state_dict(checkpoint["model"])
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    else:
        model = GPT(gcfg)

    model.to(device)
    model.tie_weights()  # rétablit le partage de poids cassé par .to() (DirectML)
    n_params = count_parameters(model)
    print(f"Modèle : {human_count(n_params)} paramètres "
          f"(n_layer={gcfg.n_layer}, n_head={gcfg.n_head}, n_embd={gcfg.n_embd}, "
          f"block_size={gcfg.block_size})")

    optimizer = model.configure_optimizers(
        tcfg.weight_decay, tcfg.learning_rate, (tcfg.beta1, tcfg.beta2)
    )
    if args.resume and os.path.exists(ckpt_path):
        optimizer.load_state_dict(checkpoint["optimizer"])

    if tcfg.compile and hasattr(torch, "compile"):
        print("torch.compile(model)...")
        model = torch.compile(model)

    # --- boucle d'entraînement ---
    print(f"\nDébut de l'entraînement : {tcfg.max_iters} itérations "
          f"(batch effectif = {tcfg.batch_size * tcfg.gradient_accumulation_steps})\n")
    t0 = time.time()
    running_loss = None

    while iter_num <= tcfg.max_iters:
        # ajuste le learning rate
        lr = get_lr(iter_num, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # évaluation + sauvegarde périodique
        if iter_num % tcfg.eval_interval == 0:
            losses = estimate_loss(model, loader, tcfg.eval_iters)
            print(f"[iter {iter_num:>5}] train loss {losses['train']:.4f} | "
                  f"val loss {losses['val']:.4f} | lr {lr:.2e}")
            if iter_num > 0:
                raw_model = getattr(model, "_orig_mod", model)
                is_best = losses["val"] < best_val_loss
                best_val_loss = min(losses["val"], best_val_loss)
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": vars(gcfg),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "val_loss": losses["val"],
                    "tokenizer_json": tokenizer_json,  # rend le checkpoint autonome
                }
                # toujours : dernier état (reprise / comparaison).
                torch.save(checkpoint, ckpt_last_path)
                # meilleur val loss : le checkpoint "officiel" (chargé par generate.py).
                if is_best or tcfg.always_save_checkpoint:
                    torch.save(checkpoint, ckpt_path)
                    print(f"         -> meilleur checkpoint ({ckpt_path}, val {losses['val']:.4f})")

        if iter_num == tcfg.max_iters:
            break

        # --- une étape d'optimisation avec accumulation de gradient ---
        optimizer.zero_grad(set_to_none=True)
        for micro in range(tcfg.gradient_accumulation_steps):
            x, y = loader.get_batch("train")
            _, loss = model(x, y)
            loss = loss / tcfg.gradient_accumulation_steps
            loss.backward()

        if tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        # log
        lossf = loss.item() * tcfg.gradient_accumulation_steps
        running_loss = lossf if running_loss is None else 0.9 * running_loss + 0.1 * lossf
        if iter_num % tcfg.log_interval == 0 and iter_num > 0:
            dt = time.time() - t0
            print(f"  iter {iter_num:>5} | loss {lossf:.4f} "
                  f"(moy {running_loss:.4f}) | {dt * 1000 / tcfg.log_interval:.0f} ms/iter")
            t0 = time.time()

        iter_num += 1

    print(f"\nEntraînement terminé. Meilleure val loss : {best_val_loss:.4f}")
    print(f"Checkpoint : {ckpt_path}")
    print("\nGénère du texte avec :  python src/generate.py --prompt \"How to build a SaaS\"")


if __name__ == "__main__":
    main()
