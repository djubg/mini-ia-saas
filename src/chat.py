"""
chat.py — La "fenêtre de chat" : discute avec le modèle conversationnel.

Charge ckpt_chat.pt (le modèle fine-tuné par finetune.py), met ta question au
format appris (`User: ... / Assistant: ...`) et génère la réponse jusqu'au
token `<|endoftext|>`.

⚠️ Rappel honnête : c'est un modèle de 29,5M params. Il a appris à RÉPONDRE (le
format, le ton), mais ses connaissances sont limitées -> réponses souvent
courtes/approximatives. C'est un "chatbot jouet", fait avec le vrai pipeline.

⚠️ Mono-tour : chaque message est traité indépendamment (le modèle a été
fine-tuné sur des paires isolées, pas sur des conversations à mémoire).

Sur DirectML, torch.multinomial est peu fiable -> on génère sur CPU par défaut.

Lance :
    python src/chat.py
    python src/chat.py --temperature 0.7 --top_p 0.9
Commandes : /quit pour sortir.
"""

import os
import sys
import pickle
import argparse

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

from config import GPTConfig
from model import GPT
from tokenizer import BPETokenizer, END_OF_TEXT
from utils import get_device, device_name


def load_model_and_tokenizer(ckpt_path, data_dir, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} introuvable. Lance d'abord :  python src/finetune.py"
        )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gcfg = GPTConfig(**checkpoint["model_args"])
    model = GPT(gcfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.tie_weights()
    model.eval()

    # tokenizer : depuis data/, sinon embarqué dans le checkpoint.
    tok = None
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        tpath = meta.get("tokenizer_path")
        if tpath and os.path.exists(tpath):
            tok = BPETokenizer.load(tpath)
    if tok is None and checkpoint.get("tokenizer_json"):
        tmp = os.path.join(os.path.dirname(ckpt_path), "_tok_from_ckpt.json")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(checkpoint["tokenizer_json"])
        tok = BPETokenizer.load(tmp)
    if tok is None:
        raise FileNotFoundError("Tokenizer introuvable (ni data/, ni checkpoint).")
    return model, tok, gcfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_chat.pt")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--device", default="cpu",
                   help="cpu conseillé (multinomial bugué sur DML).")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device(args.device)
    print(f"Device : {device_name(device)}")
    model, tok, gcfg = load_model_and_tokenizer(args.ckpt, args.data_dir, device)
    eot = tok.special_tokens.get(END_OF_TEXT)
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    top_p = args.top_p if args.top_p and args.top_p > 0 else None

    print(f"Modèle prêt ({os.path.basename(args.ckpt)}). Tape ta question, /quit pour sortir.\n")

    while True:
        try:
            user = input("Toi > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("/quit", "/exit", "/q"):
            break

        # format appris au fine-tuning. On laisse le modèle compléter après "Assistant: ".
        prompt = f"User: {user}\nAssistant: "
        ids = tok.encode(prompt, allowed_special="all")
        # garde le contexte dans la limite du modèle.
        ids = ids[-gcfg.block_size:]
        x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

        y = model.generate(
            x, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=args.repetition_penalty, eot_token=eot,
        )
        new_ids = y[0].tolist()[len(ids):]
        reply = tok.decode(new_ids)
        # nettoie : coupe au 1er endoftext, et si le modèle enchaîne un faux tour.
        reply = reply.split(END_OF_TEXT)[0]
        reply = reply.split("\nUser:")[0].split("User:")[0].strip()
        if not reply:
            reply = "(pas de réponse — modèle encore faible, réessaie ou reformule)"
        print(f"Bot > {reply}\n")


if __name__ == "__main__":
    main()
