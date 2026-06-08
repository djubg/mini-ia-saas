"""
generate.py — Génère du texte avec le modèle entraîné (inférence).

Lance :
    python src/generate.py --prompt "How to build a SaaS product"
    python src/generate.py --prompt "Pricing strategy:" --max_new_tokens 300 \
        --temperature 0.8 --top_k 40 --top_p 0.9 --repetition_penalty 1.3 --stop
"""

import os
import sys
import pickle
import argparse

import torch

# La console Windows est en cp1252 par défaut, et un pipe (ex. `| Tee-Object`)
# force aussi cp1252 -> force l'UTF-8 pour éviter les UnicodeEncodeError sur le
# texte généré accentué. line_buffering=True : flush à chaque ligne même en pipe.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

from config import GPTConfig
from model import GPT
from tokenizer import BPETokenizer
from utils import get_device, device_name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="How to build a SaaS product",
                   help="Texte de départ pour la génération.")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="<1 = prudent, >1 = créatif.")
    p.add_argument("--top_k", type=int, default=40,
                   help="Ne tire que parmi les k tokens les plus probables (0 = désactivé).")
    p.add_argument("--top_p", type=float, default=0.0,
                   help="Nucleus sampling : garde la masse de proba cumulée <= p "
                        "(ex. 0.9 ; 0 = désactivé).")
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="Pénalise les tokens déjà générés (>1, ex. 1.3) contre les boucles.")
    p.add_argument("--stop", action="store_true",
                   help="Arrêter la génération à la fin du premier document (<|endoftext|>).")
    p.add_argument("--num_samples", type=int, default=1, help="Nombre de générations.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device(args.device)
    print(f"Device : {device_name(device)}\n")

    # --- charge le checkpoint ---
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} introuvable. Entraîne d'abord : python src/train.py"
        )
    # on charge toujours sur CPU puis on déplace le modèle : évite un bug de
    # map_location avec DirectML, et c'est robuste quel que soit le device.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gcfg = GPTConfig(**checkpoint["model_args"])
    model = GPT(gcfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.tie_weights()  # rétablit le partage de poids cassé par .to() (DirectML)
    model.eval()

    # --- charge le tokenizer : depuis data/, sinon depuis le checkpoint (autonome) ---
    tok = None
    meta_path = os.path.join(args.data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        tpath = meta.get("tokenizer_path")
        if tpath and os.path.exists(tpath):
            tok = BPETokenizer.load(tpath)
    if tok is None and checkpoint.get("tokenizer_json"):
        tmp = os.path.join(args.out_dir, "_tok_from_ckpt.json")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(checkpoint["tokenizer_json"])
        tok = BPETokenizer.load(tmp)
    if tok is None:
        raise FileNotFoundError(
            "Tokenizer introuvable (ni dans data/, ni embarqué dans le checkpoint)."
        )

    # --- encode le prompt ---
    start_ids = tok.encode(args.prompt, allowed_special="all")
    if len(start_ids) == 0:
        start_ids = tok.encode("\n")
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    top_p = args.top_p if args.top_p and args.top_p > 0 else None
    from tokenizer import END_OF_TEXT
    eot_token = tok.special_tokens.get(END_OF_TEXT) if args.stop else None

    for s in range(args.num_samples):
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=args.repetition_penalty,
            eot_token=eot_token,
        )
        text = tok.decode(y[0].tolist())
        if args.stop:
            text = text.split(END_OF_TEXT)[0].rstrip()  # coupe au 1er doc
        print(f"===== Génération {s + 1}/{args.num_samples} =====")
        print(text)
        print()


if __name__ == "__main__":
    main()
