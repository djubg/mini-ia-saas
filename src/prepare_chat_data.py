"""
prepare_chat_data.py — Prépare des données de CONVERSATION pour le fine-tuning
(passage base -> assistant).

Idée : un modèle "brut" (pré-entraîné) ne fait que CONTINUER du texte. Pour qu'il
RÉPONDE, on le ré-entraîne sur des paires question/réponse formatées toujours
pareil, pour qu'il apprenne le "rôle" assistant et où s'arrêter :

    User: <question>
    Assistant: <réponse><|endoftext|>

On réutilise le TOKENIZER EXISTANT (data/tokenizer.json) — indispensable pour
rester compatible avec ckpt.pt (mêmes ids, même vocab 8193). On NE retoken­ise
PAS un nouveau tokenizer. `<|endoftext|>` (déjà connu du modèle) sert de fin de
réponse : à l'inférence on s'arrête dessus.

Sortie : data/chat_train.bin et data/chat_val.bin (uint16, comme prepare_data).

Installe (si besoin) :  python -m pip install datasets

Lance :
    python src/prepare_chat_data.py
    python src/prepare_chat_data.py --max_examples 2000      # test rapide
    python src/prepare_chat_data.py --dataset tatsu-lab/alpaca
"""

import os
import sys
import random
import argparse

import numpy as np

# Console Windows / pipe = cp1252 -> force UTF-8 (sinon UnicodeEncodeError).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

from tokenizer import BPETokenizer, END_OF_TEXT


def format_example(ex):
    """Une paire instruction/réponse -> texte au format conversation.

    Renvoie None si l'exemple est inexploitable (vide).
    Le format DOIT être identique partout pour que le modèle l'apprenne.
    """
    instruction = (ex.get("instruction") or "").strip()
    output = (ex.get("output") or "").strip()
    context = (ex.get("input") or "").strip()  # champ optionnel (Alpaca)
    if not instruction or not output:
        return None
    if context:
        prompt = f"User: {instruction}\n{context}"
    else:
        prompt = f"User: {instruction}"
    return f"{prompt}\nAssistant: {output}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="tatsu-lab/alpaca",
                   help="Dataset d'instructions HuggingFace (champs instruction/input/output).")
    p.add_argument("--data_dir", default="data",
                   help="Doit contenir tokenizer.json (le tokenizer de ckpt.pt).")
    p.add_argument("--max_examples", type=int, default=0,
                   help="Limite le nb d'exemples (0 = tout). Utile pour tester vite.")
    p.add_argument("--val_frac", type=float, default=0.05,
                   help="Fraction d'exemples réservés à la validation.")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    # --- tokenizer existant (compatibilité ckpt.pt) ---
    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(
            f"{tok_path} introuvable. Lance d'abord prepare_data.py (le fine-tuning "
            "doit utiliser le MÊME tokenizer que le modèle pré-entraîné)."
        )
    tok = BPETokenizer.load(tok_path)
    if END_OF_TEXT not in tok.special_tokens:
        raise ValueError(f"Le tokenizer n'a pas le token spécial {END_OF_TEXT}.")
    eot_id = tok.special_tokens[END_OF_TEXT]
    print(f"Tokenizer : {tok.n_vocab} tokens (eot={eot_id}).")

    # --- dataset d'instructions ---
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Installe d'abord :  python -m pip install datasets")
    print(f"Chargement de {args.dataset} ...")
    ds = load_dataset(args.dataset, split="train")

    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)
    if args.max_examples and args.max_examples < len(indices):
        indices = indices[: args.max_examples]

    n_val = int(len(indices) * args.val_frac)
    val_set = set(indices[:n_val])
    print(f"Exemples : {len(indices)} (train {len(indices) - n_val} / val {n_val}).")

    train_ids, val_ids = [], []
    kept = skipped = 0
    for i in indices:
        text = format_example(ds[i])
        if text is None:
            skipped += 1
            continue
        # encode + termine la réponse par <|endoftext|> (signal d'arrêt).
        ids = tok.encode(text, allowed_special="all")
        ids.append(eot_id)
        (val_ids if i in val_set else train_ids).extend(ids)
        kept += 1
        if kept % 5000 == 0:
            print(f"  {kept} exemples encodés...")

    print(f"Gardés {kept}, ignorés {skipped} (vides).")
    _write_bin(train_ids, os.path.join(args.data_dir, "chat_train.bin"))
    _write_bin(val_ids, os.path.join(args.data_dir, "chat_val.bin"))
    print(f"\nTokens : train {len(train_ids):,} / val {len(val_ids):,}.")
    print("Prochaine étape :  python src/finetune.py")


def _write_bin(ids, path):
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path)
    print(f"  {os.path.basename(path)} : {len(arr):,} tokens ({arr.nbytes / 1024**2:.1f} Mo)")


if __name__ == "__main__":
    main()
