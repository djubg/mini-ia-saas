"""
prepare_data.py — Transforme le corpus texte en tokens prêts pour l'entraînement.

Version STREAMING : conçue pour passer de quelques Mo à plusieurs Go sans saturer
la RAM. On traite le corpus document par document et on écrit les tokens dans les
fichiers binaires au fur et à mesure (jamais tout le corpus en mémoire d'un coup).

Pipeline :
  1. Liste tous les .txt de data/ ET des shards data/big/**/*.txt.
  2. Entraîne le tokenizer BPE sur un ÉCHANTILLON (par défaut 100 Mo) — entraîner
     le BPE sur des Go serait trop lent en Python pur ; un échantillon suffit.
  3. Encode chaque document en streaming (avec cache de chunks) et écrit les
     tokens uint16 dans train.bin / val.bin (split au niveau document).
  4. Écrit meta.pkl.

Lance :
    python src/prepare_data.py
    python src/prepare_data.py --vocab_size 4096
    python src/prepare_data.py --retrain_tokenizer --tokenizer_sample_mb 50
"""

import os
import glob
import pickle
import argparse

import numpy as np

from tokenizer import BPETokenizer, END_OF_TEXT


def list_input_files(data_dir: str, include_big: bool = True):
    """Tous les .txt d'entrée : data/*.txt + shards data/big/**/*.txt (triés).

    include_big=False ignore data/big/ (utile si ce dossier contient du brut
    non filtré qu'on ne veut pas tokeniser).
    """
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    # data/big_saas/ = sortie FILTRÉE (propre) : toujours incluse.
    files += glob.glob(os.path.join(data_dir, "big_saas", "**", "*.txt"), recursive=True)
    # data/big/ = brut non filtré : inclus sauf si --no_big.
    if include_big:
        files += glob.glob(os.path.join(data_dir, "big", "**", "*.txt"), recursive=True)
    files = sorted(set(os.path.abspath(p) for p in files))
    if not files:
        raise FileNotFoundError(
            f"Aucun .txt trouvé dans '{data_dir}' (ni dans '{data_dir}/big/').\n"
            "Ajoute du texte SaaS puis relance."
        )
    return files


def iter_documents(files, sep=END_OF_TEXT):
    """
    Génère les documents un par un (streaming).

    On lit un FICHIER à la fois (mémoire bornée par la taille d'un shard, ~100 Mo)
    et on le découpe sur le séparateur de documents. Pour 10 Go répartis en shards
    de 100 Mo, le pic mémoire reste ~100-200 Mo.
    """
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        for doc in content.split(sep):
            doc = doc.strip()
            if doc:
                yield doc


def read_sample(files, max_bytes: int) -> str:
    """Concatène du texte jusqu'à ~max_bytes, pour entraîner le tokenizer."""
    parts, total = [], 0
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        parts.append(content)
        total += len(content)
        if total >= max_bytes:
            break
    text = "\n".join(parts)
    return text[:max_bytes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--vocab_size", type=int, default=8192,
                        help="Taille du vocabulaire BPE (>= 256).")
    parser.add_argument("--retrain_tokenizer", action="store_true",
                        help="Forcer le réentraînement du tokenizer.")
    parser.add_argument("--tokenizer_sample_mb", type=int, default=5,
                        help="Taille de l'échantillon (Mo) pour entraîner le BPE. "
                             "Attention : le BPE Python pur coûte ~1-2 min/Mo, donc "
                             "garde-le petit (5-10 Mo suffisent pour un bon vocab).")
    parser.add_argument("--val_every", type=int, default=10,
                        help="1 document sur N va dans la validation (10 = ~10%).")
    parser.add_argument("--no_big", action="store_true",
                        help="Ignorer data/big/ (ex: s'il contient du brut non filtré).")
    args = parser.parse_args()

    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")

    files = list_input_files(data_dir, include_big=not args.no_big)
    total_bytes = sum(os.path.getsize(p) for p in files)
    print(f"Corpus : {len(files)} fichier(s), {total_bytes / 1024**2:.1f} Mo.")

    # --- 1) Tokenizer : entraîner (sur échantillon) ou recharger ---
    if os.path.exists(tokenizer_path) and not args.retrain_tokenizer:
        print(f"Tokenizer existant chargé depuis {tokenizer_path}")
        tok = BPETokenizer.load(tokenizer_path)
    else:
        sample_bytes = args.tokenizer_sample_mb * 1024**2
        print(f"Échantillon pour le tokenizer : ~{args.tokenizer_sample_mb} Mo")
        sample = read_sample(files, sample_bytes).replace(END_OF_TEXT, "\n")
        print(f"Entraînement du tokenizer BPE (vocab_size={args.vocab_size})...")
        tok = BPETokenizer()
        tok.train(sample, vocab_size=args.vocab_size, verbose=True)
        eot_id = len(tok.vocab)
        tok.register_special_tokens({END_OF_TEXT: eot_id})
        tok.save(tokenizer_path)
        del sample
        print(f"Tokenizer sauvegardé dans {tokenizer_path}")

    vocab_size = tok.n_vocab
    print(f"Taille du vocabulaire final : {vocab_size}")
    if vocab_size > 65536:
        raise ValueError("vocab_size > 65536 : incompatible avec le stockage uint16.")
    eot_id = tok.special_tokens[END_OF_TEXT]

    # --- 2) Encodage en streaming -> train.bin / val.bin ---
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    print("Encodage du corpus en streaming...")

    n_train = n_val = n_docs = 0
    with open(train_path, "wb") as ftrain, open(val_path, "wb") as fval:
        for i, doc in enumerate(iter_documents(files)):
            ids = tok.encode_ordinary(doc)
            ids.append(eot_id)  # marque la fin du document
            arr = np.asarray(ids, dtype=np.uint16)
            if i % args.val_every == 0:
                arr.tofile(fval)
                n_val += arr.size
            else:
                arr.tofile(ftrain)
                n_train += arr.size
            n_docs += 1
            if n_docs % 2000 == 0:
                done = n_train + n_val
                print(f"  {n_docs:,} docs | {done:,} tokens "
                      f"({done * 2 / 1024**2:.0f} Mo de .bin)")

    total_tokens = n_train + n_val
    meta = {"vocab_size": vocab_size, "tokenizer_path": tokenizer_path}
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    ratio = total_bytes / max(total_tokens, 1)
    print(f"\nTerminé : {n_docs:,} documents, {total_tokens:,} tokens "
          f"(~{ratio:.2f} car/token).")
    print(f"  train.bin : {n_train:,} tokens")
    print(f"  val.bin   : {n_val:,} tokens")
    print(f"  meta.pkl  : vocab_size={vocab_size}")
    print("\nProchaine étape :  python src/train.py")


if __name__ == "__main__":
    main()
