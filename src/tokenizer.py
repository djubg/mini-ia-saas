"""
tokenizer.py — Un tokenizer BPE (Byte Pair Encoding) écrit FROM SCRATCH.

Pourquoi BPE ?
  Un LLM ne lit pas du texte, il lit des nombres (tokens). Le BPE est la
  méthode utilisée par GPT : on part des octets bruts (0-255), puis on
  fusionne itérativement les paires de symboles les plus fréquentes pour
  créer un vocabulaire compact qui capture les morceaux de mots récurrents
  ("ing", "tion", "SaaS", " the"...).

Cette implémentation est volontairement simple et commentée pour COMPRENDRE.
Elle s'inspire de minbpe (Andrej Karpathy), version pédagogique.

Étapes :
  1. train()  : apprend les fusions à partir d'un texte.
  2. encode() : texte  -> liste d'entiers (tokens).
  3. decode() : liste d'entiers -> texte.
  4. save()/load() : persiste le tokenizer sur disque (JSON).
"""

import json
import regex as re


# Pattern de découpage type GPT-2 : empêche le BPE de fusionner à travers
# les frontières "naturelles" (espaces, ponctuation, chiffres...).
GPT_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Token spécial pour séparer les documents dans le corpus.
END_OF_TEXT = "<|endoftext|>"


def get_stats(ids, counts=None):
    """Compte la fréquence de chaque paire consécutive dans une liste d'ids."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """Remplace toutes les occurrences de `pair` par le nouvel id `idx`."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    def __init__(self):
        # merges : (id_a, id_b) -> nouvel_id. L'ordre = priorité de fusion.
        self.merges = {}
        # vocab : id -> bytes. Construit à partir des merges.
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        # tokens spéciaux : str -> id (au-dessus du vocab BPE).
        self.special_tokens = {}
        self.pattern = GPT_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        # cache d'encodage : chunk (str) -> liste de tokens. Le texte réel répète
        # énormément les mêmes mots, donc ce cache rend l'encodage quasi-linéaire
        # (indispensable pour encoder des Go). Reconstruit si les merges changent.
        self._chunk_cache = {}

    # ------------------------------------------------------------------ #
    #  Entraînement
    # ------------------------------------------------------------------ #
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """
        Apprend les fusions BPE jusqu'à atteindre `vocab_size`.

        Optimisation clé : l'anglais répète énormément les mêmes mots. Plutôt
        que de re-scanner tout le flux de tokens à chaque fusion (très lent),
        on regroupe les chunks IDENTIQUES et on compte les paires pondérées par
        leur fréquence. Le résultat est strictement identique, mais on traite
        ~7× moins de données (le nombre de mots uniques, pas total).
        """
        assert vocab_size >= 256, "vocab_size doit être >= 256 (les octets de base)"
        num_merges = vocab_size - 256

        # 1) découpe en chunks, puis regroupe les identiques avec leur fréquence.
        from collections import Counter
        chunk_freq = Counter(re.findall(self.compiled_pattern, text))
        # chunks : liste de [liste_d_octets, fréquence]
        chunks = [[list(ch.encode("utf-8")), f] for ch, f in chunk_freq.items()]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        # 2) fusionne la paire la plus fréquente, num_merges fois.
        for i in range(num_merges):
            # comptage des paires sur les chunks uniques, pondéré par fréquence.
            stats = {}
            for ids, freq in chunks:
                for pair in zip(ids, ids[1:]):
                    stats[pair] = stats.get(pair, 0) + freq
            if not stats:
                break  # plus rien à fusionner
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # applique la fusion (on saute les chunks trop courts).
            for c in chunks:
                if len(c[0]) >= 2:
                    c[0] = merge(c[0], pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and (i < 5 or (i + 1) % 500 == 0):
                print(
                    f"  merge {i + 1}/{num_merges}: {pair} -> {idx} "
                    f"({vocab[idx]!r}) freq={stats[pair]}"
                )

        self.merges = merges
        self.vocab = vocab
        self._chunk_cache = {}  # les merges ont changé : on invalide le cache

    def register_special_tokens(self, tokens):
        """Enregistre des tokens spéciaux, ex: {'<|endoftext|>': 8192}."""
        self.special_tokens = dict(tokens)

    # ------------------------------------------------------------------ #
    #  Encodage / Décodage
    # ------------------------------------------------------------------ #
    def _encode_chunk(self, text_bytes):
        """Encode une suite d'octets en appliquant les fusions par priorité."""
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            # paire dont la fusion a la plus petite priorité (= apprise tôt)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # plus aucune fusion applicable
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str):
        """Encode du texte SANS gérer les tokens spéciaux (avec cache de chunks)."""
        cache = self._chunk_cache
        ids = []
        for chunk in re.findall(self.compiled_pattern, text):
            cached = cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk.encode("utf-8"))
                cache[chunk] = cached
            ids.extend(cached)
        return ids

    def encode(self, text: str, allowed_special: str = "all"):
        """
        Encode du texte EN gérant les tokens spéciaux.
        allowed_special : "all" pour reconnaître les tokens spéciaux, "none" sinon.
        """
        special = self.special_tokens if allowed_special == "all" else {}
        if not special:
            return self.encode_ordinary(text)

        # découpe le texte autour des tokens spéciaux pour les préserver.
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        chunks = re.split(special_pattern, text)
        ids = []
        for part in chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        """Reconstruit le texte à partir d'une liste de tokens."""
        inverse_special = {v: k for k, v in self.special_tokens.items()}
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in inverse_special:
                part_bytes.append(inverse_special[idx].encode("utf-8"))
            else:
                raise ValueError(f"Token id invalide : {idx}")
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")

    @property
    def n_vocab(self):
        """Taille totale du vocabulaire = BPE + tokens spéciaux."""
        return len(self.vocab) + len(self.special_tokens)

    # ------------------------------------------------------------------ #
    #  Sauvegarde / Chargement (JSON)
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        data = {
            "pattern": self.pattern,
            # merges sérialisés en liste : [[a, b, idx], ...]
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.pattern = data["pattern"]
        tok.compiled_pattern = re.compile(tok.pattern)
        tok.merges = {(a, b): idx for a, b, idx in data["merges"]}
        tok.special_tokens = {k: int(v) for k, v in data["special_tokens"].items()}
        # reconstruit le vocab à partir des merges (ordre croissant d'idx)
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (a, b), idx in sorted(tok.merges.items(), key=lambda kv: kv[1]):
            vocab[idx] = vocab[a] + vocab[b]
        tok.vocab = vocab
        return tok
