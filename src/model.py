"""
model.py — Un mini-GPT (Transformer décodeur) écrit FROM SCRATCH en PyTorch.

L'architecture suit GPT (inspirée de nanoGPT, version pédagogique) :

    tokens ──► Embedding (mots) + Embedding (positions)
            ──► [ Bloc Transformer ] x N
                   │  LayerNorm ──► Self-Attention causale ──► (+résiduel)
                   │  LayerNorm ──► MLP (feed-forward)      ──► (+résiduel)
            ──► LayerNorm final
            ──► Linear ──► logits (probas du prochain token)

Le modèle apprend UNE seule chose : prédire le prochain token.
À partir de cette tâche simple, répétée des milliers de fois, émergent
les régularités du langage SaaS présent dans les données.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    Self-attention multi-têtes, CAUSALE.

    "Causale" = chaque token ne peut regarder que les tokens PRÉCÉDENTS
    (jamais le futur). C'est ce qui permet de générer du texte de gauche
    à droite, un token à la fois.

    Pour chaque token on calcule 3 vecteurs :
      - Query (Q) : "qu'est-ce que je cherche ?"
      - Key   (K) : "qu'est-ce que je contiens ?"
      - Value (V) : "qu'est-ce que je transmets ?"
    L'attention = produit Q·K (qui regarde qui), normalisé, puis appliqué à V.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd doit être divisible par n_head"
        # une seule projection produit Q, K, V d'un coup (3 * n_embd).
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # projection de sortie.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_flash = config.use_flash_attention and hasattr(
            F, "scaled_dot_product_attention"
        )

        if not self.use_flash:
            # masque triangulaire : interdit de regarder le futur.
            # buffer = pas un paramètre entraînable, mais déplacé avec le modèle.
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch, longueur de séquence, dimension embedding

        # calcule Q, K, V puis les sépare.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # réorganise en (B, n_head, T, head_dim) pour traiter les têtes en parallèle.
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if self.use_flash:
            # version rapide et fusionnée (CUDA). Mêmes maths que ci-dessous.
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # --- version manuelle, LISIBLE (idéale pour comprendre) ---
            # 1) scores d'attention : Q·Kᵀ / sqrt(head_dim)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            # 2) masque causal : -inf sur le futur => proba ~0 après softmax
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            # 3) softmax => poids d'attention (somme à 1 sur chaque ligne)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # 4) moyenne pondérée des Values
            y = att @ v

        # recolle les têtes : (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # projection de sortie + dropout résiduel
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Le réseau feed-forward de chaque bloc. Deux couches linéaires avec une
    non-linéarité GELU au milieu. C'est ici que le modèle "réfléchit" sur
    l'information rassemblée par l'attention. Expansion 4x (convention GPT).
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Un bloc Transformer = Attention + MLP, chacun précédé d'un LayerNorm
    et suivi d'une connexion résiduelle (x + sous-couche). Les résidus
    permettent au gradient de circuler à travers un réseau profond.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias) if config.bias else nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias) if config.bias else nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # pré-norm + résiduel
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),                     # norm finale
        ))
        # tête de sortie : projette vers le vocabulaire (logits).
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying : on partage les poids entre embedding d'entrée et sortie.
        # Économise des paramètres et améliore souvent les résultats.
        self.transformer.wte.weight = self.lm_head.weight

        # initialisation des poids.
        self.apply(self._init_weights)
        # init spéciale (mise à l'échelle) pour les projections résiduelles.
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx     : (B, T) tokens d'entrée.
        targets : (B, T) tokens cibles (= idx décalé d'un cran). Si fourni,
                  on calcule la loss (cross-entropy). Sinon, juste les logits.
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Séquence de longueur {T} > block_size {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab_size)
            # cross-entropy : compare la prédiction au vrai token suivant.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # en inférence, seul le dernier token nous intéresse pour générer.
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def tie_weights(self):
        """
        (Re)lie l'embedding d'entrée (wte) et la tête de sortie (lm_head) pour
        qu'ils partagent le MÊME tenseur (weight tying).

        À rappeler APRÈS un `model.to(device)` : certains backends, notamment
        DirectML, cassent ce partage lors du transfert vers le device (les deux
        deviennent des copies indépendantes). Sans cet appel, le modèle gonfle
        de la taille d'un embedding et entraîne deux matrices au lieu d'une.
        """
        self.lm_head.weight = self.transformer.wte.weight

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        Crée l'optimiseur AdamW avec deux groupes de paramètres :
          - matrices (poids 2D) : avec weight decay (régularisation).
          - vecteurs (biais, LayerNorm, embeddings 1D) : sans weight decay.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in param_dict.values() if p.dim() >= 2]
        no_decay = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, repetition_penalty=1.0, eot_token=None):
        """
        Génère du texte token par token (autoregression).

        temperature        : <1 = prudent/répétitif, >1 = créatif/aléatoire.
        top_k              : ne tire que parmi les k tokens les plus probables.
        top_p              : nucleus sampling — garde les tokens dont la proba
                             cumulée atteint p (ex. 0.9). Complémentaire de top_k.
        repetition_penalty : >1 pénalise les tokens déjà générés (ex. 1.3) pour
                             éviter les boucles ("SaaS SaaS SaaS...").
        eot_token          : si fourni, on arrête dès que ce token est généré.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # tronque le contexte à block_size si nécessaire.
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # 1) pénalité de répétition : sur les tokens déjà présents.
            if repetition_penalty and repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = torch.unique(idx[b])
                    vals = logits[b, seen]
                    logits[b, seen] = torch.where(
                        vals > 0, vals / repetition_penalty, vals * repetition_penalty
                    )

            # 2) température.
            logits = logits / max(temperature, 1e-8)

            # 3) top-k : ne garde que les k meilleurs.
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = float("-inf")

            # 4) top-p (nucleus) : ne garde que la masse de proba cumulée <= p.
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()  # garde toujours le 1er
                remove[..., 0] = False
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(
                    -1, sorted_idx, sorted_logits
                )

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # arrêt propre sur fin de document (batch de taille 1).
            if eot_token is not None and idx.size(0) == 1 and idx_next.item() == eot_token:
                break
        return idx
