"""
utils.py — Petites fonctions utilitaires partagées.

La plus importante : get_device(), qui choisit automatiquement le meilleur
backend de calcul disponible, en tenant compte du cas AMD/Windows.
"""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """
    Choisit le device de calcul.

    Ordre de priorité quand preference == "auto" :
      1. CUDA            (GPU NVIDIA — n'existe pas ici, mais on teste)
      2. DirectML        (GPU AMD/Intel sous Windows, via torch-directml)
      3. CPU             (toujours disponible, le défaut sûr)

    Sur une RX 6600 sous Windows, CUDA est absent et ROCm n'est pas supporté.
    Le seul accès GPU passe par DirectML : installe-le avec
        python -m pip install torch-directml
    Sinon tout tourne sur CPU, ce qui reste parfaitement OK pour apprendre.
    """
    pref = (preference or "auto").lower()

    if pref == "cpu":
        return torch.device("cpu")

    if pref == "cuda":
        return torch.device("cuda")

    if pref == "dml":
        return _try_directml(force=True)

    # --- mode auto ---
    if torch.cuda.is_available():
        return torch.device("cuda")

    dml = _try_directml(force=False)
    if dml is not None:
        return dml

    return torch.device("cpu")


def _try_directml(force: bool):
    """Tente de récupérer un device DirectML. Renvoie None si indisponible."""
    try:
        import torch_directml  # type: ignore

        return torch_directml.device()
    except Exception as e:
        if force:
            raise RuntimeError(
                "DirectML demandé mais torch-directml n'est pas installé.\n"
                "Installe-le avec : python -m pip install torch-directml"
            ) from e
        return None


def device_name(device: torch.device) -> str:
    """Nom lisible du device, pour les logs."""
    s = str(device)
    if s.startswith("cuda"):
        try:
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "CUDA"
    if "privateuseone" in s or "dml" in s:
        return "DirectML (GPU AMD/Intel)"
    return "CPU"


def count_parameters(model) -> int:
    """Nombre de paramètres entraînables, pour info."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_count(n: int) -> str:
    """Formate un grand nombre : 15000000 -> '15.0M'."""
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n}"
        n /= 1000.0
    return f"{n:.1f}T"
