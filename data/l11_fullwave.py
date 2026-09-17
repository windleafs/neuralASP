"""Backend-neutral L11 contract; old k-Wave imports remain compatible."""
from .l11_kwave import L11KWaveDataset, validate_l11_sample

L11FullwaveDataset = L11KWaveDataset
__all__ = ["L11FullwaveDataset", "validate_l11_sample"]
