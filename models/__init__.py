from typing import Any

import lightning.pytorch as pl
from torch.nn import Module


class Model(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class Encoder(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class Memory(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class Fusion(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
