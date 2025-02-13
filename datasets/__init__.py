from collections import namedtuple

import lightning.pytorch as pl


class EEGDataset(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()


EEGBatch = namedtuple(
    "EEGBatch",
    ("data", "label", "id", "sample_id", "patient", "dataset"),
)
