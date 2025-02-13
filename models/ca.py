from typing import Tuple

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryStatScores

from datasets import EEGBatch
from models import Encoder, Fusion, Memory, Model


class CAModel(Model):
    def __init__(
        self,
        encoder: Encoder,
        fusion: Fusion,
        memory: Memory,
        lr: float = 0.1,
        load_model: str = "",
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.memory = memory
        self.hrr = fusion
        self.seizure_classifier = nn.Sequential(
            nn.ELU(), nn.Linear(2 * memory.size_output, 1)
        )
        self.loss_seizure = BCEWithLogitsLoss(reduction="none")
        self.lr = lr
        self.train_f1_score = BinaryF1Score()
        self.train_accuracy_seizure = BinaryAccuracy()
        self.val_f1_score = BinaryF1Score()
        self.val_accuracy_seizure = BinaryAccuracy()
        self.test_f1_score = BinaryF1Score()
        self.test_accuracy_seizure = BinaryAccuracy()
        self.test_bss = BinaryStatScores()
        hyp = torch.empty((0))
        self.register_buffer("hyp", hyp, persistent=False)

        self.save_hyperparameters(ignore=["encoder", "memory", "fusion"])

        self.load_model = load_model

        base_weights = None

        if load_model != "":
            try:
                base_weights = torch.load(load_model, weights_only=True)
            except FileNotFoundError:
                pass

        if base_weights is not None:
            if base_weights.get("state_dict") is not None:
                base_weights = base_weights["state_dict"]
            self.load_state_dict(base_weights, strict=True)

    def forward(self, x: Tensor, patient: int, dataset: int) -> Tensor:
        out_enc = self.encoder(x)  # [bsz, mem, ch, len]
        out_hrr = self.hrr(out_enc, patient, dataset)  # [bsz, mem, len]
        out_dec = self.memory(out_hrr.transpose(-1, -2))  # [bsz, len]
        out_skip = torch.cat((out_dec, out_hrr[:, -1]), -1)  # [bsz, 2*len]
        return self.seizure_classifier(out_skip)

    @staticmethod
    @torch.no_grad()
    def _merge_patients(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        y = torch.cat(y)
        max_channels = min([b.shape[2] for b in x])
        start_channel = [
            torch.randint(-1, b.shape[2] - max_channels, (1,), device=x[0].device)
            for b in x
        ]
        x = torch.cat(
            [
                b[
                    torch.arange(b.shape[0]),
                    :,
                    start_channel[i].clip(0) : start_channel[i].clip(0) + max_channels,
                ]
                for i, b in enumerate(x)
            ]
        )

        return x, y

    def training_step(
        self,
        batch: EEGBatch,
        batch_idx: int,
    ) -> STEP_OUTPUT:
        x, y = batch.data, batch.label

        if isinstance(x, list):
            x, y = self._merge_patients(x, y)

        patient = int(batch.patient[0])
        dataset = batch.dataset[0]

        out = self.forward(x, patient, dataset)

        loss = self.loss_seizure(out, y)
        loss *= y * 2 + 1
        loss = loss.mean()
        self.train_accuracy_seizure(out.view(-1), y.view(-1).int())
        self.train_f1_score(out.view(-1), y.view(-1).int())

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc_seizure", self.train_accuracy_seizure)
        self.log(
            "train/f1_score",
            self.train_f1_score,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def validation_step(
        self,
        batch: EEGBatch,
        batch_idx: int,
    ):
        x, y = batch.data, batch.label

        if isinstance(x, list):
            x, y = self._merge_patients(x, y)

        patient = int(batch.patient[0])
        dataset = batch.dataset[0]

        out = self.forward(x, patient, dataset)

        self.val_accuracy_seizure(out.view(-1), y.view(-1).int())
        self.val_f1_score(out.view(-1), y.view(-1).int())

        self.hyp = torch.cat((self.hyp, out.view(-1)))

        self.log("val/acc_seizure", self.val_accuracy_seizure)
        self.log(
            "val/f1_score",
            self.val_f1_score,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

    def test_step(
        self,
        batch: EEGBatch,
        batch_idx: int,
    ):
        x, y = batch.data, batch.label

        if isinstance(x, list):
            x, y = self._merge_patients(x, y)

        patient = int(batch.patient[0])
        dataset = batch.dataset[0]

        out = self.forward(x, patient, dataset)

        self.test_accuracy_seizure(out.view(-1), y.view(-1).int())
        self.test_f1_score(out.view(-1), y.view(-1).int())
        self.test_bss(out.view(-1), y.view(-1).int())

        self.hyp = torch.cat((self.hyp, out.view(-1)))

        self.log("test/acc_seizure", self.test_accuracy_seizure)
        self.log(
            "test/f1_score",
            self.test_f1_score,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

    def on_test_epoch_end(self) -> None:
        bss = self.test_bss.compute()
        self.log_dict(
            {
                "test/tp_epoch": bss[0],
                "test/fp_epoch": bss[1],
                "test/tn_epoch": bss[2],
                "test/fn_epoch": bss[3],
            }
        )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer=optimizer, lr_lambda=lambda _: 1.0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
