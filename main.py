import torch
from lightning.pytorch.cli import LightningCLI

from datasets import EEGDataset
from models import Model


class BrainCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.init_args.segment_samples",
            "model.init_args.encoder.init_args.size_input",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "trainer.limit_train_batches",
            "data.init_args.limit_train_batches",
        )

    def instantiate_classes(self) -> None:
        super().instantiate_classes()


def cli_main():
    cli = BrainCLI(
        Model,
        EEGDataset,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=False,
    )

    return cli


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cli = cli_main()
