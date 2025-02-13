# A Composable Channel-Adaptive Architecture for iEEG Seizure Classification

The Channel-adaptive iEEG classifier is a framework for processing and classification of heterogeneous iEEG from multiple subjects.

We provide the code for the model and a dataloader for BIDS-formatted files, instructions for installation and use are below.

## Requirements

The `requirements.txt` file is provided in the repository. Simply install all requirements with `pip install -r requirements.txt`.

### Use

We provide a sample config file `configs/CA-EEGWaveNet_BIDS.yaml`. It leverages the BIDS dataloader to process any BIDS-compliant dataset (e.g., the BIDS version of the CHB-MIT Dataset can be found [here](https://zenodo.org/records/10259996)).
We use PyTorch Lightning to distribute the configuration files.

To perform classification on the chosen testing dataset and patient run

```
python main.py test --config configs/CA-EEGWaveNet_BIDS.yaml --model.init_args.load_model '<checkpoint_path>' --data.init_args.folders ['<dataset_path>'] --data.init_args.train_patients [['']] --data.init_args.test_patients [['<dataset_subject>']]
```

To train CA-EEGWaveNet run

```
python main.py fit --config configs/CA-EEGWaveNet_BIDS.yaml --data.init_args.folders ['<dataset_path>'] --data.init_args.train_patients [['<dataset_subject>']]
```

## License

If you would like to see the detailed LICENSE click [here](LICENSE).

```text
#
# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: Apache-2.0
#
```