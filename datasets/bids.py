import os
from math import ceil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from bids import BIDSLayout, BIDSLayoutIndexer
from edfio import read_edf as edfio_read_edf
from torch import Tensor
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset, RandomSampler

from datasets import EEGBatch, EEGDataset
from datasets.patientsampler import (
    ChunkDistributedSamplerWrapper,
    EEGPatientSampler,
    multipatient_collate,
)

_SAMPLING_RATE = 512


def read_edf(
    edf_file: str,
) -> np.ndarray:
    edf_dataset = edfio_read_edf(edf_file=edf_file, lazy_load_data=False)

    signals = np.empty(
        (len(edf_dataset.signals), edf_dataset.signals[0].data.shape[0]),
        dtype=np.float32,
    )

    for ch, signal in enumerate(edf_dataset.signals):
        signals[ch] = signal.data

    return signals


class BIDSEEGData(EEGDataset):
    def __init__(
        self,
        folders: List[Union[str, Path]],
        batch_size: Optional[int] = 32,
        train_patients: List[Optional[List[Optional[int | str]]]] = [None],
        val_patients: List[Optional[List[Optional[int | str]]]] = [[""]],
        test_patients: List[Optional[List[Optional[int | str]]]] = [[""]],
        segment_n: Optional[int] = 10,
        segment_size: Optional[int] = 10000,
        stride: Optional[int] = None,
        patients_per_batch: Optional[int] = None,
        num_workers: Optional[int] = 0,
        limit_train_batches: Optional[int | float] = None,
    ) -> None:
        super().__init__()
        self.folders = folders
        self.batch_size = batch_size
        self.train_patients = train_patients
        self.val_patients = val_patients
        self.test_patients = test_patients
        self._get_dataset_info()
        self.num_workers = num_workers
        self.limit_train_batches = limit_train_batches
        self.train_patients = self._sanitize_patients(train_patients)
        self.val_patients = self._sanitize_patients(val_patients)
        self.test_patients = self._sanitize_patients(test_patients)
        self.patients_per_batch = patients_per_batch
        self.segment_n = segment_n
        self.segment_size = segment_size
        self.segment_samples = int(self.segment_size / 1000.0 * _SAMPLING_RATE)
        self.stride = stride if stride is not None else segment_size
        self.window_size = segment_size * segment_n

    def _get_dataset_info(self) -> None:
        layouts = []
        for idx, folder in enumerate(self.folders):
            train_patients = (
                self.train_patients[idx]
                if isinstance(self.train_patients, list)
                else self.train_patients
            )
            val_patients = (
                self.val_patients[idx]
                if isinstance(self.val_patients, list)
                else self.val_patients
            )
            test_patients = (
                self.test_patients[idx]
                if isinstance(self.test_patients, list)
                else self.test_patients
            )

            if train_patients is None or val_patients is None or test_patients is None:
                lay = BIDSLayout(folder)
                layouts.append(lay)
                continue

            train_patients = (
                [str(p).zfill(1) for p in train_patients if p != ""]
                + [str(p).zfill(2) for p in train_patients if p != ""]
                + [str(p).zfill(3) for p in train_patients if p != ""]
            )
            val_patients = (
                [str(p).zfill(1) for p in val_patients if p != ""]
                + [str(p).zfill(2) for p in val_patients if p != ""]
                + [str(p).zfill(3) for p in val_patients if p != ""]
            )
            test_patients = (
                [str(p).zfill(1) for p in test_patients if p != ""]
                + [str(p).zfill(2) for p in test_patients if p != ""]
                + [str(p).zfill(3) for p in test_patients if p != ""]
            )

            total_patients = train_patients + val_patients + test_patients

            indexer = BIDSLayoutIndexer(
                ignore="sub-", force_index=[f"sub-{p}($|/)" for p in total_patients]
            )
            lay = BIDSLayout(folder, indexer=indexer)
            layouts.append(lay)

        self.layouts = layouts

    def _sanitize_patients(self, subset):
        patients_sane = []
        if len(subset) != len(self.layouts):
            return [lay.get_subjects() for lay in self.layouts]
        for idx, lay in enumerate(self.layouts):
            dataset_subset = subset[idx]
            if dataset_subset == None:
                patients_sane.append(lay.get_subjects())
            elif len(dataset_subset) == 1 and dataset_subset[0] == "":
                patients_sane.append([""])
            else:
                sub_pat = (
                    [str(p).zfill(1) for p in dataset_subset]
                    + [str(p).zfill(2) for p in dataset_subset]
                    + [str(p).zfill(3) for p in dataset_subset]
                )
                patients_sane.append(list(set(lay.get_subjects()) & set(sub_pat)))
        return patients_sane

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.dataset_train = BIDSEEGDataset(
                layouts=self.layouts,
                n_patient=self.train_patients,
                window_n=self.segment_n,
                window=self.window_size,
                stride=self.stride,
            )
            if self.val_patients:
                self.dataset_val = BIDSEEGDataset(
                    layouts=self.layouts,
                    n_patient=self.val_patients,
                    window_n=self.segment_n,
                    window=self.window_size,
                    stride=self.stride,
                )
        if stage == "validate":
            self.dataset_val = BIDSEEGDataset(
                layouts=self.layouts,
                n_patient=self.val_patients,
                window_n=self.segment_n,
                window=self.window_size,
                stride=1000,
            )
        if stage == "test" or stage == "predict":
            self.dataset_test = []
            dataset_test = BIDSEEGDataset(
                layouts=self.layouts,
                n_patient=self.test_patients,
                window_n=self.segment_n,
                window=self.window_size,
                stride=1000,
            )
            self.dataset_test.append(dataset_test)

    def train_dataloader(self) -> DataLoader:
        num_samples = None
        if self.limit_train_batches is not None:
            if isinstance(self.limit_train_batches, int):
                num_samples = self.batch_size * self.limit_train_batches
            elif isinstance(self.limit_train_batches, float):
                num_samples = int(len(self.dataset_train) * self.limit_train_batches)

        if self.patients_per_batch is not None:
            shuffle = False
            sampler = EEGPatientSampler(
                self.dataset_train,
                batch_size=self.batch_size,
                patients_per_batch=self.patients_per_batch,
                shuffle=True,
                num_samples=num_samples,
                drop_last=True,
            )
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            if distributed_sampler_kwargs is not None:
                distributed_sampler_kwargs.setdefault(
                    "seed", int(os.getenv("PL_GLOBAL_SEED", 0))
                )
                distributed_sampler_kwargs.setdefault("shuffle", False)
                sampler = ChunkDistributedSamplerWrapper(
                    sampler=sampler, **distributed_sampler_kwargs
                )
            collate_fn = multipatient_collate(self.patients_per_batch)
        else:
            shuffle = False
            sampler = RandomSampler(
                self.dataset_train, generator=None, num_samples=num_samples
            )
            collate_fn = None
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_patients:
            sampler = EEGPatientSampler(
                self.dataset_val,
                batch_size=self.batch_size,
                patients_per_batch=self.patients_per_batch,
                shuffle=False,
                num_samples=None,
                drop_last=True,
            )
            return DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                sampler=sampler,
                shuffle=False,
                collate_fn=multipatient_collate(self.patients_per_batch),
            )
        else:
            return None

    def test_dataloader(self) -> DataLoader:
        test_dataloader = []
        for dataset in self.dataset_test:
            sampler = EEGPatientSampler(
                dataset,
                batch_size=self.batch_size,
                patients_per_batch=self.patients_per_batch,
                shuffle=False,
                num_samples=None,
            )
            test_dataloader.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=False,
                    collate_fn=multipatient_collate(self.patients_per_batch),
                )
            )
        return test_dataloader

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


class BIDSEEGDataset(Dataset[EEGBatch]):
    def __init__(
        self,
        layouts: List[Union[str, Path]],
        n_patient: List[List[str]],
        sampling_rate: int = _SAMPLING_RATE,
        window_n: int = 1,
        window: int = 80000,
        stride: Optional[int] = None,
    ) -> None:
        self.window_n = window_n
        self.window = window / 1000.0
        self.srate = sampling_rate
        self.stride = stride / 1000.0
        self.n_patient = n_patient
        self.window_samples = int(self.window * self.srate)
        self.stride_samples = int(self.stride * self.srate)

        self.layouts = layouts

        self.map_items()

        self._patient_files = []

        self._datasets = [{} for _ in range(len(self.layouts))]
        self._events = [{} for _ in range(len(self.layouts))]
        self._sampling_rates = [{} for _ in range(len(self.layouts))]

    @property
    def datasets(self):
        for idx, dataset in enumerate(self._datasets):
            if len(dataset) == 0:
                patient_files = self.layouts[idx].get(
                    subject=self.n_patient[idx], extension="edf", return_type="files"
                )
                for pf in patient_files:
                    dataset[Path(pf).name] = read_edf(pf)

        return self._datasets

    @property
    def events(self):
        for idx, events in enumerate(self._events):
            if len(events) == 0:
                patient_files = self.layouts[idx].get(
                    subject=self.n_patient[idx], extension="tsv", return_type="files"
                )
                for pf in patient_files:
                    events[Path(pf).name] = pd.read_csv(pf, sep="\t")

        return self._events

    @property
    def sampling_rates(self):
        for idx, sampling_rate in enumerate(self._sampling_rates):
            if len(sampling_rate) == 0:
                patient_files = self.layouts[idx].get(
                    subject=self.n_patient[idx], extension="edf"
                )
                for pf in patient_files:
                    sampling_rate[pf.filename] = pf.get_metadata()["SamplingFrequency"]

        return self._sampling_rates

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor, int, int, float, float, int]:
        dataset = self.dataset_id[n]
        patient = self.patient_id[n]
        session = self.session_id[n]
        run = self.run_id[n]
        mapped = self.map_id[n]

        edf_filename = (
            f"sub-{patient}_ses-{session}_task-szMonitoring_run-{run}_eeg.edf"
        )
        tsv_filename = (
            f"sub-{patient}_ses-{session}_task-szMonitoring_run-{run}_events.tsv"
        )

        srate = self.sampling_rates[dataset][edf_filename]

        ann = self.events[dataset][tsv_filename]
        ann = ann.loc[ann["eventType"] != "bckg"]
        onset = ann["onset"]
        offset = onset + ann["duration"]

        patient_dataset = self.datasets[dataset][edf_filename]

        sample = patient_dataset[
            :,
            int(mapped * self.stride * srate) : int(
                mapped * self.stride * srate + self.window * srate
            ),
        ]
        sample_torch = torch.from_numpy(sample)
        downsampled_seizure = interpolate(
            sample_torch.unsqueeze(0), size=self.window_samples, mode="linear"
        )
        sample = downsampled_seizure.squeeze(0)

        sample = sample.unfold(
            1,
            self.window_samples // self.window_n,
            self.window_samples // self.window_n,
        )

        window_pos_begin = (
            mapped * self.stride + self.window - self.window // self.window_n
        )
        window_pos_end = mapped * self.stride + self.window
        before_offsets = window_pos_begin <= offset
        after_onsets = window_pos_end >= onset
        seiz_selected = np.logical_and(
            np.asarray(before_offsets), np.asarray(after_onsets)
        ).flatten()
        label = np.any(seiz_selected)
        label_t = Tensor([label])

        return EEGBatch(
            sample.transpose(0, 1).contiguous(),
            label_t,
            n,
            mapped,
            patient,
            dataset,
        )

    def map_items(self) -> None:
        dataset_id = [np.empty((0,), dtype=int)]
        patient_id = [np.empty((0,), dtype=int)]
        session_id = [np.empty((0,), dtype=int)]
        run_id = [np.empty((0,), dtype=int)]
        map_id = [np.empty((0,), dtype=int)]
        for idx, lay in enumerate(self.layouts):
            patient_files = lay.get(subject=self.n_patient[idx], extension="edf")
            for pf in patient_files:
                dur = pf.get_metadata()["RecordingDuration"]
                pat = pf.get_entities()
                if dur < self.window:
                    continue
                windows_dur = ceil((dur - self.window) / self.stride)
                dataset_id.append(np.zeros((windows_dur), dtype=int) + idx)
                patient_id.append(np.full((windows_dur), fill_value=pat["subject"]))
                session_id.append(np.full((windows_dur), fill_value=pat["session"]))
                run_id.append(np.full((windows_dur), fill_value=str(pat["run"])))
                map_id.append(np.arange(windows_dur, dtype=int))

        self.dataset_id = np.concatenate(dataset_id)
        self.patient_id = np.concatenate(patient_id)
        self.session_id = np.concatenate(session_id)
        self.run_id = np.concatenate(run_id)
        self.map_id = np.concatenate(map_id)

    def __len__(self) -> int:
        total_len = len(self.dataset_id)
        return total_len
