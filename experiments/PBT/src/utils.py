import numpy as np
import random

import torch
from torch.utils.data import Dataset


from moabb.datasets import (
    AlexMI,
    BNCI2014001,
    BNCI2014004,
    BNCI2015001,
    BNCI2015004,
    Cho2017,
    Lee2019_MI,
    PhysionetMI,
)
from moabb.paradigms import MotorImagery


# ----------------------------------------------------------------------------------------------------------------------
# Load data from MOABB
def get_AlexMI(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    PhD-Theses (french): https://theses.hal.science/tel-01196752
    data: https://zenodo.org/records/806023

    Electrode montage: corresponding to the international 10-20 system
    """

    if channels is None:
        channels = [
            "Fpz",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "T7",
            "C3",
            "Cz",
            "C4",
            "T8",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
        ]

    # Labels: right_hand, feet, rest
    if n_classes is None:
        n_classes = 3

    if subject is None:
        subject = list(range(1, 9))

    dataset = AlexMI()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "right_hand")] = 1
    labels[np.where(labels == "feet")] = 2
    labels[np.where(labels == "rest")] = 4
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_BNCI2014001(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://lampx.tugraz.at/~bci/database/001-2014/description.pdf
    This four class motor imagery data set was originally released as data set 2a of the BCI Competition IV


    Electrode montage: corresponding to the international 10-20 system
    """

    if channels is None:
        channels = [
            "Fz",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "P1",
            "Pz",
            "P2",
            "POz",
        ]

    # Labels: left_hand, right_hand, feet, tongue
    if n_classes is None:
        n_classes = 4

    if subject is None:
        subject = list(range(1, 10))

    dataset = BNCI2014001()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "left_hand")] = 0
    labels[np.where(labels == "right_hand")] = 1
    labels[np.where(labels == "feet")] = 2
    labels[np.where(labels == "tongue")] = 3
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_BNCI2014004(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://lampx.tugraz.at/~bci/database/004-2014/description.pdf
    https://ieeexplore.ieee.org/document/4359220

    Electrode montage: 3 bipolar channels (C3, Cz, C4) placed according to the extended 10-20 system
    """

    if channels is None:
        channels = ["C3", "Cz", "C4"]

    # Labels: left_hand, right_hand
    if n_classes is None:
        n_classes = 2

    if subject is None:
        subject = list(range(1, 10))

    dataset = BNCI2014004()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "left_hand")] = 0
    labels[np.where(labels == "right_hand")] = 1
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_BNCI2015001(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://lampx.tugraz.at/~bci/database/001-2015/description.pdf

    Electrode montage: 13 channels placed according to the 10-20 system
    """

    if channels is None:
        channels = [
            "FC3",
            "FCz",
            "FC4",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP3",
            "CPz",
            "CP4",
        ]

    # Labels: right_hand, feet
    if n_classes is None:
        n_classes = 2

    if subject is None:
        subject = list(range(1, 13))

    dataset = BNCI2015001()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "right_hand")] = 1
    labels[np.where(labels == "feet")] = 2
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_BNCI2015004(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://lampx.tugraz.at/~bci/database/004-2015/description.pdf

    Electrode montage: 13 channels placed according to the 10-20 system
    """

    if channels is None:
        channels = [
            "AFz",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "FC3",
            "FCz",
            "FC4",
            "T3",
            "C3",
            "Cz",
            "C4",
            "T4",
            "CP3",
            "CPz",
            "CP4",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO3",
            "PO4",
            "O1",
            "O2",
        ]

    # Labels: right_hand, feet
    if n_classes is None:
        n_classes = 2

    if subject is None:
        subject = list(range(1, 10))

    dataset = BNCI2015004()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    # drop trials with label 'word_ass' 'subtraction', 'navigation'
    idx = np.concatenate(
        (np.where(labels == "feet")[0], np.where(labels == "right_hand")[0])
    )
    data = data[idx]
    labels = labels[idx]
    meta = meta.iloc[idx]

    labels[np.where(labels == "right_hand")] = 1
    labels[np.where(labels == "feet")] = 2
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_Cho2017(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://academic.oup.com/gigascience/article/6/7/gix034/3796323

    Electrode montage: 64 channels placed according to the 10-10 system
    """

    if channels is None:
        channels = [
            "Fp1",
            "AF7",
            "AF3",
            "F1",
            "F3",
            "F5",
            "F7",
            "FT7",
            "FC5",
            "FC3",
            "FC1",
            "C1",
            "C3",
            "C5",
            "T7",
            "TP7",
            "CP5",
            "CP3",
            "CP1",
            "P1",
            "P3",
            "P5",
            "P7",
            "P9",
            "PO7",
            "PO3",
            "O1",
            "Iz",
            "Oz",
            "POz",
            "Pz",
            "CPz",
            "Fpz",
            "Fp2",
            "AF8",
            "AF4",
            "AFz",
            "Fz",
            "F2",
            "F4",
            "F6",
            "F8",
            "FT8",
            "FC6",
            "FC4",
            "FC2",
            "FCz",
            "Cz",
            "C2",
            "C4",
            "C6",
            "T8",
            "TP8",
            "CP6",
            "CP4",
            "CP2",
            "P2",
            "P4",
            "P6",
            "P8",
            "P10",
            "PO8",
            "PO4",
            "O2",
        ]

    # Labels: left_hand, right_hand
    if n_classes is None:
        n_classes = 2

    if subject is None:
        subject = list(range(1, 53))
        # ValueError: Invalid subject 32, 46, 49
        subject.remove(32)
        subject.remove(46)
        subject.remove(49)

    dataset = Cho2017()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "left_hand")] = 0
    labels[np.where(labels == "right_hand")] = 1
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_Lee2019_MI(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://academic.oup.com/gigascience/article/6/7/gix034/3796323

    64 channels placed according to the 10-10 system
    """

    if channels is None:
        channels = [
            "AF3",
            "AF4",
            "AF7",
            "AF8",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "CP1",
            "CP2",
            "CP3",
            "CP4",
            "CP5",
            "CP6",
            "CPz",
            "Cz",
            "F10",
            "F3",
            "F4",
            "F7",
            "F8",
            "F9",
            "FC1",
            "FC2",
            "FC3",
            "FC4",
            "FC5",
            "FC6",
            "FT10",
            "FT9",
            "Fp1",
            "Fp2",
            "Fz",
            "O1",
            "O2",
            "Oz",
            "P1",
            "P2",
            "P3",
            "P4",
            "P7",
            "P8",
            "PO10",
            "PO3",
            "PO4",
            "PO9",
            "POz",
            "Pz",
            "T7",
            "T8",
            "TP10",
            "TP7",
            "TP9",
        ]

    # Labels: left_hand, right_hand
    if n_classes is None:
        n_classes = 2

    if subject is None:
        subject = list(range(1, 55))

    dataset = Lee2019_MI()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "left_hand")] = 0
    labels[np.where(labels == "right_hand")] = 1
    labels = labels.astype(int)

    return data, labels, meta, channels


def get_PhysionetMI(
    subject=None, freq_min=8, freq_max=45, resample=250, channels=None, n_classes=None
):
    """
    https://academic.oup.com/gigascience/article/6/7/gix034/3796323

    Electrode montage: 64 electrodes as per the international 10-10 system
    (excluding electrodes Nz, F9, F10, FT9, FT10, A1, A2, TP9, TP10, P9, and P10)
    """

    if channels is None:
        channels = [
            "Fp1",
            "Fpz",
            "Fp2",
            "AF7",
            "AF3",
            "AFz",
            "AF4",
            "AF8",
            "F7",
            "F5",
            "F3",
            "F1",
            "Fz",
            "F2",
            "F4",
            "F6",
            "F8",
            "FT7",
            "FC5",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "FC6",
            "FT8",
            "T9",
            "T7",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "T8",
            "T10",
            "TP7",
            "CP5",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "CP6",
            "TP8",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO7",
            "PO3",
            "POz",
            "PO4",
            "PO8",
            "O1",
            "Oz",
            "O2",
            "Iz",
        ]

    # Labels: left_hand, right_hand
    if n_classes is None:
        n_classes = 4

    if subject is None:
        subject = list(range(1, 110))

    dataset = PhysionetMI()
    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=freq_min,
        fmax=freq_max,
        channels=channels,
        resample=resample,
    )
    data, labels, meta = paradigm.get_data(dataset=dataset, subjects=subject)

    labels[np.where(labels == "left_hand")] = 0
    labels[np.where(labels == "right_hand")] = 1
    labels[np.where(labels == "feet")] = 2
    labels[np.where(labels == "rest")] = 4
    # booth hands -> soft labels? [.5, .5, 0, 0, 0]
    labels[np.where(labels == "hands")] = 5

    labels = labels.astype(int)
    # data[:, np.array([19, 21, 23, 28, 29, 30, 31, 32, 33, 34, 39, 41, 43])]
    return data, labels, meta, channels


# ----------------------------------------------------------------------------------------------------------------------
# Data loader


class SeqDataset(Dataset):
    def __init__(
        self,
        dim_token,
        num_tokens_per_channel,
        augmentation=[],
    ):

        self.num_tokens_per_channel = num_tokens_per_channel
        self.dim_token = dim_token

        self.list_data_sets = []
        self.list_channel_names = []
        self.list_labels = []

        # list of tuples with (trial_data, trial_label, index_data_set)
        self.list_trials = []

        self.int_pos_channels_per_data_set = []
        self.dict_channels = {}

        # if cls-token should be learnable or not zero it can be overwritten by the model
        self.cls = torch.zeros(1, dim_token)

        if (
            len(
                set(augmentation)
                - {"time_shifts", "DC_shifts", "amplitude_scaling", "noise"}
            )
            != 0
        ):
            no_aug = str(
                set(augmentation)
                - {"time_shifts", "DC_shifts", "amplitude_scaling", "noise"}
            )
            raise ValueError(no_aug + " is not supported as data augmentation")
        self.augmentation = augmentation

    def append_data_set(self, data_set, channel_names, label):
        """
        Note: All data is loaded into RAM, which can be a problem with large amounts of data.
              If it fits, it's faster.

        data_set: np.array of size Trials x Channels x Time
        channel_names: list
        label: np.array of size Trials
        """

        if data_set.shape[0] == label.shape[0] and data_set.shape[1] == len(
            channel_names
        ):
            self.list_data_sets += [data_set]
            self.list_channel_names += [channel_names]
            self.list_labels += [label]
        else:
            raise ValueError("Append data set is not possible, size does not match!")

    def prepare_data_set(self, set_pos_channels=None):
        """
        set_pos_channels (dictionary int_pos_channels_per_data_set): to copy int. channel position from existing
        Dataset (e.g. to ensure train and test datasets return the same position)

        list_trial = list of tuples with (trial_data, trial_label, index_data_set), all as tensors
        """

        self.list_trials = [
            (
                torch.from_numpy(data[idx]).float(),
                torch.LongTensor([label[idx]]),
                torch.LongTensor([idx_ds]),
            )
            for idx_ds, (data, label) in enumerate(
                zip(self.list_data_sets, self.list_labels)
            )
            for idx in range(data.shape[0])
        ]

        unique_channel_names = list(np.unique(sum(self.list_channel_names, [])))

        if set_pos_channels is not None:
            # check if there are new channels:
            new_channels = list(
                set(unique_channel_names) - set(set_pos_channels.keys())
            )
            if len(new_channels) == 0:
                self.dict_channels = set_pos_channels
            else:
                print("Following new channels are added: " + str(new_channels))
                raise ValueError("There are some new Channels")

        else:
            # CLS token has always position 0 -> pos channel start at 1
            self.dict_channels = {
                key: torch.IntTensor(
                    [
                        *range(
                            i * self.num_tokens_per_channel + 1,
                            (i + 1) * self.num_tokens_per_channel + 1,
                        )
                    ]
                )
                for i, key in enumerate(unique_channel_names)
            }

        self.int_pos_channels_per_data_set = [
            torch.cat(
                ([self.dict_channels[key].unsqueeze(dim=0) for key in channel_names]),
                dim=0,
            )
            for channel_names in self.list_channel_names
        ]

        labels = np.array([int(trial[1]) for trial in self.list_trials])
        num_labels = [(i, np.where(labels == i)[0].shape) for i in set(labels)]
        print(num_labels)

        # free some memory
        # self.list_data_sets, self.list_channel_names, self.list_labels = None, None, None

    def __len__(self):
        return len(self.list_trials)

    def __getitem__(self, idx):
        """
        dim_time size: #token x dim batch
        label size: dim batch
        int_pos size: dim batch
        """

        dim_time = self.num_tokens_per_channel * self.dim_token

        if "time_shifts" in self.augmentation:
            data = torch.cat(
                (
                    self.cls,
                    self.list_trials[idx][0][
                        :,
                        (
                            st := random.randint(
                                0, self.list_trials[idx][0].shape[1] - dim_time - 1
                            )
                        ) : st
                        + dim_time,
                    ].reshape(-1, self.dim_token),
                ),
                dim=0,
            )
        else:
            st = (self.list_trials[idx][0].shape[1] - dim_time - 1) // 2
            data = torch.cat(
                (
                    self.cls,
                    self.list_trials[idx][0][:, st : st + dim_time].reshape(
                        -1, self.dim_token
                    ),
                ),
                dim=0,
            )

        if "DC_shifts" in self.augmentation:
            data += torch.rand(1) * 0.2 - 0.1

        if "amplitude_scaling" in self.augmentation:
            data *= torch.rand(1) * 0.2 + 0.9

        if "noise" in self.augmentation:
            data += torch.normal(mean=0, std=0.1, size=data.size())

        label = self.list_trials[idx][1]

        # cls-token has pos. 0
        int_pos = torch.cat(
            (
                torch.IntTensor([0]),
                self.int_pos_channels_per_data_set[self.list_trials[idx][2]][
                    :, : self.num_tokens_per_channel
                ].flatten(),
            ),
            dim=0,
        )

        #  data: tensor, label: tensor, int_pos: tensor
        return data, label, int_pos

    @staticmethod
    def my_collate(batch):
        """
        Converts the output of the generator into the appropriate form
        https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
        """

        num_token_per_trial = [item[0].size(0) for item in batch]
        unique_num_token_within_batch = sorted(set(num_token_per_trial))
        data = [
            torch.empty((0, num_tok, batch[0][0].size(1)))
            for num_tok in unique_num_token_within_batch
        ]
        label = [torch.empty(0) for num_tok in unique_num_token_within_batch]
        int_pos = [
            torch.empty((0, num_tok)) for num_tok in unique_num_token_within_batch
        ]
        unique_num_token_within_batch = np.array(list(unique_num_token_within_batch))
        mini_batch_idx = [
            np.where(unique_num_token_within_batch == num_tok)[0][0]
            for num_tok in num_token_per_trial
        ]

        for i, item in enumerate(batch):
            data[mini_batch_idx[i]] = torch.cat(
                (data[mini_batch_idx[i]], item[0].unsqueeze(0)), dim=0
            )
            label[mini_batch_idx[i]] = torch.cat(
                (label[mini_batch_idx[i]], item[1]), dim=0
            )
            int_pos[mini_batch_idx[i]] = torch.cat(
                (int_pos[mini_batch_idx[i]], item[2].unsqueeze(0)), dim=0
            )

        return {"patched_eeg_token": data, "labels": label, "pos_as_int": int_pos}


class base_dataset(Dataset):
    def __init__(self, data, label, subject_ids, config):

        self.data = torch.from_numpy(data).float()  # Trial x Channel x Time
        self.label = torch.LongTensor(label)  # Trial
        self.subject_ids = torch.LongTensor(subject_ids)

        self.config = config

        self.spatial_pos = torch.repeat_interleave(
            torch.arange(0, data.shape[1], 1),
            repeats=self.config["num_tokens_per_channel"],
        )

        # Todo pos embedding update
        self.temporal_pos = torch.arange(
            0, self.config["num_tokens_per_channel"], 1
        ).repeat(data.shape[1])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        dim_seq = self.config["d_input"] * self.config["num_tokens_per_channel"]

        # Todo data augmentation
        if self.config["augmentation"]:
            st = random.randint(0, self.data[idx].shape[1] - dim_seq - 1)

            data = self.data[
                idx,
                :,
                st : st + dim_seq,
            ].reshape(-1, self.config["d_input"])

        else:

            data = self.data[idx][
                :, (st := (self.data[idx].shape[1] - dim_seq - 1) // 2) : st + dim_seq
            ].reshape(-1, self.config["d_input"])

        return {
            "patched_eeg_token": data,
            "labels": self.label[idx],
            "pos_as_int": torch.arange(
                0, data.shape[0], 1
            ),  # .repeat(data.shape[0], 1),
            "temporal_pos": self.temporal_pos,
            "spatial_pos": self.spatial_pos,
            "subject_id": self.subject_ids[idx],
        }


# ----------------------------------------------------------------------------------------------------------------------
# Normalization methods
def scale(mne_epochs):
    # mne_epochs trials x channels x time
    return (mne_epochs - np.mean(mne_epochs, axis=2, keepdims=True)) / (
        np.max(mne_epochs, axis=2, keepdims=True)
        - np.min(mne_epochs, axis=2, keepdims=True)
    )


def z_score(mne_epochs, mode="global"):
    # mne_epochs trials x channels x time

    if mode == "global":
        # Z-score per trial
        mean = np.mean(mne_epochs, axis=(1, 2), keepdims=True)
        std = np.std(mne_epochs, axis=(1, 2), keepdims=True)
    elif mode == "per_channel":
        # Z-score per channel
        mean = np.mean(mne_epochs, axis=2, keepdims=True)
        std = np.std(mne_epochs, axis=2, keepdims=True)
    else:
        raise ValueError("mode must be 'per_channel' or 'global'")

    std[std == 0] = 1
    return (mne_epochs - mean) / std


def subjectwise_z_score(
    train_epochs, test_epochs, meta_train, meta_test, mode="per_channel"
):
    # train_epochs and test_epochs trials x channels x time

    subjects = sorted(set(meta_train["subject"]))

    for sub in subjects:
        train_mask = meta_train["subject"] == sub
        test_mask = meta_test["subject"] == sub

        train_sub = train_epochs[train_mask]
        test_sub = test_epochs[test_mask]

        if mode == "per_channel":
            mean_sub = np.mean(
                train_sub, axis=(0, 2), keepdims=True
            )  # mean per channel
            std_sub = np.std(train_sub, axis=(0, 2), keepdims=True)
        elif mode == "global":
            mean_sub = np.mean(train_sub)
            std_sub = np.std(train_sub)
            mean_sub = np.full_like(train_sub[0:1], mean_sub)  # match shape
            std_sub = np.full_like(train_sub[0:1], std_sub)
        else:
            raise ValueError("mode must be 'per_channel' or 'global'")

        std_sub[std_sub == 0] = 1

        train_epochs[train_mask] = (train_sub - mean_sub) / std_sub
        test_epochs[test_mask] = (test_sub - mean_sub) / std_sub

    return train_epochs, test_epochs


def zero_mean_unit_var(mne_epochs, meta_data):
    for sub in list(set(meta_data["subject"])):
        for session in list(set(meta_data["session"])):
            data = mne_epochs[
                np.where(
                    (meta_data["subject"] == sub) & (meta_data["session"] == session)
                )
            ]
            mne_std = (
                data.transpose(1, 0, 2)
                .reshape(data.shape[1], data.shape[0] * data.shape[2])
                .std(axis=1)
            )
            mne_mean = (
                data.transpose(1, 0, 2)
                .reshape(data.shape[1], data.shape[0] * data.shape[2])
                .mean(axis=1)
            )
            mne_std = np.expand_dims(mne_std, axis=1)
            mne_mean = np.expand_dims(mne_mean, axis=1)
            data = (data - mne_mean) / mne_std
            mne_epochs[
                np.where(
                    (meta_data["subject"] == sub) & (meta_data["session"] == session)
                )
            ] = data

    return mne_epochs


# ----------------------------------------------------------------------------------------------------------------------
# train test split
def train_test_split(data, labels, meta, test_size=0.05):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)

    train_data = data[idx[: int(idx.shape[0] * (1 - test_size))]]
    train_labels = labels[idx[: int(idx.shape[0] * (1 - test_size))]]
    train_meta = meta.iloc[idx[: int(idx.shape[0] * (1 - test_size))]]

    test_data = data[idx[int(idx.shape[0] * (1 - test_size)) :]]
    test_labels = labels[idx[int(idx.shape[0] * (1 - test_size)) :]]
    test_meta = meta.iloc[idx[int(idx.shape[0] * (1 - test_size)) :]]

    return train_data, train_labels, train_meta, test_data, test_labels, test_meta
