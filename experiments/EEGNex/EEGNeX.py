# Modified from Braindecode (BSD 3-Clause License) https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegnex.py
# Original copyright (c) 2017-currently Braindecode Developers
# See LICENSE_BRAINDECODE for details.
# Authors of the Base Code: Bruno Aristimunha <b.aristimunha@gmail.com>


import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from braindecode.models import EEGModuleMixin
from braindecode.modules import Conv2dWithConstraint, LinearWithConstraint


class LoRAConvPerSubject(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        groups=1,
        rank=4,
        alpha=1.0,
        num_adapters=4,
        stride=1,
        padding="same",
        bias=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self.rank = rank
        self.alpha = alpha
        self.num_adapters = num_adapters

        # LoRA adapters as experts
        self.lora_A = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=rank,
                    kernel_size=kernel_size,
                    dilation=1,
                    groups=1,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
                for _ in range(num_adapters)
            ]
        )
        self.lora_B = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=rank,
                    out_channels=out_channels,
                    kernel_size=1,
                    dilation=1,
                    groups=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
                for _ in range(num_adapters)
            ]
        )

        # Init B with zeros
        for a in self.lora_A:
            torch.nn.init.normal_(a.weight, mean=0.0, std=0.02)
        for b in self.lora_B:
            nn.init.zeros_(b.weight)
            # torch.nn.init.normal_(b.weight, mean=0.0, std=0.02)

    def forward(self, x, subject_id: torch.LongTensor):
        """
        x: batch size, sequence length, token dim
        subject_id: (batch_size,) ints in [0, num_subjects)
        """
        out = self.conv(x)

        # Apply correct adapter to each Subject
        lora_out = torch.zeros_like(out)
        for i in range(self.num_adapters):
            mask = subject_id == i
            if mask.any():
                lora_A_i = self.lora_A[i](x[mask])
                lora_B_i = self.lora_B[i](lora_A_i)
                lora_out[mask] = self.alpha / self.rank * lora_B_i

        return out + lora_out


class EEGNeX(EEGModuleMixin, nn.Module):
    """EEGNeX model from Chen et al. (2024) [eegnex]_.

    .. figure:: https://braindecode.org/dev/_static/model/eegnex.jpg
        :align: center
        :alt: EEGNeX Architecture

    Parameters
    ----------
    activation : nn.Module, optional
        Activation function to use. Default is `nn.ELU`.
    depth_multiplier : int, optional
        Depth multiplier for the depthwise convolution. Default is 2.
    filter_1 : int, optional
        Number of filters in the first convolutional layer. Default is 8.
    filter_2 : int, optional
        Number of filters in the second convolutional layer. Default is 32.
    drop_prob: float, optional
        Dropout rate. Default is 0.5.
    kernel_block_4 : tuple[int, int], optional
        Kernel size for block 4. Default is (1, 16).
    dilation_block_4 : tuple[int, int], optional
        Dilation rate for block 4. Default is (1, 2).
    avg_pool_block4 : tuple[int, int], optional
        Pooling size for block 4. Default is (1, 4).
    kernel_block_5 : tuple[int, int], optional
        Kernel size for block 5. Default is (1, 16).
    dilation_block_5 : tuple[int, int], optional
        Dilation rate for block 5. Default is (1, 4).
    avg_pool_block5 : tuple[int, int], optional
        Pooling size for block 5. Default is (1, 8).

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description and
    source code in tensorflow [EEGNexCode]_.

    References
    ----------
    .. [eegnex] Chen, X., Teng, X., Chen, H., Pan, Y., & Geyer, P. (2024).
       Toward reliable signals decoding for electroencephalogram: A benchmark
       study to EEGNeX. Biomedical Signal Processing and Control, 87, 105475.
    .. [EEGNexCode] Chen, X., Teng, X., Chen, H., Pan, Y., & Geyer, P. (2024).
       Toward reliable signals decoding for electroencephalogram: A benchmark
       study to EEGNeX. https://github.com/chenxiachan/EEGNeX
    """

    def __init__(
        self,
        # Signal related parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Model parameters
        activation: nn.Module = nn.ELU,
        depth_multiplier: int = 2,
        filter_1: int = 8,
        filter_2: int = 32,
        drop_prob: float = 0.5,
        kernel_block_1_2: int = 64,
        kernel_block_4: int = 16,
        dilation_block_4: int = 2,
        avg_pool_block4: int = 4,
        kernel_block_5: int = 16,
        dilation_block_5: int = 4,
        avg_pool_block5: int = 8,
        max_norm_conv: float = 1.0,
        max_norm_linear: float = 0.25,
        #
        mode="LoRA",  # "vanilla" or "LoRA"
        rank=4,
        alpha=1.0,
        num_adapters=10,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        print(mode)
        self.depth_multiplier = depth_multiplier
        self.filter_1 = filter_1
        self.filter_2 = filter_2
        self.filter_3 = self.filter_2 * self.depth_multiplier
        self.drop_prob = drop_prob
        self.activation = activation
        self.kernel_block_1_2 = (1, kernel_block_1_2)
        self.kernel_block_4 = (1, kernel_block_4)
        self.dilation_block_4 = (1, dilation_block_4)
        self.avg_pool_block4 = (1, avg_pool_block4)
        self.kernel_block_5 = (1, kernel_block_5)
        self.dilation_block_5 = (1, dilation_block_5)
        self.avg_pool_block5 = (1, avg_pool_block5)

        self.mode = mode

        # final layers output
        self.in_features = self._calculate_output_length()

        if mode == "vanilla":
            # Following paper nomenclature
            self.block_1 = nn.Sequential(
                Rearrange("batch ch time -> batch 1 ch time"),
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.filter_1,
                    kernel_size=self.kernel_block_1_2,
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=self.filter_1),
            )

            self.block_2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.filter_1,
                    out_channels=self.filter_2,
                    kernel_size=self.kernel_block_1_2,
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=self.filter_2),
            )

            self.block_3 = nn.Sequential(
                Conv2dWithConstraint(
                    in_channels=self.filter_2,
                    out_channels=self.filter_3,
                    max_norm=max_norm_conv,
                    kernel_size=(self.n_chans, 1),
                    groups=self.filter_2,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=self.filter_3),
                self.activation(),
                nn.AvgPool2d(
                    kernel_size=self.avg_pool_block4,
                    padding=(0, 1),
                ),
                nn.Dropout(p=self.drop_prob),
            )

            self.block_4 = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.filter_3,
                    out_channels=self.filter_2,
                    kernel_size=self.kernel_block_4,
                    dilation=self.dilation_block_4,
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=self.filter_2),
            )

            self.block_5 = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.filter_2,
                    out_channels=self.filter_1,
                    kernel_size=self.kernel_block_5,
                    dilation=self.dilation_block_5,
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=self.filter_1),
                self.activation(),
                nn.AvgPool2d(
                    kernel_size=self.avg_pool_block5,
                    padding=(0, 1),
                ),
                nn.Dropout(p=self.drop_prob),
                nn.Flatten(),
            )

        elif mode == "LoRA":
            self.block_1_0 = nn.Sequential(
                Rearrange("batch ch time -> batch 1 ch time")
            )
            self.block_1_1 = LoRAConvPerSubject(
                in_channels=1,
                out_channels=self.filter_1,
                kernel_size=self.kernel_block_1_2,
                rank=rank,
                alpha=alpha,
                num_adapters=num_adapters,
                padding="same",
                bias=False,
            )
            self.block_1_2 = nn.Sequential(nn.BatchNorm2d(num_features=self.filter_1))

            self.block_2_0 = LoRAConvPerSubject(
                in_channels=self.filter_1,
                out_channels=self.filter_2,
                kernel_size=self.kernel_block_1_2,
                rank=rank,
                alpha=alpha,
                num_adapters=num_adapters,
                padding="same",
                bias=False,
            )
            self.block_2_1 = nn.Sequential(
                nn.BatchNorm2d(num_features=self.filter_2),
            )

            self.block_3_0 = Conv2dWithConstraint(
                in_channels=self.filter_2,
                out_channels=self.filter_3,
                max_norm=max_norm_conv,
                kernel_size=(self.n_chans, 1),
                groups=self.filter_2,
                bias=False,
            )

            self.block_3_1 = nn.Sequential(
                nn.BatchNorm2d(num_features=self.filter_3),
                self.activation(),
                nn.AvgPool2d(
                    kernel_size=self.avg_pool_block4,
                    padding=(0, 1),
                ),
                nn.Dropout(p=self.drop_prob),
            )

            self.block_4_1 = LoRAConvPerSubject(
                in_channels=self.filter_3,
                out_channels=self.filter_2,
                kernel_size=self.kernel_block_4,
                dilation=self.dilation_block_4,
                padding="same",
                rank=rank,
                alpha=alpha,
                num_adapters=num_adapters,
                bias=False,
            )
            self.block_4_2 = nn.BatchNorm2d(num_features=self.filter_2)

            self.block_5_0 = LoRAConvPerSubject(
                in_channels=self.filter_2,
                out_channels=self.filter_1,
                kernel_size=self.kernel_block_5,
                dilation=self.dilation_block_5,
                padding="same",
                rank=rank,
                alpha=alpha,
                num_adapters=num_adapters,
                bias=False,
            )

            self.block_5_1 = nn.Sequential(
                nn.BatchNorm2d(num_features=self.filter_1),
                self.activation(),
                nn.AvgPool2d(
                    kernel_size=self.avg_pool_block5,
                    padding=(0, 1),
                ),
                nn.Dropout(p=self.drop_prob),
                nn.Flatten(),
            )

        self.final_layer = LinearWithConstraint(
            in_features=self.in_features,
            out_features=self.n_outputs,
            max_norm=max_norm_linear,
        )

    def forward(self, x: torch.Tensor, subject_id=None) -> torch.Tensor:
        """
        Forward pass of the EEGNeX model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs).
        """

        if self.mode == "vanilla":
            # x shape: (batch_size, n_chans, n_times)
            x = self.block_1(x)
            # (batch_size, n_filter, n_chans, n_times)
            x = self.block_2(x)
            # (batch_size, n_filter*4, n_chans, n_times)
            x = self.block_3(x)
            # (batch_size, 1, n_filter*8, n_times//4)
            x = self.block_4(x)
            # (batch_size, 1, n_filter*8, n_times//4)
            x = self.block_5(x)

        elif self.mode == "LoRA":
            x = self.block_1_0(x)
            x = self.block_1_1(x, subject_id)

            x = self.block_2_0(x, subject_id)
            x = self.block_2_1(x)

            x = self.block_3_0(x)  # , subject_id)
            x = self.block_3_1(x)

            x = self.block_4_1(x, subject_id)
            x = self.block_4_2(x)

            x = self.block_5_0(x, subject_id)
            x = self.block_5_1(x)

        # (batch_size, n_filter*(n_times//32))
        x = self.final_layer(x)

        return x

    def _calculate_output_length(self) -> int:
        # Pooling kernel sizes for the time dimension
        p4 = self.avg_pool_block4[1]
        p5 = self.avg_pool_block5[1]

        # Padding for the time dimension (assumed from padding=(0, 1))
        pad4 = 1
        pad5 = 1

        # Stride is assumed to be equal to kernel size (p4 and p5)

        # Calculate time dimension after block 3 pooling
        # Formula: floor((L_in + 2*padding - kernel_size) / stride) + 1
        T3 = math.floor((self.n_times + 2 * pad4 - p4) / p4) + 1

        # Calculate time dimension after block 5 pooling
        T5 = math.floor((T3 + 2 * pad5 - p5) / p5) + 1

        # Calculate final flattened features (channels * 1 * time_dim)
        # The spatial dimension is reduced to 1 after block 3's depthwise conv.
        final_in_features = (
            self.filter_1 * T5
        )  # filter_1 is the number of channels before flatten
        return final_in_features


import torch
from torch.utils.data import DataLoader, Dataset

# from braindecode.models import EEGNeX
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)
from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
import numpy as np
from sklearn.model_selection import train_test_split
import mne


def main(config):

    # --- 1. Load and Prepare the Dataset ---
    from utils import get_BNCI2014001

    data, labels, meta, channels = get_BNCI2014001(
        # subject=[*range(1, 10)],
        subject=config["subject"],
        freq_min=config["freq"][0],
        freq_max=config["freq"][1],
    )

    train_data = data[np.where(meta["session"] == "session_T")]
    train_labels = labels[np.where(meta["session"] == "session_T")]
    train_meta = meta.iloc[np.where(meta["session"] == "session_T")]

    test_data = data[np.where(meta["session"] == "session_E")]
    test_labels = labels[np.where(meta["session"] == "session_E")]
    test_meta = meta.iloc[np.where(meta["session"] == "session_E")]

    train_data = train_data[:, :, 244:756]
    test_data = test_data[:, :, 244:756]

    print(train_data.shape)
    print(test_data.shape)

    class TensorsDataset(Dataset):
        def __init__(self, X, y, subject_ids):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).long()
            self.subject_id = torch.LongTensor(subject_ids.to_numpy())

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.subject_id[idx]

    print(train_meta["subject"].values)
    print(test_meta["subject"].values)
    train_dataset = TensorsDataset(train_data, train_labels, train_meta["subject"])
    valid_dataset = TensorsDataset(test_data, test_labels, test_meta["subject"])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # --- 4. Define and Train the Model ---
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    print(f"Using device: {device}")

    n_channels = train_data.shape[1]
    n_classes = len(np.unique(train_labels))
    input_window_samples = train_data.shape[2]

    model = EEGNeX(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=input_window_samples,
        # final_conv_length='auto',
        mode=config["mode"],  # "vanilla" or "LoRA"
        rank=config["rank"],
        alpha=config["alpha"],
        num_adapters=config["num_adapters"],
    )

    # print trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {total_params}")

    if cuda:
        model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # --- Training Loop ---
    for epoch in range(config["epochs"]):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for inputs, labels, subject_id in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            subject_id = subject_id.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, subject_id)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        model.eval()
        valid_loss, valid_acc = 0.0, 0.0
        with torch.no_grad():
            for inputs, labels, subject_id in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                subject_id = subject_id.to(device)
                outputs = model(inputs, subject_id)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                valid_acc += (predicted == labels).sum().item()

        valid_loss /= len(valid_loader)
        valid_acc /= len(valid_loader.dataset)

        print(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}"
        )

        wandb.log(
            {
                "train_loss": train_loss,
                "test_loss": valid_loss,
                "acc_train": train_acc,
                "acc_test": valid_acc,
            }
        )

    data, labels, meta, channels = get_BNCI2014001(
        # subject=[*range(1, 10)],
        subject=[1],
        freq_min=config["freq"][0],
        freq_max=config["freq"][1],
    )

    train_data = data[np.where(meta["session"] == "session_T")]
    train_labels = labels[np.where(meta["session"] == "session_T")]
    train_meta = meta.iloc[np.where(meta["session"] == "session_T")]

    test_data = data[np.where(meta["session"] == "session_E")]
    test_labels = labels[np.where(meta["session"] == "session_E")]
    test_meta = meta.iloc[np.where(meta["session"] == "session_E")]

    train_data = train_data[:, :, 244:756]
    test_data = test_data[:, :, 244:756]

    train_meta["subject"][:] = -10

    train_dataset = TensorsDataset(train_data, train_labels, train_meta["subject"])
    valid_dataset = TensorsDataset(test_data, test_labels, test_meta["subject"])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config["batch_size"], shuffle=False
    )

    model.eval()
    valid_loss, valid_acc = 0.0, 0.0
    with torch.no_grad():
        for inputs, labels, subject_id in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            subject_id = subject_id.to(device)
            outputs = model(inputs, subject_id)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            valid_acc += (predicted == labels).sum().item()

    valid_loss /= len(valid_loader)
    valid_acc /= len(valid_loader.dataset)

    print(
        f"Epoch {epoch+1}/{config['epochs']} | "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}"
    )


if __name__ == "__main__":

    config = {
        "mode": "LoRA",  # "vanilla" or "LoRA"
        "rank": 4,
        "alpha": 1.0,
        "num_adapters": 10,
        "freq": [8, 45],  # [4, 38]
        "subject": list(range(1, 10)),
        "epochs": 2,
        "batch_size": 64,
        "weight_decay": 0.01,
        "lr": 0.001,
        # "wandb_proj": "NeurIPS_Workshop_EEGNeX",
    }

    import random

    i = 3
    config["seed"] = i
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True

    #main(config)

    import wandb

    wandb.init(
        project="NeurIPS_Workshop_EEGNeX",  # parameter["wandb_proj"],
        config=config,
        reinit=False,
    )

    main(config)
