from src.utils import *
from src.train import training, evaluation
from src.model import PBT

import numpy as np
import random
import torch


def fine_tune(config):

    if config["data_set"] == "BNCI2014001":
        data, labels, meta, channels = get_BNCI2014001(
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

    elif config["data_set"] == "BNCI2014004":
        data, labels, meta, channels = get_BNCI2014004(
            subject=config["subject"],
            freq_min=config["freq"][0],
            freq_max=config["freq"][1],
        )

        train_data = data[
            np.where(
                (meta["session"] == "session_0")
                | (meta["session"] == "session_1")
                | (meta["session"] == "session_2")
            )
        ]
        train_labels = labels[
            np.where(
                (meta["session"] == "session_0")
                | (meta["session"] == "session_1")
                | (meta["session"] == "session_2")
            )
        ]
        train_meta = meta.iloc[
            np.where(
                (meta["session"] == "session_0")
                | (meta["session"] == "session_1")
                | (meta["session"] == "session_2")
            )
        ]

        test_data = data[
            np.where(
                (meta["session"] == "session_3") | (meta["session"] == "session_4")
            )
        ]
        test_labels = labels[
            np.where(
                (meta["session"] == "session_3") | (meta["session"] == "session_4")
            )
        ]
        test_meta = meta.iloc[
            np.where(
                (meta["session"] == "session_3") | (meta["session"] == "session_4")
            )
        ]

    else:
        raise ValueError("Please choose data_set in {BNCI2014001, BNCI2014004}")

    # ToDo: mean and std should be calculated on train data!
    train_data = zero_mean_unit_var(mne_epochs=train_data, meta_data=train_meta)
    test_data = zero_mean_unit_var(mne_epochs=test_data, meta_data=test_meta)

    """
    train_data = z_score(mne_epochs=train_data, mode="per_channel")
    test_data = z_score(mne_epochs=test_data, mode="per_channel")

    train_epochs, test_epochs = subjectwise_z_score(
        train_epochs=train_data,
        test_epochs=test_data,
        meta_train=train_meta,
        meta_test=test_meta,
        mode="global",
    )
    """

    # train_data, test_data = scale2(train_data, test_data)
    from src.utils import base_dataset

    train_data_set = base_dataset(
        train_data, train_labels, subject_ids=train_meta["subject"], config=config
    )
    test_data_set = base_dataset(
        test_data, test_labels, subject_ids=train_meta["subject"], config=config
    )

    model = PBT(
        config=config,
        n_classes=len(set(test_labels)),
        num_embeddings=512,
        device=device,
    )

    print(model)
    if config["load"]:
        checkpoint = torch.load(config["load"])
        print(checkpoint["model_state_dict"].keys())
        # delete weights that should not be loaded
        # checkpoint['model_state_dict'].pop('pos_embedding.weight')
        # checkpoint["model_state_dict"].pop("cls_head.weight")
        # checkpoint["model_state_dict"].pop("cls_head.bias")
        # checkpoint["model_state_dict"].pop("linear_projection_out.weight")

        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # model parameters
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    if config["eval"]:
        evaluation(
            parameter=config,
            model=model,
            train_data_set=train_data_set,
            test_data_set=test_data_set,
        )
    else:
        training(
            parameter=config,
            model=model,
            train_data_set=train_data_set,
            test_data_set=test_data_set,
            n_classes=len(set(test_labels)),
        )


if __name__ == "__main__":

    config = {
        # Pre - Processing‚
        "freq": [8, 45],
        "normalization": "zscore",
        # Model
        "d_input": 64,
        "d_model": 128,  # Input gets expanded in lin. projection
        "dim_feedforward": 128 * 4,
        "num_tokens_per_channel": 8,
        "num_transformer_blocks": 4,
        "num_heads": 4,  # number attention heads transformer
        "learnable_cls": False,
        "bias_transformer": True,
        # LoRA stuff
        "alpha": 1,
        "rank": 8,
        "num_adapters": 10,
        "model_type": "LoRA",  # "vanilla", "LoRA",
        "model_head_type": "vanilla",  # "vanilla", "LoRA",
        # Train Hyper-Parameters
        "lr": 3e-4,
        "lr_warm_up_iters": 150,
        "batch_size": 128,
        "num_epochs": 1200,
        "betas": (0.9, 0.95),  # betas AdamW
        "clip_gradient": 1.0,
        # Regularization & Augmentation
        "weight_decay": 0.01,  # not applied to LayerNorm, self_att and biases
        "weight_decay_pos_embedding": 0.0,  # weight decay applied to learnable pos. embedding
        "weight_decay_cls_head": 0.0,  # cls_head = classification head (linear layer)
        # higher for pre-train may improve few-shot adaptation
        "dropout": 0.1,
        "label_smoothing": 0,
        "augmentation": ["time_shifts"],  # [] for no aug, else list:
        # ['time_shifts', 'DC_shifts', 'amplitude_scaling','noise']
        # WandB
        "wandb_log": True,
        "wandb_name": False,
        "wandb_proj": "NeurIPS2025Workshop",
        "wandb_watch": False,
        "save": False,  # add path to dicatory
        "checkpoints": 250,
        "load": False,
        "seed": 42,  # set random seed
        "compile_model": False,  # compile model with PyTroch to speed up
        "data_set": "BNCI2014001",
        "subject": list(range(1, 10)),
        "eval": False,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    i = 1
    config["seed"] = i
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    fine_tune(config)
    exit(0)
    # sub = list(range(1, 10))
    for sub in range(1, 10):
        config["wandb_name"] = "sub_" + str(sub)
        config["subject"] = [sub]
        fine_tune(config)
