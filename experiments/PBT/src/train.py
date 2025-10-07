import warnings
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model import LearningRateScheduler
import wandb


def init_wandb(parameter, model):

    # get rid of parameter that should not be logged
    wandb_dict = {
        k: parameter[k]
        for k in parameter.keys()
        - ["wandb_log", "compile_model", "wandb_proj", "data_set", "wandb_name"]
    }
    if parameter["wandb_name"]:
        wandb.init(
            project=parameter["wandb_proj"],
            config=wandb_dict,
            reinit=False,
            name=parameter["wandb_name"],
        )
    else:
        wandb.init(project=parameter["wandb_proj"], config=wandb_dict, reinit=False)

    if parameter["wandb_watch"]:
        wandb.watch(models=model, log="all", log_freq=25)


def evaluation(parameter, model, train_data_set, test_data_set, num_workers=3):

    train_generator = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=parameter["batch_size"],
        shuffle=False,
        drop_last=False,
        # collate_fn=test_data_set.my_collate,
        num_workers=num_workers,
    )

    test_generator = torch.utils.data.DataLoader(
        test_data_set,
        batch_size=parameter["batch_size"],
        shuffle=False,
        drop_last=False,
        # collate_fn=test_data_set.my_collate,
        num_workers=num_workers,
    )

    save_data = {
        "train_transformer_out": [],
        "train_logits": [],
        "train_subject_id": [],
        "train_labels": [],
        "test_transformer_out": [],
        "test_logits": [],
        "test_subject_id": [],
        "test_labels": [],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        current_train_loss, current_test_loss = 0, 0
        true_labels_train = torch.empty(0, dtype=torch.float).to(device)
        pred_labels_train = torch.empty(0, dtype=torch.float).to(device)
        true_labels_test = torch.empty(0, dtype=torch.float).to(device)
        pred_labels_test = torch.empty(0, dtype=torch.float).to(device)

        for data in train_generator:
            transformer_out, logits, experts_chosen_list = model.forward(
                x=data["patched_eeg_token"].to(device),
                pos=data["pos_as_int"].to(device),
                spatial_pos=data["temporal_pos"].to(device),
                temporal_pos=data["spatial_pos"].to(device),
                subject_id=data["subject_id"].to(device),
            )

            save_data["train_transformer_out"].append(transformer_out)
            save_data["train_logits"].append(logits)
            save_data["train_subject_id"].append(data["subject_id"])
            save_data["train_labels"].append(data["labels"])

            # current_train_loss += loss
            pred_labels_train = torch.cat((pred_labels_train, logits.argmax(dim=1)), 0)
            true_labels_train = torch.cat(
                (true_labels_train, data["labels"].to(device)), 0
            )

        for data in test_generator:
            transformer_out, logits, experts_chosen_list = model.forward(
                x=data["patched_eeg_token"].to(device),
                pos=data["pos_as_int"].to(device),
                spatial_pos=data["temporal_pos"].to(device),
                temporal_pos=data["spatial_pos"].to(device),
                subject_id=data["subject_id"].to(device),
            )

            save_data["test_transformer_out"].append(transformer_out)
            save_data["test_logits"].append(logits)
            save_data["test_subject_id"].append(data["subject_id"])
            save_data["test_labels"].append(data["labels"])

            # current_train_loss += loss
            pred_labels_test = torch.cat((pred_labels_train, logits.argmax(dim=1)), 0)
            true_labels_test = torch.cat(
                (true_labels_train, data["labels"].to(device)), 0
            )

    current_acc_train = torch.sum(
        true_labels_train == pred_labels_train
    ) / true_labels_train.size(0)
    current_acc_test = torch.sum(
        true_labels_test == pred_labels_test
    ) / true_labels_test.size(0)

    print(f"train acc: {current_acc_train}, test acc: {current_acc_test}")
    path = "/home/klein/Project/PatchedBrainTransformer/saved_models/worldly-wood-82/only_LoRA_embedding.pt"
    torch.save(save_data, path)


def training(
    parameter,
    model,
    train_data_set,
    test_data_set,
    n_classes,
    num_workers=3,
    chose_optimizer="AdamW",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # log with WandB
    if parameter["wandb_log"]:
        init_wandb(parameter=parameter, model=model)

    criterion = nn.CrossEntropyLoss(label_smoothing=parameter["label_smoothing"]).to(
        device
    )

    if chose_optimizer == "customAdamW":
        optimizer = model.configure_optimizers(
            weight_decay=parameter["weight_decay"],
            weight_decay_cls_head=parameter["weight_decay_cls_head"],
            learning_rate=parameter["lr"],
            betas=parameter["betas"],
            device_type=device,
        )

    elif chose_optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=parameter["lr"],
            betas=(0.9, 0.95),
            weight_decay=parameter["weight_decay"],
        )
    else:
        raise ValueError(f"Optimizer {chose_optimizer} is not supported")

    learning_rate_scheduler = LearningRateScheduler(
        warmup_iters=parameter["lr_warm_up_iters"],
        learning_rate=parameter["lr"],
        lr_decay_iters=parameter["num_epochs"],
        min_lr=parameter["lr"] * 1 / 10,
    )

    # compile_model
    if parameter["compile_model"]:
        # torch.compile requires PyTorch >=2.0
        if int(torch.__version__[0]) >= 2:
            model = torch.compile(model)
            print("Model is compiled")
        else:
            warnings.warn(
                "Compile model requires PyTorch >= 2.0, model is NOT compiled!"
            )

    # --------------------------------------------------------------------------------------------------------------
    # Training Loop
    # if not train_data_set.dict_channels == test_data_set.dict_channels:
    #    raise ValueError("Channel index between train and test is not consistence")

    train_generator = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=parameter["batch_size"],
        shuffle=True,
        drop_last=True,
        # collate_fn=train_data_set.my_collate,
        num_workers=num_workers,
    )

    test_generator = torch.utils.data.DataLoader(
        test_data_set,
        batch_size=parameter["batch_size"],
        shuffle=False,
        drop_last=False,
        # collate_fn=test_data_set.my_collate,
        num_workers=num_workers,
    )

    for ml_epochs in range(parameter["num_epochs"]):

        lr = learning_rate_scheduler.get_lr(iteration=ml_epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # save loss and labels within an epoch to log the average across the epoch
        current_train_loss, current_test_loss = 0, 0
        true_labels_train = torch.empty(0, dtype=torch.float).to(device)
        pred_labels_train = torch.empty(0, dtype=torch.float).to(device)
        true_labels_test = torch.empty(0, dtype=torch.float).to(device)
        pred_labels_test = torch.empty(0, dtype=torch.float).to(device)

        # --------------------------------------------------------------------------------------------------------------
        # check the experts choices
        all_expert_choices = [[] for _ in range(parameter["num_transformer_blocks"])]
        subject_ids = []

        for i, data in enumerate(train_generator):

            model.train()
            optimizer.zero_grad()

            logits = torch.empty(0, n_classes).to(device)
            target = torch.empty(0)
            current_help_loss = 0

            transformer_out, logits, experts_chosen_list = model.forward(
                x=data["patched_eeg_token"].to(device),
                pos=data["pos_as_int"].to(device),
                spatial_pos=data["temporal_pos"].to(device),
                temporal_pos=data["spatial_pos"].to(device),
                subject_id=data["subject_id"].to(device),
            )

            loss = criterion(logits, data["labels"].to(device))
            loss.backward()

            if parameter["clip_gradient"]:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=parameter["clip_gradient"]
                )
            optimizer.step()

            current_train_loss += loss
            pred_labels_train = torch.cat((pred_labels_train, logits.argmax(dim=1)), 0)
            true_labels_train = torch.cat(
                (true_labels_train, data["labels"].to(device)), 0
            )

        # --------------------------------------------------------------------------------------------------------------
        # Evaluation step
        for j, data in enumerate(test_generator):
            with torch.no_grad():
                model.eval()

                logits = torch.empty(0, n_classes).to(device)
                target = torch.empty(0)
                current_help_loss = 0

                transformer_out, logits, experts_chosen_list = model.forward(
                    x=data["patched_eeg_token"].to(device),
                    pos=data["pos_as_int"].to(device),
                    spatial_pos=data["temporal_pos"].to(device),
                    temporal_pos=data["spatial_pos"].to(device),
                    subject_id=data["subject_id"].to(device),
                )

                loss = criterion(logits, data["labels"].to(device))

                current_test_loss += loss
                pred_labels_test = torch.cat(
                    (pred_labels_test, logits.argmax(dim=1)), 0
                )
                true_labels_test = torch.cat(
                    (true_labels_test, data["labels"].to(device)), 0
                )

        current_train_loss = current_train_loss.cpu().detach().numpy() / (i + 1)
        current_test_loss = current_test_loss.cpu().detach().numpy() / (j + 1)

        current_acc_train = torch.sum(
            true_labels_train == pred_labels_train
        ) / true_labels_train.size(0)
        current_acc_test = torch.sum(
            true_labels_test == pred_labels_test
        ) / true_labels_test.size(0)

        if parameter["wandb_log"]:
            wandb.log(
                {
                    "train_loss": current_train_loss,
                    "test_loss": current_test_loss,
                    "acc_train": current_acc_train,
                    "acc_test": current_acc_test,
                }
            )

        else:
            print(
                f"\n Epoch: {ml_epochs}, train_loss {current_train_loss}, test_loss {current_test_loss}"
                f"\n acc_train {current_acc_train}, acc_test {current_acc_test}"
            )

        if parameter["checkpoints"]:
            if parameter["save"]:
                if parameter["wandb_log"]:
                    path = os.path.join(parameter["save"], wandb.run.name)
                else:
                    path = parameter["save"]
                if not os.path.isdir(path):
                    os.mkdir(path)

                if ml_epochs % parameter["checkpoints"] == 0 and ml_epochs != 0:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "learning_rate_scheduler": learning_rate_scheduler.state_dict(),
                            "config": parameter,
                            "epoch": ml_epochs,
                            "train_loss": current_train_loss,
                            "test_loss": current_test_loss,
                            "acc_train": current_acc_train,
                            "acc_test": current_acc_test,
                            # "dict_channels": train_data_set.dict_channels,
                        },
                        os.path.join(path, "checkpoint_" + str(ml_epochs) + ".pt"),
                    )

            else:
                warnings.warn("Checkpoints are not saved!")

    if parameter["save"]:
        if parameter["wandb_log"]:
            path = os.path.join(parameter["save"], wandb.run.name)
        else:
            path = parameter["save"]
        if not os.path.isdir(path):
            os.mkdir(path)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "learning_rate_scheduler": learning_rate_scheduler.state_dict(),
                "config": parameter,
                "epoch": ml_epochs,
                "train_loss": current_train_loss,
                "test_loss": current_test_loss,
                "acc_train": current_acc_train,
                "acc_test": current_acc_test,
                # "dict_channels": train_data_set.dict_channels,
            },
            os.path.join(path, "final_" + str(ml_epochs) + ".pt"),
        )

    if parameter["wandb_log"]:
        wandb.finish()
