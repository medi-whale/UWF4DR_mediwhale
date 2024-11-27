from typing import Type

import numpy as np
import pandas as pd
import timm
import torch
import torchmetrics
from lightning import LightningModule

from model import model
from tasks import get_task
from loss import BCELoss
from scheduler import CosineWarmup


class DrNoonML(LightningModule):
    def __init__(
        self,
        arch: str,
        task_names: list[str],
        lr: float = 2e-4,
        warmup_steps: int = 1000,
        end_steps: int = 5000,
        label_smoothing: float = 0.0,
        loss_autoweighting: bool = False,
    ):
        super().__init__()

        self.task_names = task_names
        self.subtask_names = {
            task: [
                subtask_name.replace("_score", "") for subtask_name in get_task(task).subtask_names
            ]
            for task in self.task_names
        }
        self.model = model(
            arch=arch,
            task_names=self.task_names,
        )

        self.auroc = torchmetrics.AUROC(task="binary")

        self.world_size = torch.cuda.device_count()

        self.warmup_steps, self.end_steps = warmup_steps, end_steps
        self.sup_loss = BCELoss(self.task_names, label_smoothing, loss_autoweighting)
        self.label_smoothing = label_smoothing

        self.save_hyperparameters()

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        """Train model with batch and return loss."""
        img = batch["img"]
        outputs = self(img)

        losses = self.sup_loss(
            outputs=outputs, targets=batch
        )

        total_loss = sum(losses.values())

        log_dict = {f"train_loss_{k}": float(v) for k, v in losses.items()}

        log_dict["train_total_loss"] = total_loss
        self.log_dict(
            log_dict,
            sync_dist=True,
        )
        return total_loss

    def on_validation_epoch_start(self):
        """Initialize the all_results lists for validation."""
        self.all_results = []

    def on_validation_epoch_end(self):
        """Loop through all labels, compute and log their respective AUCs."""
        total_loss = 0
        all_results = {
            k: np.concatenate([result[k] for result in self.all_results])
            for k in self.all_results[0]
            if "loss" not in k
        }
        df = pd.DataFrame.from_dict(all_results)

        log_dict = {}
        for task in self.task_names:
            for subtask_name in self.subtask_names[task]:
                preds = torch.from_numpy(df[f"{subtask_name}_score"].values)
                labels = torch.from_numpy(df[f"{subtask_name}_label"].values)
                mask = torch.from_numpy(df[f"{subtask_name}_mask"].values)

                preds = preds[mask == 1]
                labels = labels[mask == 1]

                if preds.shape[0] == 0:
                    continue

                auc = self.auroc(preds, labels)  # Assuming self.auroc is initialized in __init__
                log_dict[f"tune_{subtask_name}_auc"] = float(auc)

            loss = np.array([d[f"{task}_loss"] for d in self.all_results]).mean()
            total_loss += loss
            log_dict[f"tune_{task}_loss"] = float(loss)

        log_dict["tune_total_loss"] = float(total_loss)
        self.log_dict(log_dict, sync_dist=True)

        self.all_results = []

    def validation_step(self, batch: dict, batch_idx: int):
        img = batch["img"]
        outputs = self(img)
        losses = self.sup_loss(outputs=outputs, targets=batch)

        return_dict = {}
        return_dict["image_id"] = batch["image_id"]
        for task in self.task_names:
            for idx, subtask_name in enumerate(self.subtask_names[task]):
                return_dict[f"{subtask_name}_score"] = (
                    torch.sigmoid(outputs[task][:, idx]).to(dtype=torch.float32).cpu().numpy()
                )
                return_dict[f"{subtask_name}_label"] = (
                    batch[task][:, idx].to(dtype=torch.float32).cpu().numpy()
                )
                return_dict[f"{subtask_name}_mask"] = (
                    batch[f"{task}_mask"][:, idx].to(dtype=torch.float32).cpu().numpy()
                )

            return_dict[f"{task}_loss"] = [losses[task].to(dtype=torch.float32).cpu().numpy()]

        self.all_results.append(return_dict)
        return return_dict

    def on_test_epoch_start(self):
        """Initialize the all_results lists for testing."""
        self.all_results = []

    def on_test_epoch_end(self):
        """Loop through all labels, compute and log their respective AUCs."""

        total_loss = 0
        all_results = {
            k: np.concatenate([result[k] for result in self.all_results])
            for k in self.all_results[0]
            if "loss" not in k
        }
        df = pd.DataFrame.from_dict(all_results)
        df.to_csv("test.csv", index=False)

        log_dict = {}

        for task in self.task_names:
            for subtask_name in self.subtask_names[task]:
                preds = torch.from_numpy(df[f"{subtask_name}_score"].values)
                labels = torch.from_numpy(df[f"{subtask_name}_label"].values)
                mask = torch.from_numpy(df[f"{subtask_name}_mask"].values)

                preds = preds[mask == 1]
                labels = labels[mask == 1]

                if preds.shape[0] == 0:
                    continue

                auc = self.auroc(preds, labels)  # Assuming self.auroc is initialized in __init__

                log_dict[f"test_{subtask_name}_auc"] = float(auc)

            loss = np.array([d[f"{task}_loss"] for d in self.all_results]).mean()
            total_loss += loss

            log_dict[f"test_{task}_loss"] = float(loss)


        log_dict["test_total_loss"] = float(total_loss)

        self.log_dict(log_dict, sync_dist=True)

        self.all_results = []

    def test_step(self, batch, batch_idx):
        """Computes the loss for the given batch of test data."""
        img = batch["img"]
        outputs = self(img)

        losses = self.sup_loss(outputs=outputs, targets=batch)

        return_dict = {}
        return_dict["image_id"] = batch["image_id"]
        for task in self.task_names:
            for idx, subtask_name in enumerate(self.subtask_names[task]):
                return_dict[f"{subtask_name}_score"] = (
                    torch.sigmoid(outputs[task][:, idx]).to(dtype=torch.float32).cpu().numpy()
                )
                return_dict[f"{subtask_name}_label"] = (
                    batch[task][:, idx].to(dtype=torch.float32).cpu().numpy()
                )
                return_dict[f"{subtask_name}_mask"] = (
                    batch[f"{task}_mask"][:, idx].to(dtype=torch.float32).cpu().numpy()
                )

            return_dict[f"{task}_loss"] = losses[task].to(dtype=torch.float32).cpu().numpy()

        self.all_results.append(return_dict)
        return return_dict

    def configure_optimizers(self):
        """Define optimizer and scheduler."""
        optimizer = self.hparams.optimizer(self.parameters())
        scheduler = CosineWarmup(optimizer, self.warmup_steps, self.end_steps)
        sch_config = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return [optimizer], [sch_config]