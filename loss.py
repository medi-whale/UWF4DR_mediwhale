import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):

    def __init__(
        self, task_names: list[str], label_smoothing: int, loss_autoweighting: bool
    ) -> None:
        super().__init__()
        self.task_names = task_names
        self.label_smoothing = label_smoothing
        self.loss_autoweighting = loss_autoweighting
        if self.loss_autoweighting:
            self.loss_weights = torch.nn.Parameter(
                torch.ones(len(task_names), requires_grad=True, dtype=torch.float)
            )
        else:
            self.loss_weights = torch.ones(len(task_names), dtype=torch.float)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        
        # Dictionary to store calculated losses
        losses = {}

        # Calculate loss for each label in the experiment
        for i, label in enumerate(self.task_names):
            logit = outputs[label]
            target = targets[label]
            mask = targets[f"{label}_mask"]

            smooth_target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            loss = (
                F.binary_cross_entropy_with_logits(logit, smooth_target, reduction="none") * mask
            ).mean()

            if self.training and self.loss_autoweighting:
                losses[label] = (loss * 0.5 / (self.loss_weights[i] ** 2)) + torch.log(
                    1 + self.loss_weights[i] ** 2
                )
            else:
                losses[label] = loss

        # Return the losses as a tuple in a fixed order
        return losses