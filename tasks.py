from abc import ABC, abstractmethod

import pandas as pd
import torch


class Task(ABC):

    def __init__(
        self,
        self_training_mode: str | None = None,
        self_training_threshold: float | None = None,
        self_trainging_temperature: float | None = None,
    ) -> None:
        self._self_training_mode = self_training_mode
        self._self_training_threshold = self_training_threshold
        self._self_training_temperature = self_trainging_temperature

    @property
    @abstractmethod
    def task_labels(self) -> list[str]:
        """Returns task labels."""

    @property
    def task_name(self) -> str:
        return self.task_labels[0]

    @property
    @abstractmethod
    def subtask_names(self) -> list[str]:
        """Returns subtask names."""

    @abstractmethod
    def categorize_function(self) -> dict[str, torch.Tensor]:
        """Returns categorized labels."""

    def _create_label_tensor(self, values: list[bool]) -> torch.Tensor:
        """Creates label tensor."""
        return torch.tensor(values, dtype=torch.float32)

    def _create_mask_tensor(self, value: float) -> torch.Tensor:
        """Creates mask tensor."""
        return torch.tensor(not pd.isnull(value), dtype=torch.float32)

    def make_pseudo_labels(
        self,
        prefix: str,
        predictions: list[float],
    ) -> dict[str, torch.Tensor]:
        """Creates pseudo label."""
        if any([pd.isnull(value) for value in predictions]):
            return {
                prefix: torch.zeros(len(predictions)),
                prefix + "_mask": torch.zeros(len(predictions)),
            }

        predictions_tensor = torch.tensor(predictions, dtype=torch.float32)

        if self._self_training_mode == "soft":
            if self._self_training_temperature:
                logits = torch.logit(predictions_tensor)
                pseudo_labels = torch.sigmoid(logits / self._self_training_temperature)
            else:
                pseudo_labels = predictions_tensor
        elif self._self_training_mode == "hard":
            pseudo_labels = (predictions_tensor > 0.5).to(torch.float32)
        else:
            raise ValueError("Mode must be 'soft' or 'hard'.")

        if self._self_training_threshold:
            masks = (
                torch.logical_or(
                    predictions_tensor < self._self_training_threshold,
                    predictions_tensor > (1 - self._self_training_threshold),
                )
            ).to(torch.float32)
        else:
            masks = torch.ones_like(predictions_tensor)

        return {prefix: pseudo_labels, prefix + "_mask": masks}

    def categorize_labels(self, data: dict[str, float]) -> dict[str, torch.Tensor]:
        """Returns categorized labels or pseudo labels."""
        values = [data[label] for label in self.task_labels]
        if self._self_training_mode:
            if all([not pd.isnull(value) for value in values]):
                return self.categorize_function(*values)
            else:
                scores = [data[sub] for sub in self.subtask_names]
                return self.make_pseudo_labels(self.task_name, scores)
        else:
            return self.categorize_function(*values)


class AgeTask(Task):
    """Age task."""

    @property
    def task_labels(self) -> list[str]:
        """Returns task labels."""
        return [
            "age",
        ]

    @property
    def subtask_names(self) -> list[str]:
        """Returns subtask names."""
        return ["age_40_score", "age_50_score", "age_60_score", "age_70_score"]

    def categorize_function(self, value: float) -> dict[str, torch.Tensor]:
        """Returns categorized labels."""
        age = [
            value > 40,
            value > 50,
            value > 60,
            value > 70,
        ]
        age = self._create_label_tensor(age)
        age_mask = self._create_mask_tensor(value).expand_as(age)

        return {self.task_name: age, self.task_name + "_mask": age_mask}


class GlaucomaTask(Task):
    """Glaucoma task."""

    @property
    def task_labels(self) -> list[str]:
        """Returns task labels."""
        return [
            "Glaucoma",
        ]

    @property
    def subtask_names(self) -> list[str]:
        """Returns subtask names."""
        return [
            "Glaucoma_score",
        ]

    def categorize_function(self, value: float) -> dict[str, torch.Tensor]:
        """Returns categorized labels."""
        Glaucoma = [
            value is True,  # exceptional case
        ]
        Glaucoma = self._create_label_tensor(Glaucoma)
        Glaucoma_mask = self._create_mask_tensor(value).expand_as(Glaucoma)
        return {
            self.task_name: Glaucoma,
            self.task_name + "_mask": Glaucoma_mask,
        }


class DiabeticRetinopathyTask(Task):
    """DiabeticRetinopathy task."""

    @property
    def task_labels(self) -> list[str]:
        """Returns task labels."""
        return [
            "DiabeticRetinopathy",
        ]

    @property
    def subtask_names(self) -> list[str]:
        """Returns subtask names."""
        return [
            "DiabeticRetinopathy_score",
        ]

    def categorize_function(self, value: float) -> dict[str, torch.Tensor]:
        """Returns categorized labels."""
        DiabeticRetinopathy = [
            value is True,  # exceptional case
        ]
        DiabeticRetinopathy = self._create_label_tensor(DiabeticRetinopathy)
        DiabeticRetinopathy_mask = self._create_mask_tensor(value).expand_as(DiabeticRetinopathy)
        return {
            self.task_name: DiabeticRetinopathy,
            self.task_name + "_mask": DiabeticRetinopathy_mask,
        }


class DiabeticMacularEdemaTask(Task):
    """DiabeticMacularEdema task."""

    @property
    def task_labels(self) -> list[str]:
        """Returns task labels."""
        return [
            "DiabeticMacularEdema",
        ]

    @property
    def subtask_names(self) -> list[str]:
        """Returns subtask names."""
        return [
            "DiabeticMacularEdema_score",
        ]

    def categorize_function(self, value: float) -> dict[str, torch.Tensor]:
        """Returns categorized labels."""
        DiabeticMacularEdema = [
            value is True,  # exceptional case
        ]
        DiabeticMacularEdema = self._create_label_tensor(DiabeticMacularEdema)
        DiabeticMacularEdema_mask = self._create_mask_tensor(value).expand_as(DiabeticMacularEdema)
        return {
            self.task_name: DiabeticMacularEdema,
            self.task_name + "_mask": DiabeticMacularEdema_mask,
        }


def get_task(
    task_name: str,
    self_training_mode: str | None = None,
    self_training_threshold: float | None = None,
    self_training_temperature: float | None = None,
) -> Task:
    """Returns task object."""
    if task_name == "age":
        task_class = AgeTask
    elif task_name == "Glaucoma":
        task_class = GlaucomaTask
    elif task_name == "DiabeticRetinopathy":
        task_class = DiabeticRetinopathyTask
    elif task_name == "DiabeticMacularEdema":
        task_class = DiabeticMacularEdemaTask
    else:
        raise ValueError("Task name is not valid")
    return task_class(self_training_mode, self_training_threshold, self_training_temperature)