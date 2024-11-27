import timm
import torch
from timm.layers import NormMlpClassifierHead

from tasks import get_task
from .pooling import Pooling
from .transform import GPUTransform

__ALLOWED_TIMM_ARCHS__ = [
    "convnext_small.fb_in22k_ft_in1k_384",
]


class model(torch.nn.Module):
    def __init__(
        self,
        arch: str,
        task_names: list[str],
    ):
        super().__init__()
        if arch not in __ALLOWED_TIMM_ARCHS__:
            raise ValueError(f"arch must be one of {__ALLOWED_TIMM_ARCHS__}")
        self.task_names = task_names

        self.flatten = torch.nn.Flatten(1)
        self.backbone = timm.create_model(
            arch,
            pretrained=True,
            num_classes=0,
            drop_path_rate=0.2,
        )
        feature_dim = self.backbone.feature_info[-1]["num_chs"]

        if compile:
            self.backbone = torch.compile(self.backbone)

        for task_name in self.task_names:
            task = get_task(task_name)
            task_head = NormMlpClassifierHead(
                in_features=feature_dim,
                num_classes=len(task.subtask_names),
                drop_rate=0.2,
            )
            setattr(self, task_name, task_head)

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        feature = self.backbone(x)
        outputs = {task: getattr(self, task)(feature) for task in self.task_names}
        feature = self.flatten(feature)

        # feature's shape will be (b,c)
        outputs["feature"] = feature

        return outputs