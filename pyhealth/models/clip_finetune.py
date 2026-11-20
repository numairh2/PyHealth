import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from pyhealth.models.base_model import BaseModel
from pyhealth.datasets import SampleDataset
from pyhealth.models.clip_pretraining import ClipMultimodalEncoderPretraining


class ClipFinetuneStressPrediction(BaseModel):
    """Finetune the CLIP model for stress prediction."""

    def __init__(
        self,
        clip_model: ClipMultimodalEncoderPretraining,
        dataset: SampleDataset,
        proj_dim: int = 128,
        freeze_multimodal_encoder: bool = True
    ):
        super().__init__(dataset=dataset)
        self.multimodal_encoder = clip_model
        self.stress_prediction_head = nn.Linear(proj_dim, 1)

        if freeze_multimodal_encoder:
            for param in self.multimodal_encoder.parameters():
                param.requires_grad = False
    

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass using StressPredictionPMData batch keys."""
        stress = kwargs["stress"].to(self.device)

        embeddings = self.multimodal_encoder.encode(**kwargs)
        stress_pred = self.stress_prediction_head(embeddings)

        y_prob = torch.sigmoid(stress_pred)

        loss = nn.functional.binary_cross_entropy_with_logits(stress_pred, stress)
        return {
            "loss":   loss,   
            "y_prob": y_prob,   
            "y_true": stress,   
            "logit":  stress_pred
        }
