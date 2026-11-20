"""CLIP-style multimodal encoder for PMData stress pretraining.

Aligns text prompts with Fitbit time series plus tabular wellness/static
features using a contrastive loss. Batch keys match
:class:`~pyhealth.tasks.StressPredictionPMData`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.resnet1d import ResNet1D
from pyhealth.models.text_embedding import TextEmbedding
from pyhealth.models.transformer import TransformerLayer


class ClipMultimodalEncoderPretraining(BaseModel):
    """Contrastive encoder over hourly series, tabular fields, and text.

    Expects batches from :class:`~pyhealth.tasks.StressPredictionPMData`:

    - ``heart_rate``, ``steps``, ``calories``, ``distance``: ``(B, T)``
    - ``wellness``: ``(B, 6)``, ``static``: ``(B, 3)``
    - ``texts``: list of prompt strings (from the task)
    - ``stress``: optional binary label (unused by the contrastive loss)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        time_series_feature_size: int = 4,
        in_channels: int = 1,
        tabular_hidden_size: int = 128,
        proj_dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__(dataset=dataset)
        self.temperature = float(temperature)
        self.time_series_encoder = TransformerLayer(time_series_feature_size)
        self.tabular_encoder = ResNet1D(in_channels, tabular_hidden_size)
        self.text_encoder = TextEmbedding(
            model_name="bert-base-uncased",
            embedding_dim=768,
            return_mask=True,
            freeze=True
        )

        for p in self.text_encoder.fc.parameters():
            p.requires_grad = False
        self.txt_proj_head = nn.Linear(768, proj_dim)
        self.seq_proj_head = nn.Linear(
            time_series_feature_size + tabular_hidden_size, proj_dim
        )


    def generate_text_embeddings(self, desc_batch: list[str]) -> torch.Tensor:
        """Encodes prompts to one vector each, shape ``(B, 768)``."""
        embeddings, mask = self.text_encoder(desc_batch)
        weights = mask.unsqueeze(-1).to(embeddings.dtype)
        summed = (embeddings * weights).sum(dim=1)
        return summed / weights.sum(dim=1).clamp(min=1.0)

    def constrastive_loss(
        self,
        text_projection: torch.Tensor,
        combined_projection: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric CLIP InfoNCE between text and sequence projections."""
        text_projection = nn.functional.normalize(text_projection, dim=-1)
        combined_projection = nn.functional.normalize(combined_projection, dim=-1)
        logits = text_projection @ combined_projection.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_t2s = nn.functional.cross_entropy(logits, labels)
        loss_s2t = nn.functional.cross_entropy(logits.T, labels)
        return 0.5 * (loss_t2s + loss_s2t)

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass using StressPredictionPMData batch keys."""
        heart_rate = kwargs["heart_rate"].to(self.device)
        steps = kwargs["steps"].to(self.device)
        calories = kwargs["calories"].to(self.device)
        distance = kwargs["distance"].to(self.device)
        wellness = kwargs["wellness"].to(self.device)
        static = kwargs["static"].to(self.device)
        texts = kwargs["texts"]

        # (B, T, 4) for TransformerLayer; replace NaNs from missing hours/channels
        time_series = torch.stack(
            [heart_rate, steps, calories, distance], dim=-1
        )
        time_series = torch.nan_to_num(time_series, nan=0.0)
        _, time_series_encoding = self.time_series_encoder(time_series)

        # (B, 1, 9) for ResNet1D with in_channels=1
        tabular = torch.cat([wellness, static], dim=-1)
        tabular = torch.nan_to_num(tabular, nan=0.0).unsqueeze(1)
        tabular_encoding = self.tabular_encoder(tabular)

        combined_embedding = torch.cat(
            (time_series_encoding, tabular_encoding), dim=-1
        )
        text_embeddings = self.generate_text_embeddings(texts)
        text_projection = self.txt_proj_head(text_embeddings)
        combined_projection = self.seq_proj_head(combined_embedding)

        loss = self.constrastive_loss(text_projection, combined_projection)
        results: dict[str, torch.Tensor] = {
            "loss": loss,
            "text_projection": text_projection,
            "combined_projection": combined_projection,
        }
        if "stress" in kwargs and kwargs["stress"] is not None:
            results["y_true"] = kwargs["stress"].to(self.device)
        return results
