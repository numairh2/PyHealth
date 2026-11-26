"""Tests for CLIP PMData pretraining and finetune models."""

import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.clip_finetune import ClipFinetuneStressPrediction
from pyhealth.models.clip_pretraining import ClipMultimodalEncoderPretraining

INPUT_SCHEMA = {
    "heart_rate": "tensor",
    "steps": "tensor",
    "calories": "tensor",
    "distance": "tensor",
    "wellness": "tensor",
    "static": "tensor",
    "texts": "text",
}
OUTPUT_SCHEMA = {"stress": "binary"}

SAMPLES = [
    {
        "patient_id": "p0",
        "heart_rate": np.array([60.0, 70.0], dtype=np.float32),
        "steps": np.array([10.0, 5.0], dtype=np.float32),
        "calories": np.array([1.5, np.nan], dtype=np.float32),
        "distance": np.array([100.0, np.nan], dtype=np.float32),
        "wellness": np.array([3, 3, 5, 7, 3, 2], dtype=np.float32),
        "static": np.array([48.0, 195.0, 0.0], dtype=np.float32),
        "texts": "age: 48.0\nfatigue: 3.0\nheart_rate mean is 65.0,",
        "stress": 1,
    },
    {
        "patient_id": "p1",
        "heart_rate": np.array([80.0, 90.0, 85.0], dtype=np.float32),
        "steps": np.array([20.0, 15.0, 10.0], dtype=np.float32),
        "calories": np.array([2.0, 1.0, np.nan], dtype=np.float32),
        "distance": np.array([50.0, 40.0, np.nan], dtype=np.float32),
        "wellness": np.array([2, 4, 4, 6, 2, 1], dtype=np.float32),
        "static": np.array([60.0, 180.0, 0.0], dtype=np.float32),
        "texts": "age: 60.0\nmood: 4.0\nheart_rate mean is 85.0,",
        "stress": 0,
    },
]


def _make_dataset():
    return create_sample_dataset(
        samples=SAMPLES,
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        dataset_name="clip_pmdata_test",
    )


class _StubTextEmbedding(nn.Module):
    """Stand-in TextEmbedding: returns fixed-size embeddings without HF download."""

    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.transformer = nn.Linear(1, 1)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        for p in self.transformer.parameters():
            p.requires_grad = False
        for p in self.fc.parameters():
            p.requires_grad = False

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        batch = len(texts)
        embeddings = torch.zeros(batch, 2, self.embedding_dim)
        mask = torch.ones(batch, 2, dtype=torch.bool)
        return embeddings, mask


class TestClipPretrainingPMData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        cls.proj_dim = 128
        cls.model = ClipMultimodalEncoderPretraining(
            dataset=cls.dataset,
            proj_dim=cls.proj_dim,
            text_encoder_name=None,
        )

    @classmethod
    def tearDownClass(cls):
        cls.dataset.close()

    def _batch(self):
        return next(iter(get_dataloader(self.dataset, batch_size=2, shuffle=False)))

    def test_encode_returns_correct_shape(self):
        batch = self._batch()
        with torch.no_grad():
            emb = self.model.encode(**batch)
        self.assertEqual(emb.shape, (2, self.proj_dim))
        self.assertTrue(torch.isfinite(emb).all())

    def test_forward_returns_correct_shape(self):
        batch = self._batch()
        with torch.no_grad():
            out = self.model(**batch)
        self.assertIn("loss", out)
        self.assertIn("text_projection", out)
        self.assertIn("combined_projection", out)
        self.assertIn("y_true", out)
        self.assertEqual(out["text_projection"].shape, (2, self.proj_dim))
        self.assertEqual(out["combined_projection"].shape, (2, self.proj_dim))
        self.assertEqual(out["loss"].ndim, 0)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(out["y_true"].shape[0], 2)

    def test_text_encoder_is_frozen(self):
        with mock.patch(
            "pyhealth.models.clip_pretraining.TextEmbedding",
            _StubTextEmbedding,
        ):
            model = ClipMultimodalEncoderPretraining(
                dataset=self.dataset,
                proj_dim=self.proj_dim,
                text_encoder_name="stub",
            )
        self.assertIsNotNone(model.text_encoder)
        self.assertTrue(
            all(not p.requires_grad for p in model.text_encoder.parameters())
        )


class TestClipFinetunePMData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        cls.proj_dim = 128
        cls.pretrain = ClipMultimodalEncoderPretraining(
            dataset=cls.dataset,
            proj_dim=cls.proj_dim,
            text_encoder_name=None,
        )
        cls.model = ClipFinetuneStressPrediction(
            clip_model=cls.pretrain,
            dataset=cls.dataset,
            proj_dim=cls.proj_dim,
            freeze_multimodal_encoder=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.dataset.close()

    def test_forward_returns_correct_shape(self):
        batch = next(iter(get_dataloader(self.dataset, batch_size=2, shuffle=False)))
        with torch.no_grad():
            out = self.model(**batch)
        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertIn("logit", out)
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertEqual(out["y_prob"].shape, (2, 1))
        self.assertEqual(out["y_true"].shape[0], 2)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())
        self.assertTrue(
            all(not p.requires_grad for p in self.model.multimodal_encoder.parameters())
        )
        self.assertTrue(
            any(p.requires_grad for p in self.model.stress_prediction_head.parameters())
        )


if __name__ == "__main__":
    unittest.main()
