pyhealth.models.clip
====================

CLIP-style multimodal models for PMData stress prediction.

:class:`~pyhealth.models.ClipMultimodalEncoderPretraining` aligns text prompts
with Fitbit time series and tabular wellness/static features using a
symmetric contrastive (InfoNCE) loss. Hourly channels are encoded with a
:class:`~pyhealth.models.TransformerLayer`, tabular fields with
:class:`~pyhealth.models.resnet1d.ResNet1D`, and prompts with a frozen
:class:`~pyhealth.models.TextEmbedding` (BERT by default). Batches follow
:class:`~pyhealth.tasks.StressPredictionPMData`.

:class:`~pyhealth.models.ClipFinetuneStressPrediction` fine-tunes a trained
pretraining encoder for binary stress classification. It reuses
``encode()`` for the combined sequence projection and trains a linear head
(optionally with the multimodal encoder frozen).

.. autoclass:: pyhealth.models.ClipMultimodalEncoderPretraining
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ClipFinetuneStressPrediction
    :members:
    :undoc-members:
    :show-inheritance:
