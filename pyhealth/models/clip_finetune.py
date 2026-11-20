class ClipFinetuneStressPrediction(BaseModel):
    """Finetune the CLIP model for stress prediction."""

    def __init__(
        self,
        clip_model: ClipMultimodalEncoderPretraining,
        dataset: SampleDataset,
        proj_dim: int = 128,
        freeze_multimodal_encoder: bool = True,
    ):
        super().__init__(dataset=dataset)
        self.multimodal_encoder = clip_model
        self.stress_prediction_head = nn.Linear(proj_dim, 1)

        if freeze_multimodal_encoder:
            for param in self.multimodal_encoder.parameters():
                param.requires_grad = False

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass using StressPredictionPMData batch keys."""
        heart_rate = kwargs["heart_rate"].to(self.device)
        steps = kwargs["steps"].to(self.device)
        calories = kwargs["calories"].to(self.device)
        distance = kwargs["distance"].to(self.device)
        wellness = kwargs["wellness"].to(self.device)
        static = kwargs["static"].to(self.device)
        texts = kwargs["texts"]

        embeddings = self.multimodal_encoder(heart_rate, steps, calories, distance, wellness, static, texts)
        stress = self.stress_prediction_head(embeddings)

        loss = nn.functional.binary_cross_entropy_with_logits(stress, kwargs["stress"].to(self.device))
        return {"loss": loss, "stress": stress, "y_true": kwargs["stress"].to(self.device)}
