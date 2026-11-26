"""Fit the team's PMData encoder and stress classifier with PyHealth Trainer."""

from contextlib import chdir, nullcontext

import numpy as np
import torch
from torch.utils.data import Subset

from pyhealth.datasets import get_dataloader
from pyhealth.models import ClipFinetuneStressPrediction, ClipMultimodalEncoderPretraining
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed

from .pmdata_label_splits import FEATURE_KEYS


class _ShuffledBatches:
    """Keep every day while avoiding singleton contrastive and BatchNorm batches."""

    def __init__(self, count, size, seed):
        if min(count, size) < 2:
            raise ValueError("Training needs at least two days per batch.")
        self.count = count
        self.rng = np.random.default_rng(seed)
        self.cuts = np.arange(size, count, size)
        if count % size == 1:
            if size == 2:
                self.cuts = self.cuts[:-1]
            else:
                self.cuts[-1] -= 1

    def __len__(self):
        return len(self.cuts) + 1

    def __iter__(self):
        return iter(np.split(self.rng.permutation(self.count), self.cuts))


def training_loader(dataset, indices, size, seed):
    view = Subset(dataset, indices)
    batches = _ShuffledBatches(len(view), size, seed)
    return get_dataloader(view, batch_size=1, batch_sampler=batches, seed=seed)


def new_encoder(dataset, seed, device, bert_dir):
    set_seed(seed)
    # model loads the bert-base-uncased.
    with chdir(bert_dir.parent) if bert_dir else nullcontext():
        return ClipMultimodalEncoderPretraining(dataset=dataset).to(device)


class _CachedContrastive(torch.nn.Module):
    """Encode frozen BERT prompts in small batches, then train both projection heads."""

    def __init__(self, encoder, dataset, text_batch_size):
        super().__init__()
        self.encoder = encoder
        self.text_vectors = {}
        encoder.text_encoder.eval()
        with torch.no_grad():
            for batch in get_dataloader(dataset, text_batch_size, shuffle=False):
                keys = zip(batch["patient_id"], batch["record_id"])
                vectors = encoder.generate_text_embeddings(batch["texts"]).cpu()
                self.text_vectors.update(zip(keys, vectors, strict=True))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.text_encoder.eval()
        return self

    def forward(self, **batch):
        keys = zip(batch["patient_id"], batch["record_id"])
        vectors = torch.stack([self.text_vectors[key] for key in keys]).to(self.encoder.device)
        text = self.encoder.txt_proj_head(vectors)
        wearable = self.encoder.encode(**{key: batch[key] for key in FEATURE_KEYS})
        loss = self.encoder.contrastive_loss(text, wearable)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite contrastive loss.")
        return {"loss": loss}


class _StressClassifier(ClipFinetuneStressPrediction):
    def forward(self, *, stress, **batch):
        return super().forward(stress=stress, **{key: batch[key] for key in FEATURE_KEYS})


def pretrain(data, args, seed, device):
    encoder = new_encoder(data.pretrain, seed, device, args.bert_dir)
    model = _CachedContrastive(encoder, data.pretrain, args.text_batch_size)
    loader = training_loader(data.pretrain, range(len(data.pretrain)), args.batch_size, seed)
    trainer = Trainer(model, device=str(device), enable_logging=False)
    trainer.train(loader, epochs=args.pretrain_epochs, optimizer_params={"lr": args.lr})
    return encoder.cpu().state_dict()


def classify(data, indices, args, seed, device, pretrained=None):
    encoder = new_encoder(data.pretrain, seed, device, args.bert_dir)
    if pretrained is not None:
        encoder.load_state_dict(pretrained)
    set_seed(seed)  # Start both methods with the same classification head.
    model = _StressClassifier(encoder, data.train, freeze_multimodal_encoder=False)
    trainer = Trainer(model, metrics=["roc_auc"], device=str(device), enable_logging=False)
    train_loader = training_loader(data.train, indices, args.batch_size, seed)
    val_loader = get_dataloader(data.val, args.batch_size, shuffle=False)
    epochs = args.supervised_epochs if pretrained is None else args.finetune_epochs
    trainer.train(
        train_loader, val_dataloader=val_loader, epochs=epochs,
        optimizer_params={"lr": args.lr}, monitor="roc_auc", monitor_criterion="max",
    )
    # Trainer restored the weights with the highest validation AUROC.
    test_loader = get_dataloader(data.test, args.batch_size, shuffle=False)
    return {
        "val_auroc": float(trainer.evaluate(val_loader)["roc_auc"]),
        "test_auroc": float(trainer.evaluate(test_loader)["roc_auc"]),
    }
