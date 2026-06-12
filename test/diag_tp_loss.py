"""Reproduce the loss=inf issue in TP mode with realistic inputs.

Runs the TP model (TP=2) with FP16 and real tokenized data.
Inspects intermediate losses and identifies which produces inf.
"""
import os
import sys
import torch

sys.path.insert(0, "/home/wlx/HippoLM")

from src.models import HippoConfig
from src.models.tp_model import TPHippoModel
from src.models.model import HippoModel


def main():
    torch.manual_seed(0)
    config = HippoConfig()  # full config

    B, T = 2, 256
    input_ids = torch.randint(0, config.vocab_size, (B, T), device="cuda")
    labels = torch.randint(0, config.vocab_size, (B, T), device="cuda")
    labels[:, -5:] = -100  # standard ignore_index

    print("=" * 60)
    print("Test 1: HippoModel (non-TP) loss")
    print("=" * 60)
    model = HippoModel(config).to("cuda").to(torch.float16)
    model.train()
    outputs = model(input_ids, labels=labels)
    print(f"logits max|.|={outputs['logits'].abs().max().item():.4e}")
    print(f"loss: dtype={outputs['loss'].dtype} value={outputs['loss'].item()}")
    del model

    print()
    print("=" * 60)
    print("Test 2: TPHippoModel (TP=1) loss")
    print("=" * 60)
    model = TPHippoModel(config, devices=[0], dtype=torch.float16)
    model.train()
    outputs = model(input_ids, labels=labels)
    print(f"sharded_logits max|.|={outputs['logits'].abs().max().item():.4e}")
    print(f"loss: dtype={outputs['loss'].dtype} value={outputs['loss'].item()}")
    del model


if __name__ == "__main__":
    main()
