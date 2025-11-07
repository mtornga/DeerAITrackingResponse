#!/usr/bin/env python3
"""
Download EfficientNet-B3 pretrained weights and class mapping.
Creates:
  models/v0.1_efficientnet-b3_compiled.pt
  models/index_to_name.json
"""

import torch
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import json, os

def main():
    os.makedirs("models", exist_ok=True)

    print("⏬ Loading EfficientNet-B3 pretrained weights from torchvision...")
    weights = EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=weights)
    model.eval()

    model_path = "models/v0.1_efficientnet-b3_compiled.pt"
    json_path = "models/index_to_name.json"

    # Save model weights
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved model weights to: {model_path}")

    # Save index-to-name mapping
    with open(json_path, "w") as f:
        json.dump({i: name for i, name in enumerate(weights.meta['categories'])}, f, indent=2)
    print(f"✅ Saved class index mapping to: {json_path}")

    # Show a sample of classes
    print("\n🔎 Sample classes:")
    print(", ".join(list(weights.meta['categories'])[:10]) + " ...")

if __name__ == "__main__":
    main()