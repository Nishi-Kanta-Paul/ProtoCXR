"""
src/inference.py
================
Run ProtoCXR on new CXR images and output predictions + explanations.
"""

import os
from typing import Dict, List, Optional

import torch
from PIL import Image

from src.config import Config
from src.dataset import get_transforms
from src.explainability import get_prototype_explanation, visualize_explanation
from src.model import ProtoCXR
from src.utils import save_json


def load_model(checkpoint_path: str, config: Config, device: torch.device) -> ProtoCXR:
    """Load a trained ProtoCXR from a checkpoint file.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        config:          ``Config`` instance used to instantiate the model.
        device:          Compute device.

    Returns:
        Loaded :class:`~src.model.ProtoCXR` in ``eval`` mode.
    """
    model = ProtoCXR(
        num_classes=config.NUM_CLASSES,
        num_proto=config.NUM_PROTO,
        feat_dim=config.FEAT_DIM,
        backbone_name=config.BACKBONE,
        backbone_pretrained=False,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.to(device)
    model.eval()
    return model


def predict(
    model: ProtoCXR,
    image_path: str,
    config: Config,
    device: torch.device,
    label_names: List[str],
    threshold: float = 0.5,
) -> Dict:
    """Run inference and generate explanations for a single image.

    Args:
        model:       Trained :class:`~src.model.ProtoCXR` in eval mode.
        image_path:  Path to the input CXR image (PNG/JPEG)
        config:      ``Config`` instance.
        device:      Compute device.
        label_names: List of class labels corresponding to output logits.
        threshold:   Probability threshold for a positive finding.

    Returns:
        Dictionary with keys:

        - ``"predictions"``: ``{label: probability}`` for all classes.
        - ``"positive_findings"``: List of labels where prob > threshold.
        - ``"explanations"``: ``{label: explanation_dict}`` for each
          positive finding.
    """
    image_pil = Image.open(image_path).convert("RGB")
    transform = get_transforms(train=False, image_size=config.IMAGE_SIZE, config=config)

    img_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(img_tensor, return_sim_maps=False)
        probs = torch.sigmoid(logits).cpu().squeeze(0).numpy()

    res_preds: Dict[str, float] = {}
    positive_findings: List[str] = []

    for i, name in enumerate(label_names):
        p = float(probs[i])
        res_preds[name] = p
        if p > threshold:
            positive_findings.append(name)

    explanations: Dict[str, Dict] = {}
    for name in positive_findings:
        class_idx = label_names.index(name)
        exp = get_prototype_explanation(model, img_tensor, class_idx, device)
        # convert numpy arrays to lists for JSON serialization
        # or leave as dict if called internally (caller handles save)
        explanations[name] = exp

    return {
        "predictions": res_preds,
        "positive_findings": positive_findings,
        "explanations": explanations,
    }


def batch_inference(
    model: ProtoCXR,
    image_dir: str,
    config: Config,
    device: torch.device,
    label_names: List[str],
    save_dir: str,
) -> None:
    """Run inference on all images in a directory.

    Saves predictions to ``<save_dir>/predictions/<filename>.json``
    and explanation figures to ``<save_dir>/explanations/<filename>_<label>.png``.

    Args:
        model:       Trained :class:`~src.model.ProtoCXR`.
        image_dir:   Path to directory containing input CXR images.
        config:      ``Config`` instance.
        device:      Compute device.
        label_names: List of class labels (e.g. from CheXpert or NIH).
        save_dir:    Output directory path.
    """
    valid_exts = {".png", ".jpg", ".jpeg"}
    images = [f for f in os.listdir(image_dir) if os.path.splitext(f.lower())[1] in valid_exts]

    if not images:
        print(f"No valid images found in {image_dir}.")
        return

    pred_dir = os.path.join(save_dir, "predictions")
    exp_dir  = os.path.join(save_dir, "explanations")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Running inference on {len(images)} images from {image_dir}...")

    # For mean calculation
    class_prob_sums = {n: 0.0 for n in label_names}

    for filename in images:
        path = os.path.join(image_dir, filename)
        base = os.path.splitext(filename)[0]

        res = predict(model, path, config, device, label_names)

        # Accumulate prob sums
        for n in label_names:
            class_prob_sums[n] += res["predictions"][n]

        # Save JSON (remove raw ndarrays from explanation dict for serialization)
        json_payload = {
            "file": filename,
            "predictions": res["predictions"],
            "positive_findings": res["positive_findings"],
            "explanations": {}
        }

        for finding, exp in res["explanations"].items():
            json_payload["explanations"][finding] = {
                "proto_idx": exp["proto_idx"],
                "sim_score": exp["sim_score"],
            }

            # Visualize and save the explanation figure
            img_np = __import__('numpy').array(Image.open(path).convert("RGB"))
            fig_path = os.path.join(exp_dir, f"{base}_{finding.replace(' ', '_')}.png")
            visualize_explanation(img_np, exp, finding, save_path=fig_path, config=config)

        save_json(json_payload, os.path.join(pred_dir, f"{base}.json"))

    print("\nInference Summary")
    print("-----------------")
    print(f"Total processed: {len(images)}")
    print("Mean Confidence per Class:")
    for n in label_names:
        avg = class_prob_sums[n] / len(images)
        print(f"  - {n:<24}: {avg:.4f}")
    print(f"\nOutputs saved to {save_dir}")
