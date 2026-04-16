"""Inference utilities for ProtoCXR."""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch
from PIL import Image

from src.config import Config
from src.dataset import get_transforms
from src.explainability import get_prototype_explanation, visualize_explanation
from src.model import ProtoCXR
from src.utils import save_json


def _load_image(image_path: str) -> Image.Image:
    """Load PNG/JPG/DICOM image and return RGB PIL image.

    Args:
        image_path: Input image path.

    Returns:
        PIL RGB image.

    Raises:
        FileNotFoundError: If input path does not exist.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext in {".png", ".jpg", ".jpeg"}:
        return Image.open(image_path).convert("RGB")

    dcm = pydicom.dcmread(image_path)
    array = dcm.pixel_array.astype(np.float32)
    array -= array.min()
    max_val = float(array.max())
    if max_val > 0:
        array /= max_val
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    return Image.fromarray(array).convert("RGB")


def load_model(checkpoint_path: str, config: Config, device: torch.device) -> ProtoCXR:
    """Build ProtoCXR from config and load checkpoint weights.

    Args:
        checkpoint_path: Path to checkpoint file.
        config: Global model config.
        device: Active device.

    Returns:
        Loaded model in eval mode.

    Raises:
        FileNotFoundError: If checkpoint path is missing.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ProtoCXR(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(
    model: ProtoCXR,
    image_path: str,
    config: Config,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Run single-image inference with prototype explanations.

    Args:
        model: Trained ProtoCXR model.
        image_path: Path to PNG/JPG/DICOM image.
        config: Global config with label list.
        device: Active device.
        threshold: Positive-finding probability threshold.

    Returns:
        Dict containing predictions, positive_findings, and explanations.

    Raises:
        None.
    """

    image = _load_image(image_path)
    transform = get_transforms(train=False, image_size=config.IMAGE_SIZE)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(image_tensor, return_sim_maps=False)
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()

    predictions = {label: float(probs[idx]) for idx, label in enumerate(config.LABELS)}
    positive_findings = [label for label, prob in predictions.items() if prob > threshold]

    explanations: Dict[str, Dict[str, object]] = {}
    for label in positive_findings:
        class_idx = config.LABELS.index(label)
        explanations[label] = get_prototype_explanation(model, image_tensor, class_idx, device)

    return {
        "predictions": predictions,
        "positive_findings": positive_findings,
        "explanations": explanations,
    }


def batch_inference(
    model: ProtoCXR,
    image_dir: str,
    config: Config,
    device: torch.device,
    save_dir: str,
) -> None:
    """Run inference for all images in a directory.

    Args:
        model: Trained ProtoCXR model.
        image_dir: Directory with image files.
        config: Global config.
        device: Active device.
        save_dir: Root output directory.

    Returns:
        None.

    Raises:
        FileNotFoundError: If image_dir does not exist.
    """

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Inference directory not found: {image_dir}")

    valid_ext = {".png", ".jpg", ".jpeg", ".dcm"}
    files = [
        name
        for name in sorted(os.listdir(image_dir))
        if os.path.splitext(name)[1].lower() in valid_ext
    ]

    predictions_dir = os.path.join(save_dir, "predictions")
    explanations_dir = os.path.join(save_dir, "explanations")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(explanations_dir, exist_ok=True)

    finding_counter = {label: 0 for label in config.LABELS}

    for filename in files:
        path = os.path.join(image_dir, filename)
        image_id = os.path.splitext(filename)[0]
        result = predict(model, path, config, device)

        for finding in result["positive_findings"]:
            finding_counter[finding] += 1

        serializable_explanations: Dict[str, Dict[str, object]] = {}
        image_np = np.asarray(_load_image(path).convert("RGB"))

        for label, explanation in result["explanations"].items():
            serializable_explanations[label] = {
                "proto_idx": int(explanation["proto_idx"]),
                "class_idx": int(explanation["class_idx"]),
                "sim_score": float(explanation["sim_score"]),
                "spatial_map": np.asarray(explanation["spatial_map"]).tolist(),
                "activation_upsampled": np.asarray(explanation["activation_upsampled"]).tolist(),
                "proto_vector": np.asarray(explanation["proto_vector"]).tolist(),
            }
            fig_path = os.path.join(explanations_dir, f"{image_id}_{label.replace(' ', '_')}.png")
            figure = visualize_explanation(image_np, explanation, label, save_path=fig_path)
            plt.close(figure)

        payload = {
            "predictions": result["predictions"],
            "positive_findings": result["positive_findings"],
            "explanations": serializable_explanations,
        }
        save_json(payload, os.path.join(predictions_dir, f"{image_id}.json"))

    print(f"Total images: {len(files)}")
    print("Findings per class:")
    for label in config.LABELS:
        print(f"  {label}: {finding_counter[label]}")