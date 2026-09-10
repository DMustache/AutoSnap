import csv
import json
import tempfile
import zipfile
import logging
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from app.core.config import Settings

logger = logging.getLogger(__name__)


class ImageModel(Protocol):
    def predict(self, image: Image.Image) -> Any:
        """Run inference for one decoded image."""


class ModelManager:
    """Owns the process-wide model and keeps model-specific code out of routes."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: ImageModel | None = None
        self.load_error: str | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self.source: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load the trained artifact at startup.

        Replace ``_load_from_path`` when the trained model format is known.
        """
        try:
            if self.settings.model_path and self.settings.model_path.is_file():
                model_source = self.settings.model_path
                self.source = str(model_source)
            elif self.settings.hf_model_repo:
                model_source = self._download_from_huggingface()
                self.source = f"hf://{self.settings.hf_model_repo}@{self.settings.hf_model_revision}"
            else:
                raise FileNotFoundError(
                    f"Local model does not exist: {self.settings.model_path}; HF_MODEL_REPO is not configured"
                )
            self.model = self._load_from_path(model_source)
            self.load_error = None
        except Exception as exc:
            self.load_error = str(exc)
            logger.exception("Failed to load AutoSnap model")

    def _download_from_huggingface(self) -> Path:
        """Download the model bundle to the local HF cache and return its Keras path."""
        from huggingface_hub import snapshot_download

        snapshot_directory = Path(
            snapshot_download(
                repo_id=self.settings.hf_model_repo,
                repo_type="model",
                revision=self.settings.hf_model_revision,
                cache_dir=self.settings.hf_cache_directory,
                allow_patterns=[
                    "car_classifier.keras",
                    "class_metadata.csv",
                    "price_lookup.csv",
                    "inference_config.json",
                    "manifest.json",
                ],
            )
        )
        model_path = snapshot_directory / "car_classifier.keras"
        if not model_path.is_file():
            raise FileNotFoundError(f"Hugging Face repository has no {model_path.name}")
        return model_path

    def _load_from_path(self, path: Path) -> ImageModel:
        """Load the supplied TensorFlow bundle and its class/price metadata."""
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow and NumPy are required to load the model") from exc

        if path.suffix.lower() == ".zip":
            self._temporary_directory = tempfile.TemporaryDirectory(prefix="autosnap-model-")
            bundle_dir = Path(self._temporary_directory.name)
            with zipfile.ZipFile(path) as archive:
                archive.extractall(bundle_dir)
        else:
            bundle_dir = path.parent

        config_path = bundle_dir / "inference_config.json"
        metadata_path = bundle_dir / "class_metadata.csv"
        prices_path = bundle_dir / "price_lookup.csv"
        model_path = path if path.suffix.lower() == ".keras" else bundle_dir / "car_classifier.keras"

        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
        with metadata_path.open(newline="", encoding="utf-8") as file:
            classes = list(csv.DictReader(file))
        with prices_path.open(newline="", encoding="utf-8") as file:
            prices = list(csv.DictReader(file))

        expected_classes = config["classification"]["number_of_classes"]
        if len(classes) != expected_classes:
            raise RuntimeError(f"Metadata contains {len(classes)} classes, config expects {expected_classes}")

        keras_model = tf.keras.models.load_model(model_path, compile=False)
        return TensorFlowCarModel(keras_model, config, classes, prices, np, tf)

    def predict(self, image: Image.Image) -> Any:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        return self.model.predict(image)

    def unload(self) -> None:
        self.model = None
        self.source = None
        if self._temporary_directory:
            self._temporary_directory.cleanup()
            self._temporary_directory = None


class TensorFlowCarModel:
    def __init__(self, model: Any, config: dict[str, Any], classes: list[dict[str, str]], prices: list[dict[str, str]], np: Any, tf: Any):
        self.model = model
        self.config = config
        self.classes = classes
        self.prices = prices
        self.np = np
        self.tf = tf

    def predict(self, image: Image.Image) -> dict[str, Any]:
        # Keep values in 0..255. EfficientNet's rescaling layer is inside the saved model.
        array = self.np.asarray(image.convert("RGB"), dtype=self.np.float32)
        image_config = self.config["image"]
        array = self.tf.image.resize_with_pad(array, image_config["height"], image_config["width"])
        probabilities = self.model.predict(array[None, ...], verbose=0)[0]

        top_indices = self.np.argsort(probabilities)[::-1][:2]
        top_index = int(top_indices[0])
        top5_indices = self.np.argsort(probabilities)[::-1][:5]
        top5 = [
            {
                "label": self.classes[int(index)]["class_name"],
                "confidence": float(probabilities[int(index)]),
            }
            for index in top5_indices
        ]
        confidence = float(probabilities[top_index])
        second_confidence = float(probabilities[top_indices[1]]) if len(top_indices) > 1 else 0.0
        margin = confidence - second_confidence
        classification_config = self.config["classification"]
        accepted = confidence >= classification_config["confidence_threshold"] and margin >= classification_config["margin_threshold"]

        if not accepted:
            return {
                "known": False,
                "label": classification_config["unknown_label"],
                "confidence": confidence,
                "margin": margin,
                "top5": top5,
                "price": None,
            }

        car = self.classes[top_index]
        return {
            "known": True,
            "label": car["class_name"],
            "confidence": confidence,
            "margin": margin,
            "top5": top5,
            "class_index": top_index,
            "make": car["make"],
            "model": car["model"],
            "year": int(car["year"]),
            "price": self._find_price(car),
        }

    def _find_price(self, car: dict[str, str]) -> dict[str, Any] | None:
        make = car["make_normalized"]
        family = car["model_family"]
        year = int(car["year"])
        rules = self.config["price"]["minimum_listings"]
        fallback = [
            ("family_year", lambda row: row["manufacturer_normalized"] == make and row["matched_model_family"] == family and row["year"] == str(year)),
            ("family", lambda row: row["manufacturer_normalized"] == make and row["matched_model_family"] == family and row["year"] == ""),
            ("make_year", lambda row: row["manufacturer_normalized"] == make and row["matched_model_family"] == "" and row["year"] == str(year)),
            ("make", lambda row: row["manufacturer_normalized"] == make and row["matched_model_family"] == "" and row["year"] == ""),
        ]
        for level, matches in fallback:
            candidates = [row for row in self.prices if row["level"] == level and matches(row)]
            if candidates and int(float(candidates[0]["listing_count"])) >= rules[level]:
                row = candidates[0]
                return {
                    "currency": self.config["price"]["currency"],
                    "strategy": self.config["price"]["strategy"],
                    "level": level,
                    "median": float(row["median_price"]),
                    "lower": float(row["lower_price"]),
                    "upper": float(row["upper_price"]),
                    "listing_count": int(float(row["listing_count"])),
                }
        return None
