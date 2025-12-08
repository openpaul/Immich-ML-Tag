"""Training and inference logic for ML tag classifiers."""

import hashlib
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

from . import api
from .config import ConfigStore, settings
from .embeddings import (
    get_asset_ids_created_since,
    get_asset_ids_without_tag,
    get_embeddings_by_asset_ids,
)

logger = logging.getLogger(__name__)


@dataclass
class BinaryEmbeddingClassifier:
    """A binary classifier for embeddings using Logistic Regression."""

    model: LogisticRegression | None = None

    def train(
        self,
        positive: np.ndarray,
        negative: np.ndarray,
        normalize_embeddings: bool = True,
        C: float = 1.0,
    ) -> float:
        """
        Train the classifier on positive and negative embeddings.

        Args:
            positive: Array of positive example embeddings.
            negative: Array of negative example embeddings.
            normalize_embeddings: Whether to L2-normalize embeddings.
            C: Regularization parameter (inverse of regularization strength).

        Returns:
            Training accuracy.
        """
        if normalize_embeddings:
            positive = normalize(positive)
            negative = normalize(negative)

        X = np.vstack([positive, negative])
        y = np.concatenate(
            [
                np.ones(len(positive), dtype=np.int64),
                np.zeros(len(negative), dtype=np.int64),
            ]
        )

        self.model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            C=C,
            max_iter=5000,
            n_jobs=-1,
        )

        self.model.fit(X, y)
        preds = self.model.predict(X)
        return accuracy_score(y, preds)

    def save(self, path: str | Path) -> None:
        """Save the model to disk."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")

    @classmethod
    def load(cls, path: str | Path) -> "BinaryEmbeddingClassifier":
        """Load a model from disk."""
        path = Path(path)
        model = joblib.load(path / "model.joblib")
        return cls(model=model)

    def predict_proba(
        self,
        embeddings: np.ndarray,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Predict probability of positive class.

        Args:
            embeddings: Array of embeddings to predict.
            normalize_embeddings: Whether to L2-normalize embeddings.

        Returns:
            Array of probabilities for the positive class.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if normalize_embeddings:
            embeddings = normalize(embeddings)

        return self.model.predict_proba(embeddings)[:, 1]

    def predict(
        self,
        embeddings: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Make binary predictions.

        Args:
            embeddings: Array of embeddings to predict.
            threshold: Probability threshold for positive class.

        Returns:
            Array of binary predictions.
        """
        probs = self.predict_proba(embeddings)
        return (probs >= threshold).astype(np.int64)


def compute_training_data_hash(
    positive_asset_ids: list[str],
    manual_negative_asset_ids: list[str],
) -> str:
    """
    Compute a hash of the training data to detect changes.

    Only includes user-tagged positives and manually tagged negatives,
    not the random background samples.

    Args:
        positive_asset_ids: List of positive (user-tagged) asset IDs.
        manual_negative_asset_ids: List of manually tagged negative asset IDs.

    Returns:
        SHA256 hash of the sorted asset ID lists.
    """
    # Sort both lists for consistent hashing
    sorted_positives = sorted(positive_asset_ids)
    sorted_negatives = sorted(manual_negative_asset_ids)

    # Create a string representation
    data = f"pos:{','.join(sorted_positives)}|neg:{','.join(sorted_negatives)}"

    # Compute hash
    return hashlib.sha256(data.encode()).hexdigest()


def get_negative_samples(
    tag_id: str,
    positive_asset_ids: list[str],
    contrast_tag_id: str | None = None,
    max_ratio: int = 4,
) -> list[str]:
    """
    Get negative sample asset IDs for training.

    Args:
        tag_id: The tag ID to get negatives for.
        positive_asset_ids: List of positive asset IDs to exclude.
        contrast_tag_id: Optional tag ID for explicit negative examples.
        max_ratio: Maximum ratio of negatives to positives.

    Returns:
        List of negative asset IDs.
    """
    # Get assets with the tag to exclude
    tagged_assets = set(api.get_assets_by_tag([tag_id]))

    # Get all potential negative assets
    negative_asset_ids = get_asset_ids_without_tag(
        tagged_assets, exclude_asset_ids=set(positive_asset_ids)
    )

    # Limit to max_ratio * positives
    if len(negative_asset_ids) > max_ratio * len(positive_asset_ids):
        negative_asset_ids = random.sample(
            negative_asset_ids, max_ratio * len(positive_asset_ids)
        )

    # Add explicit contrast examples if available
    if contrast_tag_id:
        contrast_assets = api.get_assets_by_tag([contrast_tag_id])
        negative_asset_ids.extend(contrast_assets)

    return negative_asset_ids


def train_single_tag(
    tag: dict,
    contrast_tag_id: str,
    config_store: ConfigStore,
    min_samples: int = 10,
    threshold: float = 0.5,
    force: bool = False,
) -> bool:
    """
    Train a classifier for a single tag and run inference.

    Args:
        tag: The tag dictionary from the API.
        contrast_tag_id: ID of the contrast (negative examples) parent tag.
        config_store: Configuration store instance.
        min_samples: Minimum positive samples required.
        threshold: Prediction threshold for tagging.
        force: If True, retrain even if training data hasn't changed.

    Returns:
        True if training was successful.
    """
    tag_name = tag["name"]
    tag_id = tag["id"]
    ml_tag_name = f"{tag_name}{settings.ml_tag_suffix}"

    logger.info(f"Processing tag '{tag_name}' (id: {tag_id})")

    # Get ML tag if it exists
    ml_tag = api.get_tag_by_name(ml_tag_name)
    ml_tag_id = ml_tag["id"] if ml_tag else None

    # Get or create negative examples tag under contrast parent
    try:
        api.create_tag(tag_name, contrast_tag_id)
    except ValueError:
        logger.debug(f"Negative tag '{tag_name}' already exists under contrast parent")

    negative_parent_tag = api.get_tag_by_name(tag_name, parent_id=contrast_tag_id)
    if not negative_parent_tag:
        logger.warning(f"Could not find/create negative tag for '{tag_name}'")
        return False
    negative_tag_id = negative_parent_tag["id"]

    # Get positive asset IDs
    positive_asset_ids = api.get_assets_by_tag([tag_id])

    # Exclude assets only tagged with the predicted tag (not manually tagged)
    if ml_tag_id:
        ml_tagged_assets = set(api.get_assets_by_tag([ml_tag_id]))
        positive_asset_ids = list(set(positive_asset_ids) - ml_tagged_assets)

    logger.info(f"Found {len(positive_asset_ids)} positive assets")

    if len(positive_asset_ids) < min_samples:
        logger.info(f"Skipping '{tag_name}': less than {min_samples} samples")
        return False

    # Get manually tagged negative examples (contrast examples for this tag)
    manual_negative_asset_ids = api.get_assets_by_tag([negative_tag_id])

    # Compute hash of training data (user-tagged positives + manual negatives)
    current_hash = compute_training_data_hash(
        positive_asset_ids, manual_negative_asset_ids
    )
    stored_hash = config_store.get_training_data_hash(tag_name)

    if not force and stored_hash == current_hash:
        logger.info(
            f"Skipping '{tag_name}': training data unchanged (hash: {current_hash[:12]}...)"
        )
        return False

    logger.info(
        f"Training data changed for '{tag_name}' (new hash: {current_hash[:12]}...)"
    )

    # Get negative samples (includes manual negatives + random background samples)
    negative_asset_ids = get_negative_samples(
        tag_id, positive_asset_ids, negative_tag_id
    )
    logger.info(
        f"Selected {len(negative_asset_ids)} negative assets (including {len(manual_negative_asset_ids)} manual)"
    )

    # Get embeddings
    all_asset_ids = positive_asset_ids + negative_asset_ids
    embeddings_dict = get_embeddings_by_asset_ids(all_asset_ids)

    positive_embeddings = np.array(
        [embeddings_dict[aid] for aid in positive_asset_ids if aid in embeddings_dict]
    )
    negative_embeddings = np.array(
        [embeddings_dict[aid] for aid in negative_asset_ids if aid in embeddings_dict]
    )

    logger.info(
        f"Training with {len(positive_embeddings)} positive and "
        f"{len(negative_embeddings)} negative embeddings"
    )

    # Train classifier
    classifier = BinaryEmbeddingClassifier()
    accuracy = classifier.train(positive_embeddings, negative_embeddings)
    logger.info(f"Trained model with accuracy: {accuracy:.4f}")

    # Save model with training data hash
    model_path = settings.ml_resource_path / ml_tag_name
    classifier.save(model_path)
    config_store.register_model(tag_name, model_path, current_hash)
    logger.info(f"Saved model to {model_path}")

    # Manage ML tag (delete and recreate to clear old predictions)
    if ml_tag_id:
        api.delete_tag(ml_tag_id)
        # Wait for deletion to propagate
        for _ in range(5):
            time.sleep(2)
            try:
                api.create_tag(ml_tag_name, tag_id)
                break
            except ValueError:
                logger.debug("Waiting for tag deletion to propagate...")
    else:
        api.create_tag(ml_tag_name, tag_id)

    # Get the new ML tag ID
    ml_tag = api.get_tag_by_name(ml_tag_name)
    if not ml_tag:
        logger.error(f"Failed to create ML tag '{ml_tag_name}'")
        return False
    ml_tag_id = ml_tag["id"]

    # Run inference on all untagged assets
    tagged_assets = set(api.get_assets_by_tag([tag_id]))
    inference_asset_ids = get_asset_ids_without_tag(
        tagged_assets, exclude_asset_ids=set(positive_asset_ids)
    )
    logger.info(f"Running inference on {len(inference_asset_ids)} assets")

    if inference_asset_ids:
        inference_embeddings = get_embeddings_by_asset_ids(inference_asset_ids)
        inference_array = np.array(
            [
                inference_embeddings[aid]
                for aid in inference_asset_ids
                if aid in inference_embeddings
            ]
        )

        if len(inference_array) > 0:
            predictions = classifier.predict_proba(inference_array)
            assets_to_tag = [
                inference_asset_ids[i]
                for i, prob in enumerate(predictions)
                if prob >= threshold
            ]

            logger.info(
                f"Tagging {len(assets_to_tag)} assets with '{ml_tag_name}' "
                f"(threshold: {threshold})"
            )

            if assets_to_tag:
                count = api.bulk_tag_assets(assets_to_tag, [ml_tag_id])
                logger.info(f"Tagged {count} assets")

    return True


def run_training(min_samples: int = 10, threshold: float = 0.5, force: bool = False):
    """
    Run training for all eligible tags.

    Args:
        min_samples: Minimum positive samples required for training.
        threshold: Prediction threshold for tagging.
        force: If True, retrain all models even if training data hasn't changed.
    """
    config_store = ConfigStore()

    # Ensure contrast tag exists
    try:
        api.create_tag(settings.contrast_tag)
    except ValueError:
        pass  # Already exists

    contrast_tag = api.get_tag_by_name(settings.contrast_tag)
    if not contrast_tag:
        raise RuntimeError(
            f"Could not find/create contrast tag '{settings.contrast_tag}'"
        )
    contrast_tag_id = contrast_tag["id"]

    # Get all tags
    tags = api.get_tags()

    trained_count = 0
    for tag in tags:
        tag_name = tag["name"]

        # Skip predicted tags
        if tag_name.endswith(settings.ml_tag_suffix):
            logger.debug(f"Skipping '{tag_name}': is a predicted tag")
            continue

        # Skip contrast tag and its children
        if tag_name == settings.contrast_tag or tag.get("parentId") == contrast_tag_id:
            logger.debug(f"Skipping '{tag_name}': is contrast tag")
            continue

        if train_single_tag(
            tag, contrast_tag_id, config_store, min_samples, threshold, force
        ):
            trained_count += 1

    config_store.last_trained = datetime.now()
    logger.info(f"Training complete. Trained {trained_count} models.")


def run_inference(threshold: float = 0.5, incremental: bool = True):
    """
    Run inference using existing models on new assets.

    Args:
        threshold: Prediction threshold for tagging.
        incremental: If True, only process assets added since last inference.
    """
    config_store = ConfigStore()
    models = config_store.get_models()

    if not models:
        logger.warning("No trained models found. Run training first.")
        return

    # Get timestamp for incremental mode
    since_timestamp = None
    if incremental:
        since_timestamp = config_store.last_inference
        if since_timestamp:
            logger.info(f"Running incremental inference since {since_timestamp}")
        else:
            logger.info("No previous inference timestamp, running full inference")
    else:
        logger.info("Running full inference on all assets")

    # Get candidate assets (created since last inference if incremental)
    candidate_asset_ids = get_asset_ids_created_since(since_timestamp)
    logger.info(f"Found {len(candidate_asset_ids)} candidate assets to process")

    if not candidate_asset_ids:
        logger.info("No new assets to process")
        return

    for model_info in models:
        tag_name = model_info.tag
        model_path = model_info.model_path
        ml_tag_name = f"{tag_name}{settings.ml_tag_suffix}"

        logger.info(f"Running inference for '{tag_name}'")

        # Load model
        try:
            classifier = BinaryEmbeddingClassifier.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model for '{tag_name}': {e}")
            continue

        # Get ML tag
        ml_tag = api.get_tag_by_name(ml_tag_name)
        if not ml_tag:
            logger.warning(f"ML tag '{ml_tag_name}' not found, skipping")
            continue
        ml_tag_id = ml_tag["id"]

        # Get original tag
        original_tag = api.get_tag_by_name(tag_name)
        if not original_tag:
            logger.warning(f"Original tag '{tag_name}' not found, skipping")
            continue

        # Get assets already tagged
        tagged_assets = set(api.get_assets_by_tag([original_tag["id"]]))
        ml_tagged_assets = set(api.get_assets_by_tag([ml_tag_id]))

        # Get assets to run inference on:
        # - Must be in candidate set (new assets if incremental)
        # - Must not already be tagged with original or ML tag
        all_tagged = tagged_assets | ml_tagged_assets
        inference_asset_ids = list(candidate_asset_ids - all_tagged)

        logger.info(f"Running inference on {len(inference_asset_ids)} assets")

        if not inference_asset_ids:
            continue

        # Get embeddings and predict
        embeddings_dict = get_embeddings_by_asset_ids(inference_asset_ids)
        embeddings_array = np.array(
            [
                embeddings_dict[aid]
                for aid in inference_asset_ids
                if aid in embeddings_dict
            ]
        )

        if len(embeddings_array) == 0:
            continue

        predictions = classifier.predict_proba(embeddings_array)
        assets_to_tag = [
            inference_asset_ids[i]
            for i, prob in enumerate(predictions)
            if prob >= threshold
        ]

        logger.info(f"Tagging {len(assets_to_tag)} assets with '{ml_tag_name}'")

        if assets_to_tag:
            count = api.bulk_tag_assets(assets_to_tag, [ml_tag_id])
            logger.info(f"Tagged {count} assets")

    config_store.last_inference = datetime.now()
    logger.info("Inference complete.")
