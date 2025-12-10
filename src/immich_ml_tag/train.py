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

from .api import api
from .config import ConfigStore, settings
from .embeddings import (
    get_all_asset_ids_with_embeddings,
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
        unlabeled: np.ndarray,
        negative: np.ndarray | None = None,
        normalize_embeddings: bool = True,
        C: float = 1.0,
        unlabeled_weight: float = 0.5,
    ) -> float:
        if normalize_embeddings:
            positive = normalize(positive)
            unlabeled = normalize(unlabeled)
            if negative is not None:
                negative = normalize(negative)

        X = [positive, unlabeled]
        y = [
            np.ones(len(positive), dtype=np.int64),
            np.zeros(len(unlabeled), dtype=np.int64),
        ]
        weights = [np.ones(len(positive)), unlabeled_weight * np.ones(len(unlabeled))]

        if negative is not None:
            X.append(negative)
            y.append(np.zeros(len(negative), dtype=np.int64))
            weights.append(np.ones(len(negative)))

        X = np.vstack(X)
        y = np.concatenate(y)
        sample_weight = np.concatenate(weights)

        self.model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            C=C,
            max_iter=5000,
            n_jobs=1,
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        preds = self.model.predict(X)
        return float(accuracy_score(y, preds))

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


def get_unlabeled_samples(
    positive_asset_ids: list[str],
    negative_asset_ids: list[str] | None = None,
    max_ratio: int = 4,
) -> list[str]:
    """
    Get unlabeled sample asset IDs for training.

    Args:
        tag_id: The tag ID to get negatives for.
        positive_asset_ids: List of positive asset IDs to exclude.
        contrast_tag_id: Optional tag ID for explicit negative examples.
        max_ratio: Maximum ratio of negatives to positives.

    Returns:
        List of negative asset IDs.
    """
    # ids we want to exclude
    exclude_asset_ids = set(positive_asset_ids).union(set(negative_asset_ids or []))

    # Get all potential negative assets
    all_asset_ids = list(set(get_all_asset_ids_with_embeddings()) - exclude_asset_ids)

    # Limit to max_ratio * positives
    if len(all_asset_ids) > max_ratio * len(positive_asset_ids):
        all_asset_ids = random.sample(
            all_asset_ids, max_ratio * len(positive_asset_ids)
        )

    return all_asset_ids


def cleanup_old_models(config_store: ConfigStore):
    """
    Remove models from the config store that no longer have corresponding tags.

    Args:
        config_store: Configuration store instance.
    """
    existing_tags = {tag["id"] for tag in api.get_tags()}
    models = config_store.get_models()
    removed_count = 0

    for model in models:
        if model.tag_id not in existing_tags:
            config_store.unregister_model(model.tag_id)
            # remove model files
            model_path = model.model_path
            if model_path.exists() and model_path.is_dir():
                for item in model_path.iterdir():
                    if item.is_file():
                        item.unlink()
                model_path.rmdir()
            removed_count += 1
    logger.info(f"Cleaned up {removed_count} old models")


def train_single_tag(
    tag: dict,
    contrast_tag: dict,
    config_store: ConfigStore,
    min_samples: int = 10,
    threshold: float = 0.5,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    Train a classifier for a single tag and run inference.

    Args:
        tag: The tag dictionary from the API.
        contrast_tag: The contrast tag dictionary from the API.
        config_store: Configuration store instance.
        min_samples: Minimum positive samples required.
        threshold: Prediction threshold for tagging.
        force: If True, retrain even if training data hasn't changed.

    Returns:
        True if training was successful.
    """
    logger.info(f"{tag['name']}: Evaluating training ({tag['id']})")

    ml_tag_name = f"{tag['name']}{settings.ml_tag_suffix}"

    # Get ML tag if it exists
    ml_tag = api.create_tag(ml_tag_name, parent_id=tag["id"])

    # Get or create negative examples tag under contrast parent
    negative_tag = api.create_tag(tag["name"], parent_id=contrast_tag["id"])
    if not negative_tag:
        logger.warning(
            f"{tag['name']}: Could not find/create negative tag for '{tag['name']}'"
        )
        return False

    # Get positive asset IDs
    positive_asset_ids = api.get_assets_by_tag([tag["id"]])
    logger.debug(f"{tag['name']}: Found {len(positive_asset_ids)} positive assets")
    # Exclude assets only tagged with the predicted tag (not manually tagged)
    ml_tagged_assets = set(api.get_assets_by_tag([ml_tag["id"]]))
    positive_asset_ids = list(set(positive_asset_ids) - ml_tagged_assets)

    logger.debug(
        f"{tag['name']}: Reduced to {len(positive_asset_ids)} positive assets ({len(ml_tagged_assets)} ML-tagged)"
    )

    if len(positive_asset_ids) < min_samples:
        logger.debug(f"{tag['name']}: Skipping training, not enough positive samples")
        return False

    # Get manually tagged negative examples (contrast examples for this tag)
    negative_asset_ids = api.get_assets_by_tag([negative_tag["id"]])

    # Compute hash of training data (user-tagged positives + manual negatives)
    current_hash = compute_training_data_hash(positive_asset_ids, negative_asset_ids)
    stored_hash = config_store.get_training_data_hash(tag_id=tag["id"])

    if not force and stored_hash == current_hash:
        logger.info(f"{tag['name']}: Training data unchanged, skipping training")
        return False
    logger.debug(f"{tag['name']}: Training data changed.")

    # Manage ML tag (delete and recreate to clear old predictions)
    ml_tag = api.recreate_tag(ml_tag_name, parent_id=tag["id"])
    if not ml_tag:
        logger.error(f"{tag['name']}: Failed to create ML tag '{ml_tag_name}'")
        return False

    # we have removed the ml predicted tag, which unfortunatly changes the
    # positive asset ids, so we need to recalculate it
    positive_asset_ids = api.get_assets_by_tag([tag["id"]])

    # Get negative samples (includes manual negatives + random background samples)
    unlabeled_asset_ids = get_unlabeled_samples(
        positive_asset_ids=positive_asset_ids,
        negative_asset_ids=negative_asset_ids,
    )

    # Get embeddings
    all_asset_ids = positive_asset_ids + negative_asset_ids + unlabeled_asset_ids
    embeddings_dict = get_embeddings_by_asset_ids(all_asset_ids)

    positive_embeddings = np.array(
        [embeddings_dict[aid] for aid in positive_asset_ids if aid in embeddings_dict]
    )
    negative_embeddings = np.array(
        [embeddings_dict[aid] for aid in negative_asset_ids if aid in embeddings_dict]
    )
    unlabeled_embeddings = np.array(
        [embeddings_dict[aid] for aid in unlabeled_asset_ids if aid in embeddings_dict]
    )

    logger.debug(
        f"{tag['name']}: Training with {len(positive_embeddings)} positive and "
        f"{len(negative_embeddings)} negative embeddings and "
        f"{len(unlabeled_embeddings)} unlabeled embeddings"
    )

    # Train classifier
    classifier = BinaryEmbeddingClassifier()
    accuracy = classifier.train(
        positive=positive_embeddings,
        unlabeled=unlabeled_embeddings,
        negative=negative_embeddings if len(negative_embeddings) > 0 else None,
    )
    logger.info(f"{tag['name']}: Trained classifier with accuracy {accuracy:.4f}")

    # Save model with training data hash
    model_path = settings.ml_resource_path / tag["id"]
    classifier.save(model_path)
    config_store.register_model(tag["id"], model_path, current_hash)
    logger.debug(f"{tag['name']}: Saved model to {model_path}")

    # Run inference on all untagged assets
    inference_asset_ids = get_asset_ids_without_tag(
        positive_asset_ids, exclude_asset_ids=set(negative_asset_ids)
    )

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

            logger.debug(
                f"{tag['name']}: Tagging {len(assets_to_tag)} assets with '{ml_tag_name}' "
                f"(threshold: {threshold})"
            )

            if assets_to_tag and not dry_run:
                count = api.bulk_tag_assets(assets_to_tag, [ml_tag["id"]])
                logger.info(f"{tag['name']}: Tagged {count} assets")
    return True


def run_training(
    min_samples: int = 10,
    threshold: float = 0.5,
    force: bool = False,
    dry_run: bool = False,
):
    """
    Run training for all eligible tags.

    Args:
        min_samples: Minimum positive samples required for training.
        threshold: Prediction threshold for tagging.
        force: If True, retrain all models even if training data hasn't changed.
    """
    config_store = ConfigStore()
    logger.info("Starting training process")

    cleanup_old_models(config_store)

    # Ensure contrast tag exists
    contrast_tag = api.create_tag(settings.contrast_tag)

    trained_count = 0
    for tag in api.get_tags():
        if tag["name"].endswith(settings.ml_tag_suffix):
            logger.debug(f"Skipping '{tag['name']}': is an ML tag")
            continue

        if tag["id"] == contrast_tag["id"] or tag.get("parentId") == contrast_tag["id"]:
            logger.debug(f"Skipping '{tag['name']}': is contrast tag")
            continue

        if train_single_tag(
            tag,
            contrast_tag,
            config_store,
            min_samples,
            threshold,
            force,
            dry_run=dry_run,
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
        tag_id = model_info.tag_id
        tag = api.get_tag_by_id(tag_id)
        if not tag:
            logger.warning(f"Tag ID '{tag_id}' not found, skipping")
            continue

        model_path = model_info.model_path
        ml_tag_name = f"{tag['name']}{settings.ml_tag_suffix}"

        if incremental is False:
            ml_tag = api.recreate_tag(ml_tag_name, parent_id=tag["id"])
        else:
            ml_tag = api.create_tag(ml_tag_name, parent_id=tag["id"])

        if not ml_tag:
            logger.error(f"Failed to create ML tag '{ml_tag_name}'")
            return False
        logger.info(
            f"Running inference for '{tag['name']}' using model at {model_path}"
        )

        # Load model
        try:
            classifier = BinaryEmbeddingClassifier.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model for '{tag['name']}': {e}")
            continue

        # Get assets already tagged
        tagged_assets = set(api.get_assets_by_tag([tag["id"]]))
        ml_tagged_assets = set(api.get_assets_by_tag([ml_tag["id"]]))

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
            count = api.bulk_tag_assets(assets_to_tag, [ml_tag["id"]])
            logger.info(f"Tagged {count} assets")

    config_store.last_inference = datetime.now()
    logger.info("Inference complete.")
