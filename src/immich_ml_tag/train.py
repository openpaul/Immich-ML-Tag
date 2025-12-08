import os
import random
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import psycopg2
import requests
from pgvector.psycopg2 import register_vector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize


IMMICH_DB = os.environ.get("IMMICH_DB_DATABASE_NAME", "immich")
IMMICH_DB_USER = os.environ.get("IMMICH_DB_USERNAME", "postgres")
IMMICH_DB_PASSWORD = os.environ.get("IMMICH_DB_PASSWORD", "yourpassword")
IMMICH_DB_HOST = os.environ.get("IMMICH_DB_HOST", "localhost")
IMMICH_API_KEY = os.environ.get("IMMICH_API_KEY", "your_api_key")
IMMICH_DB_PORT = os.environ.get("IMMICH_DB_PORT", "5433")
IMMICH_URL = os.environ.get("IMMICH_URL", "https://immich.int.paulsaary.de")

DATABASE_URL = f"postgresql://{IMMICH_DB_USER}:{IMMICH_DB_PASSWORD}@{IMMICH_DB_HOST}:{IMMICH_DB_PORT}/{IMMICH_DB}"
ML_RESOURCE_PATH = os.environ.get("ML_RESOURCE_PATH", "./ml_resources")
ML_RESOURCE_PATH = Path(ML_RESOURCE_PATH)
ML_RESOURCE_PATH.mkdir(parents=True, exist_ok=True)

ML_TAG_SUFFIX = "_predicted"
CONTRAST_TAG = "ML Negative Examples"


@dataclass
class BinaryEmbeddingClassifier:
    model: LogisticRegression | None = None

    def train(
        self,
        positive: np.ndarray,
        negative: np.ndarray,
        normalize_embeddings: bool = True,
        C: float = 1.0,
    ) -> float:
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
        if self.model is None:
            raise RuntimeError("Model not trained")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")

    @classmethod
    def load(cls, path: str | Path) -> "BinaryEmbeddingClassifier":
        path = Path(path)
        model = joblib.load(path / "model.joblib")
        return cls(model=model)

    def predict_proba(
        self,
        embeddings: np.ndarray,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
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
        probs = self.predict_proba(embeddings)
        return (probs >= threshold).astype(np.int64)


@dataclass
class Model:
    tag: str
    model_path: Path
    last_trained: datetime | None = None


class Config:
    def __init__(self, folder: Path = ML_RESOURCE_PATH):
        self.folder = folder
        self._init_storage()

    def _init_storage(self):
        self.db_path = self.folder / "config.db"
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_inference DATETIME,
                last_trained DATETIME
            )
        """)
        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS models (tag TEXT PRIMARY KEY, model_path TEXT)"
        )
        self.conn.commit()

    def _delete_all_models(self):
        self.cur.execute("DELETE FROM models")
        self.conn.commit()

    def get_models(self) -> list[Model]:
        self.cur.execute("SELECT tag, model_path FROM models")
        data = self.cur.fetchall()
        models = []
        for tag, model_path in data:
            models.append(Model(tag=tag, model_path=Path(model_path)))
        return models

    # getter and setter for last trained and last inference
    @property
    def last_trained(self) -> datetime | None:
        self.cur.execute("SELECT last_trained FROM config")
        data = self.cur.fetchone()
        if data is None or data[0] is None:
            return None
        return datetime.fromisoformat(data[0])

    @last_trained.setter
    def last_trained(self, value: datetime):
        self.cur.execute(
            "INSERT OR REPLACE INTO config (id, last_trained) VALUES (1, ?)",
            (value.isoformat(),),
        )
        self.conn.commit()

    @property
    def last_inference(self) -> datetime | None:
        self.cur.execute("SELECT last_inference FROM config")
        data = self.cur.fetchone()
        if data is None or data[0] is None:
            return None
        return datetime.fromisoformat(data[0])

    @last_inference.setter
    def last_inference(self, value: datetime):
        self.cur.execute(
            "INSERT OR REPLACE INTO config (id, last_inference) VALUES (1, ?)",
            (value.isoformat(),),
        )
        self.conn.commit()

    @property
    def tags(self):
        # fetch tags from API
        response = requests.get(f"{IMMICH_URL}/api/tags", headers=get_headers())
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(f"Error fetching asset: {response.text}")


def get_assets_by_tag(tagIds: list[str]) -> list[str]:
    def _get_asset_tag_page(tagIds: list[str], page: int) -> list[str]:
        response = requests.post(
            f"{IMMICH_URL}/api/search/metadata",
            headers=get_headers(),
            json={
                "tagIds": tagIds,
                "page": page,
            },
        )
        if response.status_code == 200:
            assets = response.json()
            return assets["assets"]
        else:
            raise ValueError(f"Error fetching assets by tag: {response.text}")

    asset_ids = []
    page = 1
    while True:
        assets = _get_asset_tag_page(tagIds, page)
        if not assets or "items" not in assets:
            break
        elif len(assets["items"]) == 0:
            break
        else:
            asset_ids.extend([asset["id"] for asset in assets["items"]])
            page += 1
    return asset_ids


def get_embeddings() -> dict[str, np.ndarray]:
    """Get all embeddings from the database."""
    conn = psycopg2.connect(
        dbname=IMMICH_DB,
        user=IMMICH_DB_USER,
        password=IMMICH_DB_PASSWORD,
        host=IMMICH_DB_HOST,
        port=IMMICH_DB_PORT,
    )
    register_vector(conn)

    cur = conn.cursor()
    cur.execute('SELECT "assetId", embedding FROM smart_search')
    data = cur.fetchall()

    embeddings = {str(asset_id): embedding for asset_id, embedding in data}
    cur.close()
    conn.close()
    return embeddings


def get_embeddings_by_asset_ids(asset_ids: list[str]) -> dict[str, np.ndarray]:
    """Get embeddings for specific asset IDs."""
    if not asset_ids:
        return {}

    conn = psycopg2.connect(
        dbname=IMMICH_DB,
        user=IMMICH_DB_USER,
        password=IMMICH_DB_PASSWORD,
        host=IMMICH_DB_HOST,
        port=IMMICH_DB_PORT,
    )
    register_vector(conn)

    cur = conn.cursor()
    # Use ANY to query multiple asset IDs efficiently, cast to uuid[]
    cur.execute(
        'SELECT "assetId", embedding FROM smart_search WHERE "assetId" = ANY(%s::uuid[])',
        (asset_ids,),
    )
    data = cur.fetchall()

    embeddings = {str(asset_id): embedding for asset_id, embedding in data}
    cur.close()
    conn.close()
    return embeddings


def get_random_asset_ids_without_tag(
    tag_id: str, exclude_asset_ids: list[str] | None = None
) -> list[str]:
    """
    Get a random sample of asset IDs that do NOT have the specified tag.
    Uses the database to get all asset IDs with embeddings, then filters out those with the tag.
    """
    # Get all assets that have the tag
    assets_with_tag = set(get_assets_by_tag([tag_id]))

    # Get all asset IDs that have embeddings
    conn = psycopg2.connect(
        dbname=IMMICH_DB,
        user=IMMICH_DB_USER,
        password=IMMICH_DB_PASSWORD,
        host=IMMICH_DB_HOST,
        port=IMMICH_DB_PORT,
    )

    cur = conn.cursor()
    cur.execute('SELECT "assetId" FROM smart_search')
    all_asset_ids = {row[0] for row in cur.fetchall()}
    cur.close()
    conn.close()

    # Filter out assets that have the tag
    assets_without_tag = all_asset_ids - assets_with_tag

    # Also exclude any additional asset IDs if provided
    if exclude_asset_ids:
        assets_without_tag -= set(exclude_asset_ids)

    # Random sample
    return list(assets_without_tag)


def get_headers():
    return {
        "x-api-key": IMMICH_API_KEY,
        "x-immich-app-name": "immich-ml-tags",
    }


def create_ml_tag(ml_tag_name: str, parent_id: str) -> str:
    # use put tags
    response = requests.post(
        f"{IMMICH_URL}/api/tags",
        headers=get_headers(),
        json={"name": ml_tag_name, "parentId": parent_id},
    )
    if response.status_code == 201:
        tag = response.json()
        return tag["id"]
    else:
        raise ValueError(f"Error creating ML tag {ml_tag_name}: {response.text}")


def delete_ml_tag(ml_tag_id: str) -> None:
    # use delete tags
    response = requests.delete(
        f"{IMMICH_URL}/api/tags/{ml_tag_id}",
        headers=get_headers(),
    )
    if response.status_code == 204:
        return
    else:
        logger.warning(f"Error deleting ML tag {ml_tag_id}: {response.text}")


def bulk_tag_assets(asset_ids: list[str], tag_ids: list[str]) -> int:
    """
    Add multiple tags to multiple assets in a single request.
    Returns the count of assets tagged.
    """
    if not asset_ids or not tag_ids:
        return 0

    response = requests.put(
        f"{IMMICH_URL}/api/tags/assets",
        headers=get_headers(),
        json={
            "assetIds": asset_ids,
            "tagIds": tag_ids,
        },
    )
    if response.status_code == 200:
        result = response.json()
        return result.get("count", 0)
    else:
        raise ValueError(f"Error bulk tagging assets: {response.text}")


if __name__ == "__main__":
    asset_id = "1c317da1-07e1-4491-8f35-e7408e4d44c5"

    config = Config()
    # get all tags
    tags = config.tags
    tag_names = [tag["name"] for tag in tags]

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    try:
        create_ml_tag(CONTRAST_TAG, None)
    except Exception:
        pass
    contrast_tag = next(t for t in config.tags if t["name"] == CONTRAST_TAG)
    contrast_tag_id = contrast_tag["id"]

    for tag in tags:
        logger.info(f"Tag: {tag['name']}")
        if tag["name"].endswith(ML_TAG_SUFFIX):
            logger.info(f"Skipping tag {tag} as it is a predicted tag.")
            continue
        if tag["name"] == CONTRAST_TAG or tag["id"] == contrast_tag_id:
            logger.info(f"Skipping tag {tag} as it is the contrast tag.")
            continue

        ml_tag_name = f"{tag['name']}{ML_TAG_SUFFIX}"
        ml_tag_id = None
        if ml_tag_name in tag_names:
            logger.info(f"ML tag {ml_tag_name} already exists.")
            ml_tag = next(t for t in tags if t["name"] == ml_tag_name)
            ml_tag_id = ml_tag["id"]

        ml_negative_tag_name = tag["name"]
        ml_negative_tag_parent = contrast_tag_id
        ml_negative_tag_id = None
        try:
            create_ml_tag(ml_negative_tag_name, ml_negative_tag_parent)
        except Exception:
            logger.info(
                f"ML negative tag {ml_negative_tag_name} already exists or you need to create it in the UI."
            )
            pass
        ml_negative_tag_id = next(
            t
            for t in config.tags
            if t["name"] == ml_negative_tag_name
            and t.get("parentId") == contrast_tag_id
        )["id"]
        logger.info(f"Using ML negative tag id {ml_negative_tag_id}")

        tag_id = tag["id"]
        logger.info(f"Processing tag {tag['name']} with id {tag_id}")
        positive_asset_ids = get_assets_by_tag([tag_id])
        negative_asset_ids_contrast = get_assets_by_tag([ml_negative_tag_id])
        # remove assets that are only tagged with the predicted tag
        if ml_tag_id is not None:
            positive_asset_ids_ml = get_assets_by_tag([ml_tag_id])
            positive_asset_ids = list(
                set(positive_asset_ids) - set(positive_asset_ids_ml)
            )

        logger.info(f"Found {len(positive_asset_ids)} positive assets.")
        if len(positive_asset_ids) < 10:
            logger.info(f"Skipping tag {tag['name']} as it has less than 10 assets.")
            continue

        negative_asset_ids = get_random_asset_ids_without_tag(tag_id)
        all_negative_asset_ids = negative_asset_ids.copy()
        if len(negative_asset_ids) > 4 * len(positive_asset_ids):
            negative_asset_ids = random.sample(
                negative_asset_ids, 4 * len(positive_asset_ids)
            )
        negative_asset_ids += negative_asset_ids_contrast
        logger.info(f"Selected {len(negative_asset_ids)} negative assets.")

        embeddings_dict = get_embeddings_by_asset_ids(
            positive_asset_ids + negative_asset_ids
        )
        positive_embeddings = np.array(
            [
                embeddings_dict[asset_id]
                for asset_id in positive_asset_ids
                if asset_id in embeddings_dict
            ]
        )
        negative_embeddings = np.array(
            [
                embeddings_dict[asset_id]
                for asset_id in negative_asset_ids
                if asset_id in embeddings_dict
            ]
        )
        logger.info(
            f"Using {len(positive_embeddings)} positive and {len(negative_embeddings)} negative embeddings for training."
        )
        # Train model
        classifier = BinaryEmbeddingClassifier()
        accuracy = classifier.train(positive_embeddings, negative_embeddings)
        logger.info(f"Trained model for tag {tag['name']} with accuracy {accuracy}")
        # Save model
        model_path = ML_RESOURCE_PATH / f"{tag['name']}{ML_TAG_SUFFIX}"
        classifier.save(model_path)
        logger.info(f"Saved model to {model_path}")

        if ml_tag_name not in tag_names:
            create_ml_tag(ml_tag_name, tag_id)
        elif ml_tag_id is not None:
            delete_ml_tag(ml_tag_id)
            while True:
                try:
                    create_ml_tag(ml_tag_name, tag_id)
                    break
                except Exception:
                    logger.info(
                        f"Waiting to recreate ML tag {ml_tag_name} after deletion..."
                    )
                    time.sleep(10)
                    delete_ml_tag(ml_tag_id)
        else:
            raise RuntimeError("ml_tag_id is None but tag exists")

        # getml_tag_id
        tags = config.tags
        ml_tag = next(t for t in tags if t["name"] == ml_tag_name)
        ml_tag_id = ml_tag["id"]

        # get_all_ids we can predict on
        all_asset_ids = get_random_asset_ids_without_tag(tag_id, positive_asset_ids)
        logger.info(f"Predicting on {len(all_asset_ids)} assets.")
        all_embeddings = get_embeddings_by_asset_ids(all_asset_ids)
        all_embeddings_array = np.array(
            [
                all_embeddings[asset_id]
                for asset_id in all_asset_ids
                if asset_id in all_embeddings
            ]
        )
        predictions = classifier.predict_proba(all_embeddings_array)
        # assign tag to assets with prediction > 0.9
        threshold = 0.5
        assets_to_tag = [
            all_asset_ids[i] for i, prob in enumerate(predictions) if prob >= threshold
        ]
        logger.info(
            f"Tagging {len(assets_to_tag)} assets with tag {ml_tag_name} (threshold {threshold})."
        )
        count = bulk_tag_assets(assets_to_tag, [ml_tag_id])
