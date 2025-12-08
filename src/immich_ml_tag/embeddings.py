"""Database operations for managing embeddings from Immich's PostgreSQL database."""

from datetime import datetime, timedelta
import logging

from immich_ml_tag.api import get_assets_since
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from .config import settings


def get_connection():
    """Create a connection to the Immich PostgreSQL database."""
    conn = psycopg2.connect(
        dbname=settings.immich_db,
        user=settings.immich_db_user,
        password=settings.immich_db_password,
        host=settings.immich_db_host,
        port=settings.immich_db_port,
    )
    register_vector(conn)
    return conn


def get_all_embeddings() -> dict[str, np.ndarray]:
    """
    Get all embeddings from the database.

    Returns:
        Dictionary mapping asset IDs to their embeddings.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute('SELECT "assetId", embedding FROM smart_search')
        data = cur.fetchall()
        embeddings = {str(asset_id): embedding for asset_id, embedding in data}
        cur.close()
        return embeddings
    finally:
        conn.close()


def get_embeddings_by_asset_ids(asset_ids: list[str]) -> dict[str, np.ndarray]:
    """
    Get embeddings for specific asset IDs.

    Args:
        asset_ids: List of asset IDs to fetch embeddings for.

    Returns:
        Dictionary mapping asset IDs to their embeddings.
    """
    if not asset_ids:
        return {}

    conn = get_connection()
    try:
        cur = conn.cursor()
        # Use ANY to query multiple asset IDs efficiently, cast to uuid[]
        cur.execute(
            'SELECT "assetId", embedding FROM smart_search WHERE "assetId" = ANY(%s::uuid[])',
            (asset_ids,),
        )
        data = cur.fetchall()
        embeddings = {str(asset_id): embedding for asset_id, embedding in data}
        cur.close()
        return embeddings
    finally:
        conn.close()


def get_all_asset_ids_with_embeddings() -> set[str]:
    """
    Get all asset IDs that have embeddings in the database.

    Returns:
        Set of asset IDs.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute('SELECT "assetId" FROM smart_search')
        asset_ids = {str(row[0]) for row in cur.fetchall()}
        cur.close()
        return asset_ids
    finally:
        conn.close()


def get_asset_ids_without_tag(
    tagged_asset_ids: set[str],
    exclude_asset_ids: set[str] | None = None,
) -> list[str]:
    """
    Get asset IDs that have embeddings but are NOT in the tagged set.

    Args:
        tagged_asset_ids: Set of asset IDs that have the tag.
        exclude_asset_ids: Additional asset IDs to exclude.

    Returns:
        List of asset IDs without the tag.
    """
    all_asset_ids = get_all_asset_ids_with_embeddings()

    # Filter out assets that have the tag
    assets_without_tag = all_asset_ids - tagged_asset_ids

    # Also exclude any additional asset IDs if provided
    if exclude_asset_ids:
        assets_without_tag -= exclude_asset_ids

    return list(assets_without_tag)


def get_asset_ids_created_since(
    since: datetime | None = None,
) -> set[str]:
    """
    Get asset IDs that have embeddings and were created since a given timestamp.

    Args:
        since: Only return assets created after this timestamp.
               If None, returns all assets with embeddings.

    Returns:
        Set of asset IDs.
    """

    asset_ids = get_all_asset_ids_with_embeddings()
    if since is not None:
        logging.info(f"Filtering assets created since {since.isoformat()}")
        since = since - timedelta(hours=4)
        ids_added_since = get_assets_since(since.isoformat())
        asset_ids = asset_ids.intersection(set(ids_added_since))
    return asset_ids
