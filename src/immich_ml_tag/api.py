"""Immich API client for interacting with tags and assets."""

import logging
from typing import Any

import requests

from .config import settings

logger = logging.getLogger(__name__)


def get_headers() -> dict[str, str]:
    """Get the API headers for Immich requests."""
    return {
        "x-api-key": settings.immich_api_key,
        "x-immich-app-name": "immich-ml-tags",
    }


def get_tags() -> list[dict[str, Any]]:
    """Fetch all tags from Immich."""
    response = requests.get(
        f"{settings.immich_url}/api/tags",
        headers=get_headers(),
    )
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Error fetching tags: {response.text}")


def get_assets_by_tag(tag_ids: list[str]) -> list[str]:
    """
    Get all asset IDs that have any of the specified tags.

    Args:
        tag_ids: List of tag IDs to search for.

    Returns:
        List of asset IDs.
    """

    def _get_asset_tag_page(tag_ids: list[str], page: int) -> dict[str, Any]:
        response = requests.post(
            f"{settings.immich_url}/api/search/metadata",
            headers=get_headers(),
            json={
                "tagIds": tag_ids,
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
        assets = _get_asset_tag_page(tag_ids, page)
        if not assets or "items" not in assets:
            break
        elif len(assets["items"]) == 0:
            break
        else:
            asset_ids.extend([asset["id"] for asset in assets["items"]])
            page += 1
    return asset_ids


def create_tag(name: str, parent_id: str | None = None) -> str:
    """
    Create a new tag in Immich.

    Args:
        name: The name of the tag.
        parent_id: Optional parent tag ID.

    Returns:
        The ID of the created tag.
    """
    payload = {"name": name}
    if parent_id:
        payload["parentId"] = parent_id

    response = requests.post(
        f"{settings.immich_url}/api/tags",
        headers=get_headers(),
        json=payload,
    )
    if response.status_code == 201:
        tag = response.json()
        return tag["id"]
    else:
        raise ValueError(f"Error creating tag '{name}': {response.text}")


def delete_tag(tag_id: str) -> bool:
    """
    Delete a tag from Immich.

    Args:
        tag_id: The ID of the tag to delete.

    Returns:
        True if successful, False otherwise.
    """
    response = requests.delete(
        f"{settings.immich_url}/api/tags/{tag_id}",
        headers=get_headers(),
    )
    if response.status_code == 204:
        return True
    else:
        logger.warning(f"Error deleting tag {tag_id}: {response.text}")
        return False


def bulk_tag_assets(asset_ids: list[str], tag_ids: list[str]) -> int:
    """
    Add multiple tags to multiple assets in a single request.

    Args:
        asset_ids: List of asset IDs to tag.
        tag_ids: List of tag IDs to apply.

    Returns:
        The count of assets tagged.
    """
    if not asset_ids or not tag_ids:
        return 0

    response = requests.put(
        f"{settings.immich_url}/api/tags/assets",
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


def bulk_untag_assets(asset_ids: list[str], tag_ids: list[str]) -> int:
    """
    Remove multiple tags from multiple assets in a single request.

    Args:
        asset_ids: List of asset IDs to untag.
        tag_ids: List of tag IDs to remove.

    Returns:
        The count of assets untagged.
    """
    if not asset_ids or not tag_ids:
        return 0

    response = requests.delete(
        f"{settings.immich_url}/api/tags/assets",
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
        raise ValueError(f"Error bulk untagging assets: {response.text}")


def get_tag_by_name(name: str, parent_id: str | None = None) -> dict[str, Any] | None:
    """
    Find a tag by name, optionally under a specific parent.

    Args:
        name: The tag name to search for.
        parent_id: Optional parent tag ID to filter by.

    Returns:
        The tag dict if found, None otherwise.
    """
    tags = get_tags()
    for tag in tags:
        if tag["name"] == name:
            if parent_id is None or tag.get("parentId") == parent_id:
                return tag
    return None


def ensure_tag_exists(name: str, parent_id: str | None = None) -> str:
    """
    Ensure a tag exists, creating it if necessary.

    Args:
        name: The tag name.
        parent_id: Optional parent tag ID.

    Returns:
        The tag ID.
    """
    tag = get_tag_by_name(name, parent_id)
    if tag:
        return tag["id"]
    return create_tag(name, parent_id)
