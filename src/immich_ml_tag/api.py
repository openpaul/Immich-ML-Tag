import logging
import time
from typing import Any, List, Dict, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from .config import settings

logger = logging.getLogger(__name__)


class ImmichAPI:
    def __init__(
        self, url: str = settings.immich_url, api_key: str = settings.immich_api_key
    ):
        self.url = url.rstrip("/")
        logger.info(f"Immich API URL set to: {self.url}")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update(
            {
                "x-api-key": self.api_key,
                "x-immich-app-name": "immich-ml-tags",
            }
        )
        retries = Retry(
            total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def _get(self, endpoint: str) -> Any:
        r = self.session.get(f"{self.url}{endpoint}")
        r.raise_for_status()
        return r.json()

    def _post(self, endpoint: str, payload: dict) -> Any:
        r = self.session.post(f"{self.url}{endpoint}", json=payload)
        r.raise_for_status()
        return r.json()

    def _put(self, endpoint: str, payload: dict) -> Any:
        r = self.session.put(f"{self.url}{endpoint}", json=payload)
        r.raise_for_status()
        return r.json()

    def _delete(self, endpoint: str, payload: Optional[dict] = None) -> Any:
        r = self.session.delete(f"{self.url}{endpoint}", json=payload)
        if r.status_code not in (200, 204):
            r.raise_for_status()
        return r.json() if r.content else None

    def get_tags(self) -> list[dict[str, Any]]:
        return self._get("/api/tags")

    def get_tag_by_name(
        self, name: str, parent_id: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        for tag in self.get_tags():
            if tag["name"] == name and (
                parent_id is None or tag.get("parentId") == parent_id
            ):
                return tag
        return None

    def get_tag_by_id(
        self, tag_id: str, parent_id: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        if parent_id is not None:
            logger.warning("parent_id parameter is ignored in get_tag_by_id")
        try:
            return self._get(f"/api/tags/{tag_id}")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def create_tag(self, name: str, parent_id: Optional[str] = None) -> dict:
        tag = self.get_tag_by_name(name, parent_id)
        if tag:
            return tag

        payload = {"name": name}
        if parent_id:
            payload["parentId"] = parent_id
        return self._post("/api/tags", payload)

    def ensure_tag_exists(self, name: str, parent_id: Optional[str] = None) -> str:
        tag = self.get_tag_by_name(name, parent_id)
        return tag["id"] if tag else self.create_tag(name, parent_id)["id"]

    def check_tag_exists(self, name: str, parent_id: Optional[str] = None) -> bool:
        tag = self.get_tag_by_name(name, parent_id)
        return tag is not None

    def recreate_tag(
        self,
        name: str,
        parent_id: Optional[str] = None,
        n: int = 10,
        sleep_seconds: int = 5,
    ) -> dict | None:
        tag = self.get_tag_by_name(name, parent_id)
        if tag:
            self.delete_tag(tag["id"])
        for _ in range(n):
            if api.get_tag_by_name(name, parent_id) is None:
                return api.create_tag(name, parent_id)
            time.sleep(sleep_seconds)
        return None

    def delete_tag(self, tag_id: str) -> bool:
        try:
            self._delete(f"/api/tags/{tag_id}")
            return True
        except requests.HTTPError as e:
            logger.warning(f"Error deleting tag {tag_id}: {e}")
            return False

    def bulk_tag_assets(self, asset_ids: list[str], tag_ids: list[str]) -> int:
        if not asset_ids or not tag_ids:
            return 0
        return self._put(
            "/api/tags/assets", {"assetIds": asset_ids, "tagIds": tag_ids}
        ).get("count", 0)

    def bulk_untag_assets(self, asset_ids: list[str], tag_ids: list[str]) -> int:
        if not asset_ids or not tag_ids:
            return 0
        return self._delete(
            "/api/tags/assets", {"assetIds": asset_ids, "tagIds": tag_ids}
        ).get("count", 0)

    def get_assets_by_tag(self, tag_ids: list[str]) -> list[str]:
        asset_ids = []
        page = 1
        while True:
            result = self._post(
                "/api/search/metadata", {"tagIds": tag_ids, "page": page}
            )
            items = result.get("assets", {}).get("items", [])
            if not items:
                break
            asset_ids.extend([a["id"] for a in items])
            page += 1
        return asset_ids

    def get_assets_since(self, timestamp: str) -> list[str]:
        asset_ids = []
        page = 1
        while True:
            result = self._post(
                "/api/search/metadata", {"createdAfter": timestamp, "page": page}
            )
            items = result.get("assets", {}).get("items", [])
            if not items:
                break
            asset_ids.extend([a["id"] for a in items])
            page += 1
        return asset_ids


api = ImmichAPI()
