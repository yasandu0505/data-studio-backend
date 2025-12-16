import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Data Studio Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

KIND_FILE = Path(__file__).parent / "kinds.json"
CACHE_REFRESH_SECONDS = 15 * 60  # 15 minutes


def load_kind_pairs(path: Path = KIND_FILE) -> list[dict[str, str]]:
    """
    Read available kind pairs from disk.
    Each entry should be an object with 'major' and 'minor' keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"Kind file not found: {path}")
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Kind file must contain a list of kind objects")
    pairs: list[dict[str, str]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict) or "major" not in item or "minor" not in item:
            raise ValueError(f"Invalid kind entry at index {idx}: {item}")
        pairs.append({"major": str(item["major"]), "minor": str(item["minor"])})
    return pairs


async def fetch_kind_data(api_url: str, pair: dict[str, str], client: httpx.AsyncClient) -> Any:
    payload = {
        "kind": {
            "major": pair["major"],
            "minor": pair["minor"]
            }
        }
    resp = await client.post(api_url.rstrip("/") + "/v1/entities/search", json=payload)
    resp.raise_for_status()
    return resp.json()


@app.on_event("startup")
async def startup_event() -> None:
    """
    On startup, load env vars, read kind pairs, and warm the cache.
    A background task refreshes the cache every 15 minutes.
    """
    load_dotenv()
    api_url = os.getenv("OPENGIN_READ_API")
    if not api_url:
        raise RuntimeError("OPENGIN_READ_API environment variable is required")

    kind_pairs = load_kind_pairs()

    app.state.cache_lock = asyncio.Lock()
    app.state.kind_cache: dict[str, Any] = {"data": [], "last_updated": None}
    app.state.refresh_task = None
    app.state.api_url = api_url  # Store API URL for use in endpoints

    async def refresh_cache_once() -> None:
        """Fetch all kinds in parallel and update the in-memory cache."""
        async def fetch_with_error_handling(
            pair: dict[str, str], client: httpx.AsyncClient
        ) -> tuple[dict[str, str], Any]:
            try:
                result = await fetch_kind_data(api_url, pair, client)
                return (pair, {"result": result})
            except Exception as exc:  # noqa: BLE001
                return (pair, {"error": str(exc)})

        async with httpx.AsyncClient() as client:
            tasks = [fetch_with_error_handling(pair, client) for pair in kind_pairs]
            results = await asyncio.gather(*tasks)

        cache_payload = {
            "data": [
                {"pair": pair, **outcome}
                for pair, outcome in results
            ],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        async with app.state.cache_lock:
            app.state.kind_cache = cache_payload
        print(f"Cache refreshed with {len(results)} entries at {cache_payload['last_updated']}")

    async def refresh_cache_periodically() -> None:
        """Refresh cache forever on an interval."""
        while True:
            await refresh_cache_once()
            await asyncio.sleep(CACHE_REFRESH_SECONDS)

    # Warm cache once, then start periodic background refresh
    await refresh_cache_once()
    app.state.refresh_task = asyncio.create_task(refresh_cache_periodically())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cancel the periodic refresh task on shutdown."""
    task = getattr(app.state, "refresh_task", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

@app.get("/counts")
async def get_counts() -> dict[str, Any]:
    """Get the counts of each kind."""
    async with app.state.cache_lock:
        cache = app.state.kind_cache
        
    def extract_count(result: Any) -> int:
        """Extract count from cached API response.

        Expected shape (example):
        {
          "body": [ ...items... ],
          ...other fields...
        }
        Fallbacks: count/total fields or list-like bodies.
        """
        if isinstance(result, dict):
            # Primary shape we expect: list under "body"
            if "body" in result and isinstance(result["body"], list):
                return len(result["body"])
            # Try common count fields
            if "count" in result:
                return int(result["count"])
            if "total" in result:
                return int(result["total"])
            # Other list-bearing fields
            if "items" in result and isinstance(result["items"], list):
                return len(result["items"])
            if "entities" in result and isinstance(result["entities"], list):
                return len(result["entities"])
            if "data" in result and isinstance(result["data"], list):
                return len(result["data"])
        elif isinstance(result, list):
            return len(result)
        return 0
    
    total_count = 0
    major_counts: dict[str, int] = {}
    minor_counts: dict[str, dict[str, int]] = {}  # Nested: major -> minor -> count
    
    for entry in cache.get("data", []):
        if "error" in entry:
            continue  # Skip entries with errors
        
        pair = entry.get("pair", {})
        major = pair.get("major", "")
        minor = pair.get("minor", "")
        result = entry.get("result", {})
        
        count = extract_count(result)
        total_count += count
        
        # Aggregate by major kind
        if major:
            major_counts[major] = major_counts.get(major, 0) + count
        
        # Aggregate by minor kind, nested under major kind
        if major and minor:
            if major not in minor_counts:
                minor_counts[major] = {}
            minor_counts[major][minor] = minor_counts[major].get(minor, 0) + count
    
    return {
        "total_count": total_count,
        "major_counts": major_counts,
        "minor_counts": minor_counts,
        "last_updated": cache.get("last_updated")
    }


@app.get("/entities")
async def get_entities(
    major: str,
    minor: str,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Return cached entities for a given major/minor kind with pagination.
    Pulls from the in-memory cache; does not call the upstream API.
    """
    if limit < 0 or offset < 0:
        return {"error": "offset and limit must be non-negative integers"}

    async with app.state.cache_lock:
        cache = app.state.kind_cache
        entries = cache.get("data", [])

    # Find the entry for the requested pair
    target = None
    for entry in entries:
        pair = entry.get("pair", {})
        if pair.get("major") == major and pair.get("minor") == minor:
            target = entry
            break

    if target is None:
        return {"error": f"No cached data for {major}/{minor}"}
    if "error" in target:
        return {"error": target.get("error", "Unknown error")}

    result = target.get("result", {})
    body = result.get("body") if isinstance(result, dict) else None
    if not isinstance(body, list):
        return {"error": "Cached entry does not contain a list under 'body'"}

    sliced = body[offset: offset + limit]
    return {
        "pair": {"major": major, "minor": minor},
        "offset": offset,
        "limit": limit,
        "count": len(sliced),
        "total": len(body),
        "items": sliced,
        "last_updated": cache.get("last_updated"),
    }


@app.get("/entities/{entity_id}/relations")
async def get_entity_relations(entity_id: str) -> Any:
    """
    Get relations for a specific entity by ID, then fetch entity details for each relation in parallel.
    Makes fresh API calls (does not use cache).
    """
    api_url = getattr(app.state, "api_url", None)
    if not api_url:
        return {"error": "API URL not configured"}

    async def fetch_entity_by_id(client: httpx.AsyncClient, related_entity_id: str) -> dict[str, Any]:
        """Fetch entity details by ID."""
        try:
            search_url = f"{api_url.rstrip('/')}/v1/entities/search"
            search_payload = {"id": related_entity_id}
            search_resp = await client.post(search_url, json=search_payload, timeout=30)
            search_resp.raise_for_status()
            return search_resp.json()
        except Exception:  # noqa: BLE001
            return {}

    async with httpx.AsyncClient() as client:
        try:
            # First, get the relations
            relations_url = f"{api_url.rstrip('/')}/v1/entities/{entity_id}/relations"
            relations_payload = {
                "id": "",
                "relatedEntityId": "",
                "name": "",
                "activeAt": "",
                "startTime": "",
                "endTime": "",
                "direction": ""
            }
            relations_resp = await client.post(relations_url, json=relations_payload, timeout=30)
            relations_resp.raise_for_status()
            relations_data = relations_resp.json()
            
            # Extract relatedEntityId values
            if not isinstance(relations_data, list):
                return relations_data
            
            related_ids = []
            for relation in relations_data:
                if isinstance(relation, dict) and "relatedEntityId" in relation:
                    related_ids.append(relation["relatedEntityId"])
            
            # Fetch entity details for all related entities in parallel
            if related_ids:
                tasks = [fetch_entity_by_id(client, related_id) for related_id in related_ids]
                entity_details = await asyncio.gather(*tasks)
                return entity_details
            else:
                return []
                
        except httpx.HTTPStatusError as e:
            return {"error": f"API returned {e.response.status_code}", "details": str(e)}
        except httpx.RequestError as e:
            return {"error": "Failed to connect to API", "details": str(e)}
        except Exception as e:  # noqa: BLE001
            return {"error": "Unexpected error", "details": str(e)}


@app.get("/entities/{entity_id}/metadata")
async def get_entity_metadata(entity_id: str) -> dict[str, Any]:
    """
    Get metadata for a specific entity by ID.
    Makes a fresh API call (does not use cache).
    """
    api_url = getattr(app.state, "api_url", None)
    if not api_url:
        return {"error": "API URL not configured"}

    metadata_url = f"{api_url.rstrip('/')}/v1/entities/{entity_id}/metadata"
    
    async with httpx.AsyncClient() as client:
        try:
            metadata_resp = await client.get(metadata_url, timeout=30)
            metadata_resp.raise_for_status()
            metadata = metadata_resp.json()
            return metadata if metadata else {}
        except httpx.HTTPStatusError as e:
            return {"error": f"API returned {e.response.status_code}", "details": str(e)}
        except httpx.RequestError as e:
            return {"error": "Failed to connect to API", "details": str(e)}
        except Exception as e:  # noqa: BLE001
            return {"error": "Unexpected error", "details": str(e)}


@app.get("/entities/{entity_id}/categories-tree-datasets")
async def get_entity_categories_tree(entity_id: str) -> Any:
    """
    Get categories tree for a specific entity by recursively fetching categories and datasets.
    """
    api_url = getattr(app.state, "api_url", None)
    if not api_url:
        return {"error": "API URL not configured"}

    # Snapshot cache to resolve names
    async with app.state.cache_lock:
        cache_snapshot = app.state.kind_cache.get("data", [])

    def build_name_lookup() -> tuple[dict[str, str], dict[str, str]]:
        """Build lookup dictionaries for Category and Dataset names keyed by entity id."""
        category_names: dict[str, str] = {}
        dataset_names: dict[str, str] = {}
        for entry in cache_snapshot:
            if "error" in entry:
                continue
            pair = entry.get("pair", {})
            major = pair.get("major")
            result = entry.get("result", {})
            body = result.get("body") if isinstance(result, dict) else None
            if not isinstance(body, list):
                continue
            for item in body:
                if not isinstance(item, dict):
                    continue
                ent_id = item.get("id")
                name = item.get("name")
                if not ent_id or name is None:
                    continue
                if major == "Category":
                    category_names[ent_id] = name
                if major == "Dataset":
                    dataset_names[ent_id] = name
        return category_names, dataset_names

    category_names, dataset_names = build_name_lookup()

    def decode_protobuf_name(encoded_name: str) -> str:
        """Decode protobuf encoded name string.
        
        Format: {"typeUrl":"type.googleapis.com/google.protobuf.StringValue","value":"35656435623032666533"}
        The value is hex-encoded bytes that need to be decoded to UTF-8.
        """
        try:
            if not encoded_name or not isinstance(encoded_name, str):
                return encoded_name
            # Parse JSON string
            parsed = json.loads(encoded_name)
            if isinstance(parsed, dict) and "value" in parsed:
                hex_value = parsed["value"]
                # Decode hex to bytes, then to UTF-8 string
                decoded_bytes = bytes.fromhex(hex_value)
                return decoded_bytes.decode("utf-8")
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):  # noqa: BLE001
            # If decoding fails, return original
            pass
        return encoded_name

    async def fetch_metadata(client: httpx.AsyncClient, target_id: str) -> dict[str, Any]:
        """Fetch metadata for an entity id."""
        try:
            metadata_url = f"{api_url.rstrip('/')}/v1/entities/{target_id}/metadata"
            resp = await client.get(metadata_url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    async def fetch_relations(
        client: httpx.AsyncClient, 
        entity_id: str, 
        relation_name: str
    ) -> list[dict[str, Any]]:
        """Fetch relations for an entity with a specific relation name."""
        try:
            relations_url = f"{api_url.rstrip('/')}/v1/entities/{entity_id}/relations"
            relations_payload = {
                "id": "",
                "relatedEntityId": "",
                "name": relation_name,
                "activeAt": "",
                "startTime": "",
                "endTime": "",
                "direction": "OUTGOING"
            }
            relations_resp = await client.post(relations_url, json=relations_payload, timeout=30)
            relations_resp.raise_for_status()
            relations_data = relations_resp.json()
            return relations_data if isinstance(relations_data, list) else []
        except Exception:  # noqa: BLE001
            return []

    async def build_category_tree(
        client: httpx.AsyncClient, 
        entity_id: str
    ) -> list[dict[str, Any]]:
        """Recursively build the categories tree for an entity."""
        # Get AS_CATEGORY relations
        categories = await fetch_relations(client, entity_id, "AS_CATEGORY")
        # Replace relation names with actual category names when available, drop id
        for cat in categories:
            if isinstance(cat, dict):
                cat.pop("id", None)  # Drop id field
                rid = cat.get("relatedEntityId")
                if rid and rid in category_names:
                    cat["name"] = category_names[rid]
        
        if not categories:
            # No categories: fetch attributes and metadata for this entity in parallel
            attributes, metadata = await asyncio.gather(
                fetch_relations(client, entity_id, "IS_ATTRIBUTE"),
                fetch_metadata(client, entity_id),
            )
            for attr in attributes:
                if isinstance(attr, dict):
                    attr.pop("id", None)  # Drop id field
                    rid = attr.get("relatedEntityId")
                    print(f"Processing attribute, rid: {rid}, in dataset_names: {rid in dataset_names if rid else False}")
                    if rid and rid in dataset_names:
                        encoded_name = dataset_names[rid]
                        decoded_name = decode_protobuf_name(encoded_name)
                        # Match with metadata: if key exists, use its value and pop it
                        if decoded_name in metadata:
                            attr["name"] = metadata.pop(decoded_name)
                        else:
                            attr["name"] = None
            return {"children": attributes if attributes else []}
        
        # If categories exist, recursively fetch children in parallel
        async def process_category(category: dict[str, Any]) -> dict[str, Any]:
            related_id = category.get("relatedEntityId", "")
            if related_id:
                children_result = await build_category_tree(client, related_id)
                # If result is a dict with "children" (attributes case), extract the children
                if isinstance(children_result, dict) and "children" in children_result:
                    children = children_result["children"]
                else:
                    children = children_result if children_result else []
                return {
                    **category,
                    "children": children
                }
            # No related id; still return category
            return category
        
        # Process all categories in parallel
        tasks = [process_category(cat) for cat in categories]
        processed_categories = await asyncio.gather(*tasks)
        
        return processed_categories

    async with httpx.AsyncClient() as client:
        try:
            result = await build_category_tree(client, entity_id)
            return result
        except Exception as e:  # noqa: BLE001
            return {"error": "Unexpected error", "details": str(e)}


@app.post("/entities/{entity_id}/metadata-add")
async def add_entity_metadata(entity_id: str, payload: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Add entity metadata by sending a PUT request to the write API.
    Accepts a payload like [{"key": "value"}, {"key": "value"}] and uses it directly as metadata.
    """
    write_api_url = os.getenv("OPENGIN_WRITE_API")
    if not write_api_url:
        return {"error": "OPENGIN_WRITE_API environment variable is required"}

    # Check for duplicate keys across all objects in the payload
    all_keys = []
    for item in payload:
        if isinstance(item, dict):
            all_keys.extend(item.keys())
    
    # Check if any key appears more than once
    if len(all_keys) != len(set(all_keys)):
        return {"error": "can't have same key"}

    # Use payload directly as metadata
    metadata = payload

    # Prepare the request payload
    request_payload = {
        "id": entity_id,
        "metadata": metadata
    }
    print(f"Request payload: {request_payload}")
    
    url = f"{write_api_url.rstrip('/')}/entities/{entity_id}"

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.put(url, json=request_payload, timeout=30)
            resp.raise_for_status()
            return resp.json() 
        except httpx.HTTPStatusError as e:
            return {"error": f"API returned {e.response.status_code}"}
        except httpx.RequestError as e:
            return {"error": "Failed to connect to API", "details": str(e)}
        except Exception as e:  # noqa: BLE001
            return {"error": "Unexpected error", "details": str(e)}
