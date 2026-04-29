from __future__ import annotations

import argparse
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from .config import Config, validate_slug


def migrate(
    source_collection: str,
    qdrant: QdrantClient,
    config: Config,
    dry_run: bool = True,
) -> dict[str, int]:
    collections = {c.name for c in qdrant.get_collections().collections}
    if source_collection not in collections:
        print(f"Source collection '{source_collection}' not found.")
        print(f"Available: {', '.join(sorted(collections)) or '(none)'}")
        return {}

    source_info = qdrant.get_collection(source_collection)
    total = source_info.points_count or 0
    print(f"Source collection '{source_collection}' has {total} memories.")

    if total == 0:
        return {}

    vector_size = None
    if source_info.config and source_info.config.params:
        params = source_info.config.params
        if hasattr(params, "size"):
            vector_size = params.size
    vector_size = vector_size or config.embedding_dimensions

    grouped: dict[str, list[PointStruct]] = {}
    offset = None

    while True:
        points, offset = qdrant.scroll(
            collection_name=source_collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        for pt in points:
            app_name = (
                pt.payload.get("app_name")
                or pt.payload.get("app")
                or pt.payload.get("project")
                or "unknown"
            )
            try:
                slug = validate_slug(app_name)
            except ValueError:
                slug = "unknown"

            if slug not in grouped:
                grouped[slug] = []

            grouped[slug].append(
                PointStruct(
                    id=str(pt.id),
                    vector=pt.vector,
                    payload=pt.payload,
                )
            )

        if offset is None:
            break

    print(f"\nFound {len(grouped)} projects:")
    counts: dict[str, int] = {}
    for slug, pts in sorted(grouped.items()):
        counts[slug] = len(pts)
        print(f"  {slug}: {len(pts)} memories")

    if dry_run:
        print("\n[DRY RUN] No changes made. Run without --dry-run to migrate.")
        return counts

    for slug, pts in grouped.items():
        target = config.collection_name(slug)
        if target not in collections:
            qdrant.create_collection(
                collection_name=target,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            print(f"  Created collection: {target}")
        qdrant.upsert(collection_name=target, points=pts)
        print(f"  Migrated {len(pts)} memories to {target}")

    print(f"\nMigration complete. {sum(counts.values())} memories across {len(counts)} projects.")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate memories from upstream mem0-aio")
    parser.add_argument(
        "--source-collection",
        default="mem0",
        help="Name of the source Qdrant collection (default: mem0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview migration without making changes",
    )
    args = parser.parse_args()

    config = Config.from_env()
    if config.qdrant_url:
        qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    else:
        qdrant = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)

    try:
        migrate(args.source_collection, qdrant, config, dry_run=args.dry_run)
    finally:
        qdrant.close()


if __name__ == "__main__":
    sys.exit(main() or 0)
