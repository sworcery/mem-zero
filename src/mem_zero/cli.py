from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any

import httpx


def _client(args: argparse.Namespace) -> httpx.Client:
    headers: dict[str, str] = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    return httpx.Client(base_url=args.url, headers=headers, timeout=30.0)


def _format_time(ts: float | None) -> str:
    if not ts or ts == 0:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _print_json(data: Any) -> None:
    json.dump(data, sys.stdout, indent=2)
    sys.stdout.write("\n")


def cmd_projects(args: argparse.Namespace) -> int:
    with _client(args) as client:
        resp = client.get("/api/v1/projects")
        resp.raise_for_status()
        projects = resp.json()

    if args.json:
        _print_json(projects)
        return 0

    if not projects:
        print("No projects found.")
        return 0

    max_slug = max(len(p["slug"]) for p in projects)
    print(f"{'PROJECT':<{max_slug}}  {'MEMORIES':>8}  LAST UPDATED")
    print(f"{'-' * max_slug}  {'-' * 8}  {'-' * 20}")
    for p in sorted(projects, key=lambda x: x["slug"]):
        slug = p["slug"]
        count = p["memory_count"]
        updated = _format_time(p.get("last_updated"))
        print(f"{slug:<{max_slug}}  {count:>8}  {updated}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    with _client(args) as client:
        resp = client.get(
            f"/api/v1/projects/{args.project}/memories",
            params={"limit": args.limit},
        )
        resp.raise_for_status()
        memories = resp.json()

    if args.json:
        _print_json(memories)
        return 0

    if not memories:
        print(f"No memories in project '{args.project}'.")
        return 0

    for mem in memories:
        ts = _format_time(mem.get("created_at"))
        print(f"[{mem['id'][:8]}] {ts}")
        print(f"  {mem['text']}")
        print()
    print(f"{len(memories)} memor{'y' if len(memories) == 1 else 'ies'} total.")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    with _client(args) as client:
        resp = client.post(
            f"/api/v1/projects/{args.project}/search",
            json={"query": args.query, "top_k": args.top_k},
        )
        resp.raise_for_status()
        results = resp.json()

    if args.json:
        _print_json(results)
        return 0

    if not results:
        print("No results.")
        return 0

    for r in results:
        score = f"{r.get('score', 0):.3f}" if r.get("score") is not None else "-"
        print(f"[{r['id'][:8]}] score={score}")
        print(f"  {r['text']}")
        print()
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    text = args.text
    if text == "-":
        text = sys.stdin.read().strip()
        if not text:
            print("Error: no input received from stdin.", file=sys.stderr)
            return 1

    with _client(args) as client:
        resp = client.post(
            f"/api/v1/projects/{args.project}/memories",
            json={"text": text},
            params={"user_id": args.user},
        )
        resp.raise_for_status()
        result = resp.json()

    if args.json:
        _print_json(result)
    else:
        print(f"Stored {result['stored']} memor{'y' if result['stored'] == 1 else 'ies'}.")
        for mid in result.get("ids", []):
            print(f"  {mid}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    with _client(args) as client:
        resp = client.delete(
            f"/api/v1/projects/{args.project}/memories/{args.memory_id}"
        )
        resp.raise_for_status()
    print(f"Deleted {args.memory_id}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    with _client(args) as client:
        resp = client.get(
            f"/api/v1/projects/{args.project}/memories",
            params={"limit": 1000},
        )
        resp.raise_for_status()
        memories = resp.json()

    export_data = {
        "project": args.project,
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "count": len(memories),
        "memories": memories,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"Exported {len(memories)} memories to {args.output}")
    else:
        _print_json(export_data)
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    with open(args.file) as f:
        data = json.load(f)

    memories = data.get("memories", [])
    if not memories:
        print("No memories found in file.")
        return 0

    project = args.project or data.get("project")
    if not project:
        print("Error: no project specified (use --project or provide a file with 'project' key).",
              file=sys.stderr)
        return 1

    imported = 0
    with _client(args) as client:
        for mem in memories:
            text = mem.get("text", "")
            if not text:
                continue
            user_id = mem.get("user_id", "default")
            resp = client.post(
                f"/api/v1/projects/{project}/memories",
                json={"text": text, "metadata": mem.get("metadata", {})},
                params={"user_id": user_id},
            )
            resp.raise_for_status()
            imported += 1

    print(f"Imported {imported} memories into project '{project}'.")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    with _client(args) as client:
        resp = client.get("/health")
        data = resp.json()

    exit_code = 0 if resp.status_code == 200 else 1
    if args.json:
        _print_json(data)
        return exit_code

    status = data.get("status", "unknown")
    version = data.get("version", "?")
    services = data.get("services", {})
    print(f"Status: {status} (v{version})")
    for svc, healthy in services.items():
        indicator = "ok" if healthy else "DEGRADED"
        print(f"  {svc}: {indicator}")
    return exit_code


def cmd_stats(args: argparse.Namespace) -> int:
    with _client(args) as client:
        if args.project:
            resp = client.get(f"/api/v1/projects/{args.project}/diagnostics")
        else:
            resp = client.get("/api/v1/diagnostics")
        if resp.status_code == 404:
            print("Diagnostics are disabled on this server.", file=sys.stderr)
            return 1
        resp.raise_for_status()
        data = resp.json()

    if args.json:
        _print_json(data)
        return 0

    if args.project:
        print(f"Project: {args.project}")
        for k, v in data.get("counters", {}).items():
            print(f"  {k}: {v}")
        return 0

    usage = data.get("usage", {})
    print("Operations:")
    print(f"  Total:    {usage.get('total_operations', 0)}")
    print(f"  Per day:  {usage.get('operations_per_day', 0)}")
    print(f"  Adds:     {usage.get('add_operations', 0)}")
    print(f"  Searches: {usage.get('search_operations', 0)}")
    print(f"  Facts:    {usage.get('total_facts_stored', 0)}")

    perf = data.get("performance", {})
    if perf:
        print("\nLatency (ms):")
        for key, metrics in perf.items():
            p50 = metrics.get("p50")
            if p50 is not None:
                print(f"  {key}: p50={p50} p95={metrics.get('p95')} p99={metrics.get('p99')}")

    reliability = data.get("reliability", {})
    errs = reliability.get("total_errors", 0)
    rate = reliability.get("error_rate", "N/A")
    print(f"\nErrors: {errs} ({rate})")
    print(f"Fallback activations: {reliability.get('backend_fallback_activations', 0)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mem-zero",
        description="CLI client for mem-zero memory server",
    )
    parser.add_argument(
        "--url", default="http://localhost:8765",
        help="Server URL (default: http://localhost:8765)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key for authentication",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON",
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("projects", help="List all projects")
    sub.add_parser("health", help="Check server health")

    p_list = sub.add_parser("list", help="List memories in a project")
    p_list.add_argument("project", help="Project slug")
    p_list.add_argument("--limit", type=int, default=50, help="Max memories to return")

    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("project", help="Project slug")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=10, help="Number of results")

    p_add = sub.add_parser("add", help="Add a memory")
    p_add.add_argument("project", help="Project slug")
    p_add.add_argument("text", help="Text to store (use '-' for stdin)")
    p_add.add_argument("--user", default="default", help="User ID")

    p_delete = sub.add_parser("delete", help="Delete a specific memory")
    p_delete.add_argument("project", help="Project slug")
    p_delete.add_argument("memory_id", help="Memory UUID to delete")

    p_export = sub.add_parser("export", help="Export project memories to JSON")
    p_export.add_argument("project", help="Project slug")
    p_export.add_argument("-o", "--output", help="Output file (default: stdout)")

    p_import = sub.add_parser("import", help="Import memories from JSON file")
    p_import.add_argument("file", help="JSON file to import")
    p_import.add_argument("--project", help="Target project (overrides file metadata)")

    p_stats = sub.add_parser("stats", help="View diagnostics")
    p_stats.add_argument("--project", help="Show stats for a specific project")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "projects": cmd_projects,
        "list": cmd_list,
        "search": cmd_search,
        "add": cmd_add,
        "delete": cmd_delete,
        "export": cmd_export,
        "import": cmd_import,
        "health": cmd_health,
        "stats": cmd_stats,
    }

    handler = commands.get(args.command)
    if not handler:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except httpx.HTTPStatusError as exc:
        print(f"Error: {exc.response.status_code} - {exc.response.text}", file=sys.stderr)
        return 1
    except httpx.ConnectError:
        print(f"Error: cannot connect to {args.url}", file=sys.stderr)
        return 1
    except httpx.TimeoutException as exc:
        print(f"Error: request timed out ({exc})", file=sys.stderr)
        return 1
    except httpx.RequestError as exc:
        print(f"Error: request failed - {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid response from server ({exc})", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
