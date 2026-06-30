"""
Manual pricing refresh entrypoint.

Usage:
  python scripts/refresh_pricing.py           # live run, updates pricing_snapshot.json
  python scripts/refresh_pricing.py --dry-run # show what would change, no writes
"""

import argparse
import io
import json
import logging
import sys
from pathlib import Path

# Windows cp1252 fix
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from backend.pricing_refresh import refresh_pricing


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh AI model pricing data")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    args = parser.parse_args()

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Refreshing pricing snapshot...\n")
    summary = refresh_pricing(dry_run=args.dry_run, timeout=args.timeout)

    print("-" * 60)
    print(f"Snapshot date : {summary['snapshot_date']}")
    print()

    if summary["updated_api_models"]:
        print(f"API prices updated ({len(summary['updated_api_models'])}):")
        for m in summary["updated_api_models"]:
            print(
                f"  {m['key']:25s}  "
                f"${m['old']['input']:.4f}/${m['old']['output']:.4f} → "
                f"${m['new']['input']:.4f}/${m['new']['output']:.4f}/M tokens"
                f"  (via {m['resolved_as']})"
            )
    else:
        print("API prices   : no changes")

    if summary["stale_api_models"]:
        print(f"\nNo LiteLLM match (kept as-is): {', '.join(summary['stale_api_models'])}")
        print("  → add these to API_MODEL_CANDIDATES in backend/pricing_refresh.py")

    if summary["updated_gpu_providers"]:
        print(f"\nGPU prices updated ({len(summary['updated_gpu_providers'])}):")
        for g in summary["updated_gpu_providers"]:
            print(f"  {g['key']:25s}  ${g['old']:.2f} → ${g['new']:.2f}/hr")

    if summary["manual_verify_urls"]:
        print("\nGPU providers without public pricing APIs — verify manually:")
        for key, url in summary["manual_verify_urls"].items():
            print(f"  {key:25s}  {url}")

    if not args.dry_run:
        print("\npricing_snapshot.json updated.")
    print()


if __name__ == "__main__":
    main()
