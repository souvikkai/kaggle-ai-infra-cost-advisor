"""
Pricing Refresh
===============
Fetches current API model pricing from LiteLLM's community-maintained pricing
database and GPU provider pricing from public sources, then atomically updates
data/pricing_snapshot.json.

Sources:
  API models  — https://github.com/BerriAI/litellm (model_prices_and_context_window.json)
                Updated by the LiteLLM community as providers change pricing.
  GPU prices  — Lambda Labs public pricing page (simple scrape)
                CoreWeave / AWS / GCP: fall back to snapshot if scraping unavailable.

Run schedule: bi-weekly via APScheduler in backend/main.py
Manual run:   python scripts/refresh_pricing.py [--dry-run]
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "pricing_snapshot.json"

LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

# Map our pricing_snapshot.json model keys → LiteLLM model IDs (in priority order)
# Multiple candidates are tried left-to-right; first match wins.
API_MODEL_CANDIDATES: dict[str, list[str]] = {
    "claude_haiku_4_5": [
        "claude-haiku-4-5-20251001",
        "claude-haiku-4-5",
        "claude-haiku-3-5",
    ],
    "claude_sonnet_4_6": [
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-3-5-sonnet-20241022",
    ],
    "gpt_4o_mini": [
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
    ],
    "gpt_5_5": [
        "gpt-5.5",
        "gpt-5",
        "gpt-4.5-preview",
        "gpt-4o",  # fallback to most capable available
    ],
    "gemini_flash_35": [
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-1.5-flash",
    ],
    "gemini_pro_31": [
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.0-pro",
        "gemini/gemini-1.5-pro",
    ],
}

# GPU provider pricing: public URLs + simple extraction patterns.
# Each entry: (url, hourly_cost_pattern_group, fallback_key_in_snapshot)
# We attempt a lightweight fetch+regex; on any failure we keep the snapshot value.
GPU_PROVIDER_SOURCES: dict[str, dict] = {
    "lambda_labs_h100": {
        "url": "https://lambdalabs.com/service/gpu-cloud",
        # Lambda Labs pricing page lists H100 SXM prices
        "pattern": r"H100\s+SXM[^$]*\$([\d.]+)\s*/\s*hr",
        "fallback_field": "hourly_cost_per_gpu",
    },
    # CoreWeave, AWS Trainium2, GCP TPU v5e don't have reliably parseable
    # public pricing pages — use snapshot values and log a reminder.
    "coreweave_h100": {"url": None, "pattern": None, "fallback_field": "hourly_cost_per_gpu"},
    "aws_trainium2": {"url": None, "pattern": None, "fallback_field": "hourly_cost_per_gpu"},
    "gcp_tpu_v5": {"url": None, "pattern": None, "fallback_field": "hourly_cost_per_gpu"},
}

GPU_MANUAL_VERIFY_URLS = {
    "coreweave_h100": "https://www.coreweave.com/gpu-cloud-computing",
    "aws_trainium2": "https://aws.amazon.com/machine-learning/trainium/pricing/",
    "gcp_tpu_v5": "https://cloud.google.com/tpu/pricing",
}


def _http_get(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pricing-refresh/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_litellm_prices(timeout: int = 20) -> dict[str, Any]:
    """Fetch LiteLLM's full pricing JSON. Returns {} on failure."""
    try:
        data = _http_get(LITELLM_PRICING_URL, timeout=timeout)
        pricing = json.loads(data)
        logger.info("LiteLLM pricing fetched — %d model entries", len(pricing))
        return pricing
    except Exception as exc:
        logger.warning("Failed to fetch LiteLLM pricing: %s", exc)
        return {}


def _resolve_api_model_price(
    model_key: str,
    litellm_data: dict[str, Any],
) -> dict[str, float] | None:
    """
    Try each candidate model ID in order. Return {input_per_million, output_per_million}
    for the first match found, or None if nothing matches.
    """
    for candidate in API_MODEL_CANDIDATES.get(model_key, []):
        entry = litellm_data.get(candidate)
        if not entry:
            continue
        input_cost = entry.get("input_cost_per_token")
        output_cost = entry.get("output_cost_per_token")
        if input_cost is None or output_cost is None:
            continue
        return {
            "input_per_million": round(float(input_cost) * 1_000_000, 4),
            "output_per_million": round(float(output_cost) * 1_000_000, 4),
            "_resolved_as": candidate,
        }
    return None


def _scrape_gpu_price(provider_key: str, source: dict) -> float | None:
    """Attempt to scrape hourly GPU cost from a public pricing page. Returns None on failure."""
    url = source.get("url")
    pattern = source.get("pattern")
    if not url or not pattern:
        return None
    try:
        html = _http_get(url, timeout=15).decode("utf-8", errors="replace")
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            price = float(match.group(1))
            logger.info("Scraped %s GPU price: $%.2f/hr", provider_key, price)
            return price
    except Exception as exc:
        logger.warning("GPU price scrape failed for %s: %s", provider_key, exc)
    return None


def refresh_pricing(
    dry_run: bool = False,
    timeout: int = 20,
) -> dict[str, Any]:
    """
    Core refresh logic. Reads current snapshot, fetches updated prices, merges,
    and (unless dry_run) atomically writes back to pricing_snapshot.json.

    Returns a summary dict with:
      updated_api_models   — list of model keys that changed
      stale_api_models     — list of model keys with no LiteLLM match (kept as-is)
      updated_gpu_providers — list of GPU provider keys that changed
      snapshot_date        — ISO date string of the new snapshot
      dry_run              — bool
    """
    # Load current snapshot
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    summary: dict[str, Any] = {
        "updated_api_models": [],
        "stale_api_models": [],
        "updated_gpu_providers": [],
        "kept_gpu_providers": [],
        "manual_verify_urls": {},
        "dry_run": dry_run,
    }

    # ── API models ──────────────────────────────────────────────────────────
    litellm_data = fetch_litellm_prices(timeout=timeout)

    for model_key, model_config in snapshot.get("models", {}).items():
        resolved = _resolve_api_model_price(model_key, litellm_data)
        if resolved is None:
            logger.warning(
                "No LiteLLM match for %s (tried: %s) — keeping snapshot value",
                model_key,
                API_MODEL_CANDIDATES.get(model_key, []),
            )
            summary["stale_api_models"].append(model_key)
            continue

        old_in = model_config.get("input_per_million")
        old_out = model_config.get("output_per_million")
        new_in = resolved["input_per_million"]
        new_out = resolved["output_per_million"]

        if old_in != new_in or old_out != new_out:
            logger.info(
                "%s price changed: $%.4f/$%.4f → $%.4f/$%.4f (via %s)",
                model_key, old_in, old_out, new_in, new_out, resolved["_resolved_as"],
            )
            summary["updated_api_models"].append({
                "key": model_key,
                "resolved_as": resolved["_resolved_as"],
                "old": {"input": old_in, "output": old_out},
                "new": {"input": new_in, "output": new_out},
            })
        else:
            logger.debug("%s unchanged (via %s)", model_key, resolved["_resolved_as"])

        # Update display_name to reflect the resolved model ID if it was a fallback
        snapshot["models"][model_key]["input_per_million"] = new_in
        snapshot["models"][model_key]["output_per_million"] = new_out
        snapshot["models"][model_key]["_resolved_as"] = resolved["_resolved_as"]

    # ── GPU providers ────────────────────────────────────────────────────────
    for provider_key, provider_config in snapshot.get("gpu_providers", {}).items():
        source = GPU_PROVIDER_SOURCES.get(provider_key, {})
        scraped = _scrape_gpu_price(provider_key, source)

        if scraped is not None:
            old = provider_config.get("hourly_cost_per_gpu")
            if scraped != old:
                logger.info(
                    "%s GPU price changed: $%.2f → $%.2f/hr", provider_key, old, scraped
                )
                summary["updated_gpu_providers"].append({
                    "key": provider_key,
                    "old": old,
                    "new": scraped,
                })
                snapshot["gpu_providers"][provider_key]["hourly_cost_per_gpu"] = scraped
            else:
                summary["kept_gpu_providers"].append(provider_key)
        else:
            # No scrape available — keep snapshot, remind operator to verify
            summary["kept_gpu_providers"].append(provider_key)
            if provider_key in GPU_MANUAL_VERIFY_URLS:
                summary["manual_verify_urls"][provider_key] = GPU_MANUAL_VERIFY_URLS[provider_key]

    # ── Metadata ─────────────────────────────────────────────────────────────
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snapshot["snapshot_date"] = now_iso
    snapshot["source_note"] = (
        f"Auto-refreshed {now_iso} via LiteLLM pricing database. "
        "GPU providers without public pricing APIs kept from previous snapshot — "
        "verify manually at: " + ", ".join(GPU_MANUAL_VERIFY_URLS.values())
    )
    summary["snapshot_date"] = now_iso

    if dry_run:
        logger.info("DRY RUN — no files written. Summary: %s", summary)
        return summary

    # Atomic write: write to temp file in same directory, then rename
    tmp = SNAPSHOT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    os.replace(tmp, SNAPSHOT_PATH)
    logger.info("pricing_snapshot.json updated → %s", now_iso)

    return summary
