"""
GPU Pricing MCP Server
======================
Exposes real-time GPU pricing as MCP tools (stdio transport).

Tools:
  get_aws_trainium2_price  — snapshot (trn2.48xlarge is Capacity-Block-only; no on-demand API exists)
  get_gcp_tpu_v5e_price   — GCP Cloud Billing Catalog API (requires GOOGLE_CLOUD_API_KEY)
  get_snapshot_gpu_prices  — CoreWeave H100 + Lambda Labs H100 from pricing_snapshot.json

AWS Trainium2: trn2.48xlarge uses EC2 Capacity Block Reservations only — the standard Price List
  API returns $0.00 for this instance type (no on-demand SKU exists). Rate in snapshot is derived
  from the ~$29.60/hr effective CB rate ÷ 16 chips = $1.85/chip.
GCP: Requires GOOGLE_CLOUD_API_KEY env var. Falls back to snapshot ($1.20/chip) if absent.

Run standalone: python mcp_servers/gpu_pricing_server.py
ADK connects via MCPToolset(StdioServerParameters(command="python", args=[...]))
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "pricing_snapshot.json"

# trn2.48xlarge has 16 Trainium2 NeuronCores (chips). Hourly instance price ÷ 16 = per-chip price.
CHIPS_PER_TRN2_48XLARGE = 16

# AWS Pricing API uses display location names, not region codes.
AWS_REGION_TO_LOCATION: dict[str, str] = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
    "eu-west-1": "Europe (Ireland)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
}

mcp = FastMCP("GPU Pricing MCP")


def _load_snapshot() -> dict:
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _http_get_json(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "gpu-pricing-mcp/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


@mcp.tool()
def get_aws_trainium2_price(region: str = "us-east-2") -> dict:
    """
    Get the effective hourly price per Trainium2 chip (USD) for AWS.

    trn2.48xlarge is ONLY available via EC2 Capacity Block Reservations — there is no
    on-demand SKU. The AWS Price List Query API returns $0 for this instance type.
    This tool returns the snapshot-derived rate ($1.85/chip, from ~$29.60/hr effective
    CB rate ÷ 16 chips per trn2.48xlarge). Verify current CB rates at
    https://aws.amazon.com/ec2/capacityblocks/pricing/
    """
    snapshot = _load_snapshot()
    fallback = snapshot.get("gpu_providers", {}).get("aws_trainium2", {})
    return {
        "hourly_cost_per_gpu": fallback.get("hourly_cost_per_gpu", 1.85),
        "instance_type": "trn2.48xlarge",
        "chips_per_instance": CHIPS_PER_TRN2_48XLARGE,
        "source": "snapshot_capacity_block_estimate",
        "note": (
            "trn2.48xlarge has no on-demand pricing — Capacity Block Reservations only. "
            "Rate is an estimate based on effective CB pricing. "
            "Verify at aws.amazon.com/ec2/capacityblocks/pricing/"
        ),
        "region": region,
    }


@mcp.tool()
def get_gcp_tpu_v5e_price(region: str = "us-central1") -> dict:
    """
    Get current on-demand hourly price per TPU v5e chip (USD) from the GCP Cloud Billing Catalog.
    Requires GOOGLE_CLOUD_API_KEY environment variable.
    Falls back to pricing_snapshot.json if the env var is absent or the call fails.
    """
    api_key = os.getenv("GOOGLE_CLOUD_API_KEY", "")

    if not api_key:
        snapshot = _load_snapshot()
        fallback = snapshot.get("gpu_providers", {}).get("gcp_tpu_v5", {})
        return {
            "hourly_cost_per_gpu": fallback.get("hourly_cost_per_gpu", 1.20),
            "source": "snapshot_fallback",
            "note": "Set GOOGLE_CLOUD_API_KEY env var to enable live GCP pricing.",
        }

    try:
        # Step 1: find the Cloud TPU service name
        services_data = _http_get_json(
            f"https://cloudbilling.googleapis.com/v1/services?key={api_key}"
        )
        tpu_service_name = None
        for svc in services_data.get("services", []):
            if svc.get("displayName", "") == "Cloud TPU":
                tpu_service_name = svc["name"]
                break

        if not tpu_service_name:
            raise ValueError("Cloud TPU service not found in GCP billing catalog services list")

        # Step 2: scan SKUs for TPU v5e on-demand price in the requested region
        page_token: str = ""
        tpu_v5e_price: float | None = None

        while tpu_v5e_price is None:
            sku_url = (
                f"https://cloudbilling.googleapis.com/v1/{tpu_service_name}/skus"
                f"?key={api_key}&pageSize=500"
                + (f"&pageToken={page_token}" if page_token else "")
            )
            skus_data = _http_get_json(sku_url)

            for sku in skus_data.get("skus", []):
                desc = sku.get("description", "").lower()
                service_regions = sku.get("serviceRegions", [])
                if (
                    "v5e" in desc
                    and "tpu" in desc
                    and "preemptible" not in desc
                    and "spot" not in desc
                    and (not service_regions or region in service_regions)
                ):
                    for pi in sku.get("pricingInfo", []):
                        expr = pi.get("pricingExpression", {})
                        for rate in expr.get("tieredRates", []):
                            unit_price = rate.get("unitPrice", {})
                            units = float(unit_price.get("units", "0") or 0)
                            nanos = float(unit_price.get("nanos", 0)) / 1_000_000_000
                            price = units + nanos
                            if price > 0:
                                tpu_v5e_price = price
                                break
                    if tpu_v5e_price is not None:
                        break

            next_token = skus_data.get("nextPageToken")
            if not next_token:
                break
            page_token = next_token

        if tpu_v5e_price is None:
            raise ValueError(f"TPU v5e on-demand SKU not found in GCP catalog for region {region!r}")

        return {
            "hourly_cost_per_gpu": round(tpu_v5e_price, 4),
            "source": "gcp_billing_catalog",
            "region": region,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as exc:
        logger.warning("GCP Billing Catalog API failed (%s) — using snapshot fallback", exc)
        snapshot = _load_snapshot()
        fallback = snapshot.get("gpu_providers", {}).get("gcp_tpu_v5", {})
        return {
            "hourly_cost_per_gpu": fallback.get("hourly_cost_per_gpu", 1.20),
            "source": "snapshot_fallback",
            "error": str(exc),
        }


@mcp.tool()
def get_snapshot_gpu_prices() -> dict:
    """
    Read CoreWeave H100 and Lambda Labs H100 hourly GPU prices from pricing_snapshot.json.
    These providers don't have public machine-readable pricing APIs, so the snapshot
    (refreshed bi-weekly via LiteLLM) is the authoritative source.

    Returns:
        dict with keys "coreweave_h100" and "lambda_labs_h100", each containing
        hourly_cost_per_gpu (float), display_name (str), source ("snapshot"), snapshot_date (str).
    """
    try:
        snapshot = _load_snapshot()
        providers = snapshot.get("gpu_providers", {})
        snapshot_date = snapshot.get("snapshot_date", "unknown")
        result: dict = {}
        for key in ("coreweave_h100", "lambda_labs_h100"):
            p = providers.get(key, {})
            result[key] = {
                "hourly_cost_per_gpu": p.get("hourly_cost_per_gpu"),
                "display_name": p.get("display_name", key),
                "source": "snapshot",
                "snapshot_date": snapshot_date,
            }
        return result
    except Exception as exc:
        return {
            "coreweave_h100": {
                "hourly_cost_per_gpu": 6.16,
                "source": "hardcoded_fallback",
                "error": str(exc),
            },
            "lambda_labs_h100": {
                "hourly_cost_per_gpu": 3.29,
                "source": "hardcoded_fallback",
                "error": str(exc),
            },
        }


if __name__ == "__main__":
    mcp.run()
