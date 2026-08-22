"""Order lookup tool with privacy-by-design sanitization.

Design decisions:
- The LLM NEVER receives the full orders.json.
- The tool strips ALL internal/sensitive fields BEFORE returning.
- Status is authoritative: cancelled/returned orders suppress stale ETA/carrier.
- Exception status triggers handoff recommendation.
- Normalization handles case, whitespace, common variations.
- Tool result includes _tool_executed flag for anti-hallucination checks.
"""

import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class OrderLookupResult:
    """Sanitized result from an order lookup."""
    found: bool
    order_id: str
    error: Optional[str] = None
    data: Optional[dict] = None
    _tool_executed: bool = True

    def to_dict(self) -> dict:
        """Convert to dict for LLM consumption. Only safe fields."""
        result = {
            "_tool_executed": self._tool_executed,
            "found": self.found,
            "order_id": self.order_id,
        }
        if self.error:
            result["error"] = self.error
        if self.data:
            result["data"] = self.data
        return result


# Customer-safe fields from the data dictionary
CUSTOMER_SAFE_FIELDS = {
    "order_id",
    "membership_tier",
    "items",           # Only name, quantity, final_sale (not sku)
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
}

# Fields that must NEVER be exposed
FORBIDDEN_FIELDS = {
    "customer",   # Contains name, email, shipping_address
    "internal",   # Contains risk_score, warehouse_note, support_tags
}

# Statuses where delivery-related fields are stale and must be suppressed
STALE_DELIVERY_STATUSES = {"cancelled", "returned"}

# Status that requires human handoff
EXCEPTION_STATUS = "exception"


def normalize_order_id(raw_input: str) -> Optional[str]:
    """Normalize an order ID from user input.

    Handles:
    - Whitespace stripping
    - Case normalization (uppercase)
    - Common format: ORD-NNNN

    Returns normalized ID or None if the input is clearly malformed.
    """
    if not raw_input or not isinstance(raw_input, str):
        return None

    cleaned = raw_input.strip().upper()

    # Accept ORD-NNNN pattern
    match = re.match(r"^(ORD-\d+)$", cleaned)
    if match:
        return match.group(1)

    # Try adding ORD- prefix if just digits
    digit_match = re.match(r"^(\d{4,})$", cleaned)
    if digit_match:
        return f"ORD-{digit_match.group(1)}"

    # Malformed
    return None


def sanitize_items(items: list[dict]) -> list[dict]:
    """Extract only customer-safe item fields."""
    safe_items = []
    for item in items:
        safe_items.append({
            "name": item.get("name", "Unknown"),
            "quantity": item.get("quantity", 0),
            "final_sale": item.get("final_sale", False),
        })
    return safe_items


def sanitize_order(raw_order: dict) -> dict:
    """Strip all internal/sensitive fields and apply status-aware rules.

    This is the PRIMARY privacy boundary. The LLM only ever sees
    what this function returns.
    """
    status = raw_order.get("status", "").lower()

    safe = {
        "order_id": raw_order.get("order_id", ""),
        "membership_tier": raw_order.get("membership_tier", ""),
        "items": sanitize_items(raw_order.get("items", [])),
        "placed_at": raw_order.get("placed_at", ""),
        "status": raw_order.get("status", ""),
        "status_updated_at": raw_order.get("status_updated_at", ""),
        "customer_safe_message": raw_order.get("customer_safe_message", ""),
    }

    # Status-aware delivery field handling
    if status in STALE_DELIVERY_STATUSES:
        # Suppress stale delivery fields for cancelled/returned orders
        safe["shipped_at"] = None
        safe["delivered_at"] = None
        safe["carrier"] = None
        safe["tracking_number"] = None
        safe["estimated_delivery"] = None
        safe["_delivery_note"] = (
            f"This order has been {status}. "
            "Delivery-related fields are suppressed because they may be stale."
        )
    else:
        safe["shipped_at"] = raw_order.get("shipped_at")
        safe["delivered_at"] = raw_order.get("delivered_at")
        safe["carrier"] = raw_order.get("carrier")
        safe["tracking_number"] = raw_order.get("tracking_number")
        safe["estimated_delivery"] = raw_order.get("estimated_delivery")

    # Exception status: flag for handoff
    if status == EXCEPTION_STATUS:
        safe["_requires_handoff"] = True
        safe["_handoff_reason"] = "Order has an exception that requires human support review."

    # Missing ETA handling
    if status not in STALE_DELIVERY_STATUSES and safe.get("estimated_delivery") is None:
        safe["_eta_note"] = "No estimated delivery date is currently available. Do not invent one."

    return safe


class OrderLookupTool:
    """Order lookup tool that enforces privacy at the data boundary."""

    def __init__(self, orders_path: Path):
        """Load orders from JSON file."""
        with open(orders_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.snapshot_at = data.get("snapshot_at", "")
        self._orders: dict[str, dict] = {}
        for order in data.get("orders", []):
            oid = order.get("order_id", "").upper()
            if oid:
                self._orders[oid] = order

    def lookup(self, raw_order_id: str) -> OrderLookupResult:
        """Look up an order by ID. Returns ONLY sanitized, customer-safe data.

        This is the function the LLM calls via tool/function calling.
        The raw order data NEVER reaches the LLM.
        """
        # Normalize
        order_id = normalize_order_id(raw_order_id)
        if order_id is None:
            return OrderLookupResult(
                found=False,
                order_id=raw_order_id,
                error=f"'{raw_order_id}' does not appear to be a valid order ID. "
                      f"Order IDs follow the format ORD-NNNN (e.g., ORD-1007).",
            )

        # Look up
        raw_order = self._orders.get(order_id)
        if raw_order is None:
            return OrderLookupResult(
                found=False,
                order_id=order_id,
                error=f"No order found with ID {order_id}. "
                      f"Please check the order ID and try again.",
            )

        # Sanitize — this is the privacy boundary
        safe_data = sanitize_order(raw_order)

        return OrderLookupResult(
            found=True,
            order_id=order_id,
            data=safe_data,
        )

    def get_tool_schema(self) -> dict:
        """Return the function/tool schema for LLM function calling."""
        return {
            "name": "order_lookup",
            "description": (
                "Look up the current status and details of a customer order "
                "by order ID. Returns only customer-safe information. "
                "Call this tool ONLY when the user asks about a specific order "
                "and you have an order ID. Do NOT call without an order ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up, e.g. 'ORD-1007'",
                    }
                },
                "required": ["order_id"],
            },
        }
