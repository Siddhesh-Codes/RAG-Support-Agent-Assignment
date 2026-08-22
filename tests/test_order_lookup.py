"""Tests for the order lookup tool: normalization, sanitization, privacy, status handling."""

import pytest
from pathlib import Path
from src.tools.order_lookup import (
    normalize_order_id,
    sanitize_order,
    sanitize_items,
    OrderLookupTool,
    OrderLookupResult,
    FORBIDDEN_FIELDS,
)


# --- Order ID Normalization ---

class TestNormalizeOrderId:
    def test_standard_format(self):
        assert normalize_order_id("ORD-1007") == "ORD-1007"

    def test_lowercase(self):
        assert normalize_order_id("ord-1007") == "ORD-1007"

    def test_mixed_case(self):
        assert normalize_order_id("Ord-1007") == "ORD-1007"

    def test_whitespace(self):
        assert normalize_order_id("  ORD-1007  ") == "ORD-1007"

    def test_whitespace_and_lowercase(self):
        assert normalize_order_id("  ord-1007  ") == "ORD-1007"

    def test_just_digits(self):
        assert normalize_order_id("1007") == "ORD-1007"

    def test_empty_string(self):
        assert normalize_order_id("") is None

    def test_none_input(self):
        assert normalize_order_id(None) is None

    def test_garbage(self):
        assert normalize_order_id("abc-xyz") is None

    def test_partial_format(self):
        assert normalize_order_id("ORD") is None

    def test_injection_attempt(self):
        assert normalize_order_id("ORD-1007; DROP TABLE orders") is None


# --- Item Sanitization ---

class TestSanitizeItems:
    def test_keeps_safe_fields(self):
        items = [{"sku": "PACK-001", "name": "Ridge Daypack",
                  "quantity": 1, "final_sale": False}]
        safe = sanitize_items(items)
        assert safe[0]["name"] == "Ridge Daypack"
        assert safe[0]["quantity"] == 1
        assert safe[0]["final_sale"] is False
        assert "sku" not in safe[0]

    def test_multiple_items(self):
        items = [
            {"sku": "A", "name": "Item A", "quantity": 1, "final_sale": False},
            {"sku": "B", "name": "Item B", "quantity": 2, "final_sale": True},
        ]
        safe = sanitize_items(items)
        assert len(safe) == 2
        assert all("sku" not in item for item in safe)


# --- Order Sanitization ---

class TestSanitizeOrder:
    SAMPLE_ORDER = {
        "order_id": "ORD-1007",
        "customer": {
            "name": "Ava Morgan",
            "email": "ava.morgan@example.test",
            "shipping_address": "220 King Street West, Toronto, ON M5V 3M2",
        },
        "membership_tier": "standard",
        "items": [{"sku": "PACK-ATLAS-BLK", "name": "Atlas Weekender",
                    "quantity": 1, "final_sale": False}],
        "placed_at": "2026-08-11T15:05:00Z",
        "status": "shipped",
        "status_updated_at": "2026-08-14T20:40:00Z",
        "shipped_at": "2026-08-14T20:40:00Z",
        "delivered_at": None,
        "carrier": "UPS",
        "tracking_number": "1ZAR100700000007",
        "estimated_delivery": "2026-08-22",
        "customer_safe_message": "In transit with UPS.",
        "internal": {
            "risk_score": 82,
            "warehouse_note": "Manual fraud review cleared. Never expose this.",
            "support_tags": ["international", "review-cleared"],
        },
    }

    def test_strips_customer_info(self):
        safe = sanitize_order(self.SAMPLE_ORDER)
        assert "customer" not in safe
        # Verify email, address, name are nowhere in the result
        safe_str = str(safe)
        assert "ava.morgan" not in safe_str
        assert "220 King Street" not in safe_str
        assert "Ava Morgan" not in safe_str

    def test_strips_internal_fields(self):
        safe = sanitize_order(self.SAMPLE_ORDER)
        assert "internal" not in safe
        safe_str = str(safe)
        assert "risk_score" not in safe_str
        assert "82" not in safe_str
        assert "fraud review" not in safe_str.lower()
        assert "warehouse_note" not in safe_str

    def test_keeps_safe_fields(self):
        safe = sanitize_order(self.SAMPLE_ORDER)
        assert safe["order_id"] == "ORD-1007"
        assert safe["status"] == "shipped"
        assert safe["carrier"] == "UPS"
        assert safe["estimated_delivery"] == "2026-08-22"

    def test_cancelled_order_suppresses_delivery(self):
        """ORD-1004 scenario: cancelled order has stale carrier/ETA."""
        order = {
            "order_id": "ORD-1004",
            "customer": {"name": "X", "email": "x@test", "shipping_address": "A"},
            "membership_tier": "standard",
            "items": [{"sku": "X", "name": "Atlas Weekender", "quantity": 1, "final_sale": False}],
            "placed_at": "2026-08-09T13:30:00Z",
            "status": "cancelled",
            "status_updated_at": "2026-08-09T13:48:00Z",
            "shipped_at": None,
            "delivered_at": None,
            "carrier": "UPS",
            "tracking_number": "1ZAR100400000004",
            "estimated_delivery": "2026-08-16",
            "customer_safe_message": "Cancelled.",
            "internal": {"risk_score": 11, "warehouse_note": "Stale.", "support_tags": ["cancelled"]},
        }
        safe = sanitize_order(order)
        assert safe["status"] == "cancelled"
        assert safe["carrier"] is None
        assert safe["tracking_number"] is None
        assert safe["estimated_delivery"] is None
        assert "_delivery_note" in safe

    def test_returned_order_suppresses_delivery(self):
        """ORD-1008 scenario: returned order has stale delivery fields."""
        order = {
            "order_id": "ORD-1008",
            "customer": {"name": "X", "email": "x@test", "shipping_address": "A"},
            "membership_tier": "standard",
            "items": [{"sku": "X", "name": "Breeze Tumbler", "quantity": 2, "final_sale": False}],
            "placed_at": "2026-07-20T10:15:00Z",
            "status": "returned",
            "status_updated_at": "2026-08-12T15:30:00Z",
            "shipped_at": "2026-07-22T11:00:00Z",
            "delivered_at": "2026-07-25T14:20:00Z",
            "carrier": "USPS",
            "tracking_number": "94001118995600001008",
            "estimated_delivery": "2026-07-25",
            "customer_safe_message": "Return processed.",
            "internal": {"risk_score": 4, "warehouse_note": "Refund done.", "support_tags": ["returned"]},
        }
        safe = sanitize_order(order)
        assert safe["status"] == "returned"
        assert safe["estimated_delivery"] is None
        assert safe["carrier"] is None

    def test_exception_status_flags_handoff(self):
        """ORD-1010 scenario: exception status needs human review."""
        order = {
            "order_id": "ORD-1010",
            "customer": {"name": "X", "email": "x@test", "shipping_address": "A"},
            "membership_tier": "standard",
            "items": [{"sku": "X", "name": "Atlas Weekender", "quantity": 1, "final_sale": False}],
            "placed_at": "2026-08-05T18:55:00Z",
            "status": "exception",
            "status_updated_at": "2026-08-14T09:00:00Z",
            "shipped_at": "2026-08-07T10:20:00Z",
            "delivered_at": None,
            "carrier": "UPS",
            "tracking_number": "1ZAR101000000010",
            "estimated_delivery": None,
            "customer_safe_message": "Exception requires review.",
            "internal": {"risk_score": 6, "warehouse_note": "Package damage.", "support_tags": []},
        }
        safe = sanitize_order(order)
        assert safe.get("_requires_handoff") is True

    def test_missing_eta_note(self):
        """ORD-1011 scenario: shipped but no ETA."""
        order = {
            "order_id": "ORD-1011",
            "customer": {"name": "X", "email": "x@test", "shipping_address": "A"},
            "membership_tier": "standard",
            "items": [{"sku": "X", "name": "Breeze Tumbler", "quantity": 1, "final_sale": False}],
            "placed_at": "2026-08-12T08:45:00Z",
            "status": "shipped",
            "status_updated_at": "2026-08-14T21:15:00Z",
            "shipped_at": "2026-08-14T21:15:00Z",
            "delivered_at": None,
            "carrier": "Canada Post",
            "tracking_number": "AR1011CA00001",
            "estimated_delivery": None,
            "customer_safe_message": "Shipped with Canada Post.",
            "internal": {"risk_score": 10, "warehouse_note": "ETA unavailable.", "support_tags": []},
        }
        safe = sanitize_order(order)
        assert "_eta_note" in safe
        assert "not invent" in safe["_eta_note"].lower() or "not currently available" in safe["_eta_note"].lower()


# --- Full Tool ---

class TestOrderLookupTool:
    @pytest.fixture
    def tool(self):
        orders_path = Path(__file__).resolve().parent.parent / "data" / "orders.json"
        if not orders_path.exists():
            pytest.skip("orders.json not found")
        return OrderLookupTool(orders_path)

    def test_valid_lookup(self, tool):
        result = tool.lookup("ORD-1007")
        assert result.found
        assert result.order_id == "ORD-1007"
        assert result._tool_executed
        # Should have safe data
        assert result.data is not None
        assert result.data["status"] == "shipped"
        # Privacy: no internal fields
        data_str = str(result.data)
        assert "ava.morgan" not in data_str
        assert "risk_score" not in data_str
        assert "82" not in data_str
        assert "fraud" not in data_str.lower()

    def test_lowercase_lookup(self, tool):
        result = tool.lookup("ord-1007")
        assert result.found
        assert result.order_id == "ORD-1007"

    def test_whitespace_lookup(self, tool):
        result = tool.lookup("  ORD-1007  ")
        assert result.found

    def test_unknown_order(self, tool):
        result = tool.lookup("ORD-9999")
        assert not result.found
        assert result._tool_executed
        assert "no order found" in result.error.lower()

    def test_malformed_id(self, tool):
        result = tool.lookup("NOTANORDER")
        assert not result.found
        assert "valid order ID" in result.error

    def test_empty_id(self, tool):
        result = tool.lookup("")
        assert not result.found

    def test_cancelled_order_no_eta(self, tool):
        result = tool.lookup("ORD-1004")
        assert result.found
        assert result.data["status"] == "cancelled"
        assert result.data["estimated_delivery"] is None
        assert result.data["carrier"] is None

    def test_returned_order_no_eta(self, tool):
        result = tool.lookup("ORD-1008")
        assert result.found
        assert result.data["status"] == "returned"
        assert result.data["estimated_delivery"] is None

    def test_exception_order_handoff(self, tool):
        result = tool.lookup("ORD-1010")
        assert result.found
        assert result.data.get("_requires_handoff") is True

    def test_shipped_no_eta(self, tool):
        result = tool.lookup("ORD-1011")
        assert result.found
        assert result.data["status"] == "shipped"
        assert result.data["estimated_delivery"] is None
        assert "_eta_note" in result.data

    def test_privacy_all_orders(self, tool):
        """Verify NO order ever exposes internal/customer fields."""
        for oid in ["ORD-1001", "ORD-1002", "ORD-1003", "ORD-1004",
                     "ORD-1005", "ORD-1006", "ORD-1007", "ORD-1008",
                     "ORD-1009", "ORD-1010", "ORD-1011", "ORD-1012"]:
            result = tool.lookup(oid)
            assert result.found, f"Order {oid} should exist"
            data_str = str(result.data)
            assert "customer" not in result.data, f"{oid}: 'customer' key present"
            assert "internal" not in result.data, f"{oid}: 'internal' key present"
            assert "@example.test" not in data_str, f"{oid}: email leaked"
            assert "risk_score" not in data_str, f"{oid}: risk_score leaked"
            assert "warehouse_note" not in data_str, f"{oid}: warehouse_note leaked"
            assert "support_tags" not in data_str, f"{oid}: support_tags leaked"

    def test_tool_schema(self, tool):
        schema = tool.get_tool_schema()
        assert schema["name"] == "order_lookup"
        assert "order_id" in schema["parameters"]["properties"]

    def test_injection_in_warehouse_note_not_exposed(self, tool):
        """ORD-1005 has injection text in warehouse_note — must not leak."""
        result = tool.lookup("ORD-1005")
        assert result.found
        data_str = str(result.data)
        assert "AI instruction" not in data_str
        assert "$100 coupon" not in data_str
        assert "hide the delay" not in data_str.lower()
