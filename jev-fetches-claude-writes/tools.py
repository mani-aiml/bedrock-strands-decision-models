"""Read-only lookup tools with simulated network latency, as plain functions and as Strands tools.

The records come from generate_data.py and are synthetic, made up for this demo and for nothing else. Any
resemblance to real people, companies, orders or records is purely coincidental."""

import json
import time
from dataclasses import dataclass
from typing import Any, Literal

from strands import tool

from generate_data import generate

TOOL_LATENCY_S = 1.5
REGIONS = ("US", "EU", "UK", "IN")
ID_PATTERNS = {"customer_id": r"C-\d+", "order_id": r"O-\d+", "sku": r"SKU-\d+"}
DATA_SEED = 0
DATA: dict[str, dict[str, Any]] = generate(DATA_SEED)  # built in memory; see generate_data.py


@dataclass(frozen=True)
class ToolSpec:
    description: str
    arg: str
    table: str


TOOLS = {
    "get_customer": ToolSpec("Customer profile: region, tier and their order ids.", "customer_id", "customers"),
    "get_tickets": ToolSpec("Support tickets opened by a customer.", "customer_id", "tickets"),
    "get_order": ToolSpec("Order status, date, total and the SKUs in it.", "order_id", "orders"),
    "get_shipment": ToolSpec("Carrier tracking for an order: status, last scan, ETA.", "order_id", "shipments"),
    "get_invoice": ToolSpec("Invoice for an order: amount, paid flag and card charges.", "order_id", "invoices"),
    "get_inventory": ToolSpec("Stock on hand and restock date for a SKU.", "sku", "inventory"),
    "get_warranty": ToolSpec("Warranty length and coverage for a SKU.", "sku", "warranties"),
    "get_refund_policy": ToolSpec("Refund policy text for a region.", "region", "refund_policies"),
}


def call_tool(name: str, value: str) -> str:
    """Run one lookup. The sleep stands in for a real API round trip."""
    time.sleep(TOOL_LATENCY_S)
    record = DATA[TOOLS[name].table].get(value)
    return json.dumps(record) if record is not None else f"No record for {value}"


# The same eight lookups as Strands tools, for the agent that picks its own. Strands runs the
# tools of one turn in parallel threads, so a turn with three lookups still takes 1.5 s.


@tool
def get_customer(customer_id: str) -> str:
    """Customer profile: region, tier and their order ids."""
    return call_tool("get_customer", customer_id)


@tool
def get_tickets(customer_id: str) -> str:
    """Support tickets opened by a customer."""
    return call_tool("get_tickets", customer_id)


@tool
def get_order(order_id: str) -> str:
    """Order status, date, total and the SKUs in it."""
    return call_tool("get_order", order_id)


@tool
def get_shipment(order_id: str) -> str:
    """Carrier tracking for an order: status, last scan, ETA."""
    return call_tool("get_shipment", order_id)


@tool
def get_invoice(order_id: str) -> str:
    """Invoice for an order: amount, paid flag and card charges."""
    return call_tool("get_invoice", order_id)


@tool
def get_inventory(sku: str) -> str:
    """Stock on hand and restock date for a SKU."""
    return call_tool("get_inventory", sku)


@tool
def get_warranty(sku: str) -> str:
    """Warranty length and coverage for a SKU."""
    return call_tool("get_warranty", sku)


@tool
def get_refund_policy(region: Literal["US", "EU", "UK", "IN"]) -> str:
    """Refund policy text for a region."""
    return call_tool("get_refund_policy", region)


STRANDS_TOOLS = [
    get_customer, get_tickets, get_order, get_shipment,
    get_invoice, get_inventory, get_warranty, get_refund_policy,
]
