"""Generate the synthetic support catalog the lookups read, instead of shipping a data file.

Seeded and model-free. Each task in tasks.json needs one situation to be true in the data: an
order held at customs past the EU 10-day limit, a double charge, a dead switch under warranty with
none in stock, and so on. Those situations are planted around the ids the tasks name. Everything
else (names, dates, amounts, carriers, cities, stock levels, ticket ids) is drawn from the seed,
and `--extra` adds distractor customers with orders of their own.

`check()` verifies every planted situation and every cross-reference, so any seed yields a catalog
the eight tasks can be answered from.

    python generate_data.py                       # print a summary of the seed-0 catalog
    python generate_data.py --seed 7 --extra 20 --out data.json

All of it is made up. Any resemblance to real people, companies, orders or records is coincidental."""

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

TODAY = date(2026, 9, 21)
TABLES = ("customers", "orders", "shipments", "invoices", "inventory", "warranties", "tickets", "refund_policies")

FIRST = ["Priya", "Marcus", "Aiko", "Ravi", "Lena", "Tomas", "Sofia", "Kwame", "Mei", "Diego", "Anya", "Omar",
         "Hannah", "Kenji", "Fatima", "Lucas", "Ingrid", "Arjun", "Chloe", "Mateo"]
LAST = ["Nair", "Hill", "Tanaka", "Menon", "Novak", "Silva", "Okafor", "Chen", "Garcia", "Petrova", "Haddad",
        "Berg", "Sato", "Rahman", "Moreau", "Kowalski", "Iyer", "Walsh", "Rossi", "Lindqvist"]
TIERS = ("starter", "business", "enterprise")
CARRIERS = {"US": ("UPS", "FedEx"), "EU": ("DHL", "DPD"), "UK": ("Royal Mail", "DPD UK"), "IN": ("Blue Dart", "Delhivery")}
CITIES = {"US": ("Austin", "Denver", "Columbus"), "EU": ("Leipzig", "Lyon", "Milan"),
          "UK": ("Leeds", "Bristol", "Glasgow"), "IN": ("Pune", "Chennai", "Jaipur")}
PRODUCTS = {  # sku: name, price range, warranty months, coverage
    "SKU-301": ("Edge gateway G2", (1500, 2000), 24, "hardware faults, advance replacement"),
    "SKU-302": ("Sensor pack (10)", (180, 320), 12, "hardware faults"),
    "SKU-303": ("Rack mount kit", (150, 300), 12, "manufacturing defects"),
    "SKU-304": ("Core switch S48", (4500, 5500), 36, "hardware faults, next business day replacement"),
    "SKU-305": ("PoE injector", (250, 450), 12, "hardware faults"),
}
REFUND_POLICIES = {  # policy, not data: fixed, because the tasks' answers depend on its wording
    "US": "Full refund within 30 days of delivery. Duplicate charges are refunded in 3 business days.",
    "EU": "Full refund within 14 days of delivery. Orders stuck in transit over 10 days may be refunded or reshipped at the customer's choice.",
    "UK": "Full refund within 30 days of delivery. Faulty goods may be replaced or refunded within 6 months.",
    "IN": "Refund within 10 days of delivery. Replacement preferred for hardware faults.",
}


class Catalog:
    """Builds the tables record by record, keeping every cross-reference consistent."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)  # nosec B311: seeded synthetic data, not security
        self.data: dict[str, dict[str, Any]] = {table: {} for table in TABLES}
        self.price = {sku: round(self.rng.uniform(*spec[1]), -1) for sku, spec in PRODUCTS.items()}
        self.names = [f"{f} {l}" for f in FIRST for l in LAST]  # 400 names, drawn without repeats
        self.rng.shuffle(self.names)
        self.ticket_no = self.rng.randint(800, 850)

    def ago(self, lo: int, hi: int) -> date:
        return TODAY - timedelta(days=self.rng.randint(lo, hi))

    def ahead(self, lo: int, hi: int) -> date:
        return TODAY + timedelta(days=self.rng.randint(lo, hi))

    def customer(self, cid: str, region: str, tier: str | None = None) -> None:
        self.data["customers"][cid] = {"name": self.names.pop(), "region": region,
                                       "tier": tier or self.rng.choice(TIERS), "order_ids": []}
        self.data["tickets"][cid] = []

    def order(self, oid: str, cid: str, status: str, placed: date, skus: list[str]) -> float:
        total = round(sum(self.price[s] for s in skus), 2)
        self.data["orders"][oid] = {"customer_id": cid, "status": status, "placed": placed.isoformat(),
                                    "skus": skus, "total": total}
        self.data["customers"][cid]["order_ids"].append(oid)
        return total

    def invoice(self, oid: str, paid: bool, times_charged: int = 1) -> None:
        order = self.data["orders"][oid]
        charge = {"date": order["placed"], "amount": order["total"]}
        self.data["invoices"][oid] = {"amount": order["total"], "paid": paid,
                                      "charges": [dict(charge) for _ in range(times_charged if paid else 0)]}

    def shipment(self, oid: str, status: str, last_scan: date, eta: date | None) -> None:
        region = self.data["customers"][self.data["orders"][oid]["customer_id"]]["region"]
        self.data["shipments"][oid] = {
            "carrier": self.rng.choice(CARRIERS[region]), "status": status,
            "last_scan": f"{last_scan.isoformat()} {self.rng.choice(CITIES[region])}",
            "eta": eta.isoformat() if eta else None,
        }

    def ticket(self, cid: str, opened: date, subject: str) -> None:
        self.ticket_no += self.rng.randint(1, 9)
        self.data["tickets"][cid].append({"id": f"T-{self.ticket_no}", "opened": opened.isoformat(),
                                          "subject": subject, "status": "open"})

    def stock(self, sku: str, in_stock: int, restock: date | None = None) -> None:
        name, _, months, covers = PRODUCTS[sku]
        self.data["inventory"][sku] = {"name": name, "in_stock": in_stock,
                                       "restock": restock.isoformat() if restock else None}
        self.data["warranties"][sku] = {"months": months, "covers": covers}


def plant_scenarios(c: Catalog) -> None:
    """The situations tasks.json asks about, one block per task, around the ids it names."""
    c.stock("SKU-301", c.rng.randint(5, 30))
    c.stock("SKU-302", c.rng.randint(100, 300))
    low = c.rng.randint(0, 6)
    c.stock("SKU-303", low, c.ahead(7, 20) if low < 5 else None)
    c.stock("SKU-304", 0, c.ahead(14, 30))                        # dead-switch: none to ship today
    c.stock("SKU-305", 0, c.ahead(5, 15))                         # backorder and unpaid: waiting on stock

    # late-gateway: an EU gateway order held at customs for more than 10 days, paid, with a ticket
    c.customer("C-17", "EU")
    placed = c.ago(17, 22)
    c.order("O-1042", "C-17", "shipped", placed, ["SKU-301"])
    scan = c.ago(11, 15)
    c.shipment("O-1042", "held at customs", scan, None)
    c.invoice("O-1042", paid=True)
    c.ticket("C-17", scan + timedelta(days=c.rng.randint(1, 3)), "Where is my gateway?")

    # unpaid: the same customer's second order, processing, not paid, and one SKU out of stock
    c.order("O-1077", "C-17", "processing", c.ago(5, 8), ["SKU-302", "SKU-305"])
    c.invoice("O-1077", paid=False)

    # double-charge: a delivered US order charged twice on the same day, with a ticket
    c.customer("C-23", "US", "starter")
    placed = c.ago(25, 35)
    c.order("O-1051", "C-23", "delivered", placed, ["SKU-303"])
    delivered = placed + timedelta(days=c.rng.randint(2, 5))
    c.shipment("O-1051", "delivered", delivered, delivered)
    c.invoice("O-1051", paid=True, times_charged=2)
    c.ticket("C-23", delivered + timedelta(days=c.rng.randint(0, 3)), "Charged twice")

    # dead-switch and account-review: a UK switch delivered weeks ago, well inside its warranty
    c.customer("C-31", "UK", "enterprise")
    placed = c.ago(20, 30)
    c.order("O-1063", "C-31", "delivered", placed, ["SKU-304"])
    delivered = placed + timedelta(days=c.rng.randint(2, 4))
    c.shipment("O-1063", "delivered", delivered, delivered)
    c.invoice("O-1063", paid=True)
    c.ticket("C-31", delivered + timedelta(days=c.rng.randint(1, 5)),
             f"Switch port {c.rng.randint(1, 48)} dead on arrival")

    # backorder: the same customer's paid order, waiting on a SKU with a restock date
    c.order("O-1090", "C-31", "backordered", c.ago(2, 6), ["SKU-305"])
    c.invoice("O-1090", paid=True)

    # in-transit: an Indian order on its way and paid, no tickets
    c.customer("C-48", "IN", "business")
    c.order("O-1084", "C-48", "shipped", c.ago(8, 12), ["SKU-301", "SKU-303"])
    c.shipment("O-1084", "in transit", c.ago(1, 3), c.ahead(1, 4))
    c.invoice("O-1084", paid=True)

    c.data["refund_policies"] = dict(REFUND_POLICIES)             # policy-only: the UK policy


def add_distractors(c: Catalog, n: int) -> None:
    """Customers C-50 onward with one to three ordinary orders each, O-2000 onward."""
    order_no = 2000
    for i in range(n):
        cid = f"C-{50 + i}"
        c.customer(cid, c.rng.choice(tuple(REFUND_POLICIES)))
        for _ in range(c.rng.randint(1, 3)):
            oid, order_no = f"O-{order_no}", order_no + 1
            status = c.rng.choice(("delivered", "delivered", "shipped", "processing"))
            placed = c.ago(3, 60)
            c.order(oid, cid, status, placed, c.rng.sample(sorted(PRODUCTS), c.rng.randint(1, 2)))
            c.invoice(oid, paid=status != "processing" or c.rng.random() < 0.5)
            if status == "delivered":
                delivered = min(placed + timedelta(days=c.rng.randint(2, 6)), TODAY)
                c.shipment(oid, "delivered", delivered, delivered)
            elif status == "shipped":
                c.shipment(oid, "in transit", c.ago(0, 2), c.ahead(1, 5))


def _require(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"catalog check failed: {what}")


def check(data: dict[str, Any]) -> None:
    """Every cross-reference resolves, and every planted situation holds."""
    customers, orders, shipments, invoices, inventory = (data[t] for t in TABLES[:5])
    for oid, order in orders.items():
        _require(oid in customers[order["customer_id"]]["order_ids"], oid)
        _require(all(sku in inventory and sku in data["warranties"] for sku in order["skus"]), oid)
        _require(invoices[oid]["amount"] == order["total"], oid)
    for cid, customer in customers.items():
        _require(all(orders[oid]["customer_id"] == cid for oid in customer["order_ids"]), cid)
        _require(cid in data["tickets"] and customer["region"] in data["refund_policies"], cid)

    held = date.fromisoformat(shipments["O-1042"]["last_scan"].split()[0])
    _require(customers["C-17"]["region"] == "EU" and shipments["O-1042"]["status"] == "held at customs", "late-gateway")
    _require((TODAY - held).days > 10, "late-gateway: stuck in transit over 10 days")
    _require(orders["O-1077"]["status"] == "processing" and not invoices["O-1077"]["paid"], "unpaid")
    charges = invoices["O-1051"]["charges"]
    _require(len(charges) == 2 and charges[0] == charges[1], "double-charge")
    delivered = date.fromisoformat(shipments["O-1063"]["eta"])
    _require((TODAY - delivered).days < 30 * data["warranties"]["SKU-304"]["months"], "dead-switch: under warranty")
    _require(inventory["SKU-304"]["in_stock"] == 0, "dead-switch: none to ship today")
    _require(orders["O-1090"]["status"] == "backordered" and inventory["SKU-305"]["in_stock"] == 0, "backorder")
    _require(inventory["SKU-305"]["restock"], "backorder: a restock date to quote")
    _require(shipments["O-1084"]["status"] == "in transit" and invoices["O-1084"]["paid"], "in-transit")
    _require(data["tickets"]["C-31"] and data["tickets"]["C-31"][0]["status"] == "open", "account-review")


def generate(seed: int = 0, extra: int = 0) -> dict[str, dict[str, Any]]:
    catalog = Catalog(seed)
    plant_scenarios(catalog)
    add_distractors(catalog, extra)
    check(catalog.data)
    return catalog.data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", type=int, default=0, help="distractor customers to add")
    ap.add_argument("--out", type=Path, help="write the catalog as JSON here")
    args = ap.parse_args()
    data = generate(args.seed, args.extra)
    if args.out:
        args.out.write_text(json.dumps(data, indent=2))
    print(", ".join(f"{len(data[t])} {t}" for t in TABLES) + (f" -> {args.out}" if args.out else ""))


if __name__ == "__main__":
    main()
