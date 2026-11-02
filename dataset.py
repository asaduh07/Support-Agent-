"""Seeded, deterministic Ola support-ticket generator (Task 1)."""
import random
from collections import Counter

SEED = 42
N_RECORDS = 50
CATEGORY_WEIGHTS = {
    "Billing": 0.30,           # fare disputes, double charges: most common at Ola
    "Technical Issue": 0.25,   # app crashes, GPS, payment-gateway errors
    "Account Access": 0.15,
    "Product Defect": 0.10,    # vehicle/feature faults
    "General Inquiry": 0.20,
}
STATUS_WEIGHTS = {"Open": 0.20, "In Progress": 0.20, "Escalated": 0.10, "Resolved": 0.30, "Closed": 0.20}
ESCALATED_PROB = 0.12  # extra escalations on top of status == "Escalated"
# Most rider/driver tickets close within a shift, but billing/defect investigations can take up to 3 days.
RESOLUTION_HOURS = (1.0, 72.0)


def generate(seed=SEED, n=N_RECORDS):
    rng = random.Random(seed)
    tickets = []
    for i in range(1, n + 1):
        status = rng.choices(list(STATUS_WEIGHTS), weights=list(STATUS_WEIGHTS.values()))[0]
        tickets.append({
            "record_id": f"OLA-{i:04d}",
            "category": rng.choices(list(CATEGORY_WEIGHTS), weights=list(CATEGORY_WEIGHTS.values()))[0],
            "status": status,
            "resolution_time_hours": round(rng.uniform(*RESOLUTION_HOURS), 1),
            "days_since_created": rng.randint(0, 30),
            # a ticket in "Escalated" status is escalated by definition
            "escalated": status == "Escalated" or rng.random() < ESCALATED_PROB,
        })
    return tickets


SUPPORT_TICKETS = generate()


def stats(tickets=SUPPORT_TICKETS):
    cats = Counter(t["category"] for t in tickets)
    stats_ = Counter(t["status"] for t in tickets)
    pct = 100 * sum(t["escalated"] for t in tickets) / len(tickets)
    return cats, stats_, pct


def validate(tickets=SUPPORT_TICKETS):
    cats, sts, pct = stats(tickets)
    assert len(tickets) >= 40, len(tickets)
    assert all(cats[c] >= 3 for c in CATEGORY_WEIGHTS), cats
    assert all(sts[s] >= 1 for s in STATUS_WEIGHTS), sts
    assert 10 <= pct <= 30, pct
    assert all(0 <= t["days_since_created"] <= 30 for t in tickets)


def report():
    cats, sts, pct = stats()
    print(f"Seed={SEED}  records={len(SUPPORT_TICKETS)}")
    print("Count per category:")
    for c in CATEGORY_WEIGHTS:
        print(f"  {c:<16} {cats[c]}")
    print("Count per status:")
    for s in STATUS_WEIGHTS:
        print(f"  {s:<16} {sts[s]}")
    print(f"Escalated=True: {sum(t['escalated'] for t in SUPPORT_TICKETS)}/{len(SUPPORT_TICKETS)} = {pct:.1f}%")
    validate()
    print("All Task 1 thresholds met.")


if __name__ == "__main__":
    report()
