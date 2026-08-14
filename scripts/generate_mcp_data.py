"""Generate the MCP server's dummy enterprise data.

Derived from the document corpus rather than invented independently, so the
structured records and the narrative documents describe the same events. That
is what makes the combination worth having: the corpus explains *why* an
incident happened in prose, and the MCP records let the analysis tool count and
group them. An agent that can do both can answer "which service fails most often
and why", which neither source supports alone.

    uv run python scripts/generate_mcp_data.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import yaml

SEED = 20260814
CORPUS_DIR = Path("data/corpus")
OUTPUT_DIR = Path("mcp_server/data")

TEAMS = [
    ("Payments Platform Team", "payments"),
    ("Core Platform Engineering", "platform"),
    ("Information Security", "security"),
    ("Operational Risk", "risk"),
    ("Retail Products", "retail-banking"),
    ("People Operations", "people"),
]

FIRST_NAMES = [
    "Nimali",
    "Dilan",
    "Ruwan",
    "Sanduni",
    "Kasun",
    "Ishara",
    "Tharindu",
    "Amaya",
    "Chathura",
    "Nadeesha",
    "Suresh",
    "Malith",
    "Hasini",
    "Roshan",
    "Piyumi",
    "Lakmal",
    "Dinesha",
    "Anushka",
    "Kavindu",
    "Shalini",
    "Buddhika",
    "Tharushi",
]
SURNAMES = [
    "Perera",
    "Fernando",
    "Jayasinghe",
    "Silva",
    "Wickramasinghe",
    "Bandara",
    "Rajapaksa",
    "Gunawardena",
    "Dissanayake",
    "Ekanayake",
    "Weerasinghe",
    "Herath",
    "Senanayake",
    "Amarasinghe",
    "Karunaratne",
]

TITLES = {
    "payments": ["Payments Engineer", "Senior Payments Engineer", "Payments Lead"],
    "platform": ["Platform Engineer", "Site Reliability Engineer", "Principal Engineer"],
    "security": ["Security Analyst", "Security Engineer", "Head of Security"],
    "risk": ["Risk Analyst", "Operational Risk Manager"],
    "retail-banking": ["Product Manager", "Product Owner", "Head of Retail Products"],
    "people": ["People Partner", "Talent Lead"],
}

SALARY_BANDS = ["B3", "B4", "B5", "C1", "C2", "D1"]

SERVICES = [
    ("settlement-service", "payments", 1, ["core-ledger", "card-processor-bridge"]),
    ("authorization-gateway", "payments", 1, ["iso8583-adapter", "fraud-scoring"]),
    ("card-processor-bridge", "payments", 1, ["card-network-link"]),
    ("payment-orchestrator", "payments", 2, ["settlement-service", "authorization-gateway"]),
    ("iso8583-adapter", "payments", 1, ["card-network-link"]),
    ("reconciliation-engine", "payments", 2, ["core-ledger", "settlement-service"]),
    ("core-ledger", "platform", 1, []),
    ("identity-platform", "platform", 1, []),
    ("event-backbone", "platform", 1, []),
    ("observability-pipeline", "platform", 3, ["event-backbone"]),
    ("data-platform", "platform", 3, ["event-backbone", "core-ledger"]),
    ("fraud-scoring", "security", 2, ["data-platform"]),
    ("card-network-link", "payments", 1, []),
]


def load_incidents() -> list[dict[str, object]]:
    """Build structured incident records from the corpus incident documents."""
    records: list[dict[str, object]] = []
    if not CORPUS_DIR.is_dir():
        return records

    known_causes = {
        "connection-pool-exhaustion",
        "certificate-expiry",
        "downstream-gateway-timeout",
        "retry-storm",
        "database-failover-lag",
        "autoscaling-misconfiguration",
        "schema-migration-defect",
        "third-party-degradation",
    }

    for path in sorted(CORPUS_DIR.glob("INC-*.md")):
        raw = path.read_text(encoding="utf-8")
        frontmatter = yaml.safe_load(raw.split("---", 2)[1])
        body = raw.split("---", 2)[2]

        tags = frontmatter.get("tags", [])
        cause = next((t for t in tags if t in known_causes), "unclassified")
        service = next(
            (
                t
                for t in tags
                if "-" in t
                and t not in known_causes
                and t not in {"outage", "payments", "platform"}
                and not t.startswith("sev")
            ),
            "unknown",
        )

        failed = _first_int(body, r"Failed authorizations: ([\d,]+)")
        customers = _first_int(body, r"Customers affected: ([\d,]+)")
        duration = _first_int(body, r"Duration: (\d+) minutes")
        severity = next((t.upper() for t in tags if t.startswith("sev")), "SEV3")

        records.append(
            {
                "incident_id": frontmatter["doc_id"],
                "service": service,
                "department": frontmatter["department"],
                "severity": severity,
                "opened_date": str(frontmatter["created_date"]),
                "duration_minutes": duration,
                "failed_transactions": failed,
                "customers_affected": customers,
                "root_cause_category": cause,
                "status": "resolved",
                # Links the structured record back to the narrative document,
                # so an agent can move between the two.
                "report_doc_id": frontmatter["doc_id"],
                "access_level": frontmatter["access_level"],
            }
        )
    return records


def _first_int(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    return int(match.group(1).replace(",", "")) if match else 0


def build_employees(rng: random.Random) -> list[dict[str, object]]:
    employees: list[dict[str, object]] = []
    used: set[str] = set()

    for index in range(48):
        while True:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(SURNAMES)}"
            if name not in used:
                used.add(name)
                break

        team, department = rng.choice(TEAMS)
        employees.append(
            {
                "employee_id": f"E{1000 + index}",
                "name": name,
                "title": rng.choice(TITLES[department]),
                "department": department,
                "team": team,
                "email": f"{name.split()[0].lower()}.{name.split()[1].lower()}@combank.example",
                "location": rng.choice(["Colombo", "Colombo", "Kandy", "Remote"]),
                "on_call": rng.random() < 0.3,
                # Deliberately sensitive. The MCP server returns this field to
                # anyone who asks; our client redacts it below the administrator
                # ceiling. A third-party tool server cannot be trusted to
                # enforce our access policy — see app/tools/mcp_client.py.
                "salary_band": rng.choice(SALARY_BANDS),
            }
        )
    return employees


def build_services() -> list[dict[str, object]]:
    return [
        {
            "service_name": name,
            "department": department,
            "tier": tier,
            "owner_team": next(t for t, d in TEAMS if d == department),
            "dependencies": dependencies,
            "runbook": f"RB-{department[:3].upper()}-001",
            "sla_availability": {1: "99.95%", 2: "99.9%", 3: "99.5%"}[tier],
        }
        for name, department, tier, dependencies in SERVICES
    ]


def main() -> None:
    rng = random.Random(SEED)  # noqa: S311 — reproducible fixtures, not security
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    incidents = load_incidents()
    datasets = {
        "employees.json": build_employees(rng),
        "services.json": build_services(),
        "incidents.json": incidents,
    }

    for filename, payload in datasets.items():
        (OUTPUT_DIR / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        print(f"  {filename}: {len(payload)} records")

    if not incidents:
        print("\n  Note: no corpus found, so incidents.json is empty.")
        print("  Run `make corpus` first to link incident records to their reports.")


if __name__ == "__main__":
    main()
