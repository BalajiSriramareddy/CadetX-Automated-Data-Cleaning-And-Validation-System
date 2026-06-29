"""
SYNTHETIC DATASET GENERATOR
===========================
Generates a synthetic UK rural fibre-broadband customer dataset, modelled on
the domain of a provider like Gigaclear (rural counties, gigabit packages,
churn, NPS). The data is ENTIRELY SYNTHETIC — no real customers.

It deliberately plants known data-quality issues so the profiling, cleaning,
and validation modules have real problems to find:

  ISSUES INJECTED (ground truth — keep this list for your demo)
  ------------------------------------------------------------
  1. Missing values        scattered across email, postcode, nps_score, tenure
  2. Duplicate rows         exact duplicates of some customers
  3. Invalid emails         missing @ or domain
  4. Invalid postcodes      wrong UK format
  5. Invalid phone numbers  too short / non-numeric
  6. Out-of-range NPS       values above 10
  7. Impossible numbers     negative tenure, negative monthly charge
  8. Inconsistent churn      "Yes" / "yes" / "Y" / "1" all meaning the same
  9. Mixed-type column       monthly_charges has some "£29.99" strings
 10. Inconsistent casing     region values with stray spaces / lower case
 11. Inconsistent dates      mix of YYYY-MM-DD, DD/MM/YYYY, "Mon YYYY"

Run:  python data/generate_dataset.py
Out:  data/raw/broadband_customers.csv
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

SEED = 42
N_ROWS = 1000
OUT = Path(__file__).resolve().parent / "raw" / "broadband_customers.csv"

random.seed(SEED)

# --- domain pools (Gigaclear-style rural fibre) ---
REGIONS = ["Oxfordshire", "Gloucestershire", "Wiltshire", "Cambridgeshire",
           "Northamptonshire", "Buckinghamshire", "Hampshire", "Berkshire"]
PROPERTY_TYPES = ["Detached", "Semi-detached", "Cottage", "Farmhouse",
                  "Bungalow", "Terraced"]
PACKAGES = {"Fibre 100": 29.99, "Fibre 300": 44.99,
            "Fibre 500": 54.99, "Fibre 900": 64.99}
CONTRACTS = ["Monthly", "12-month", "24-month"]
PAYMENT = ["Direct Debit", "Credit Card", "Bank Transfer"]
FIRST = ["James", "Olivia", "Harry", "Amelia", "George", "Isla", "Noah",
         "Ava", "Oliver", "Emily", "Jack", "Sophie", "Charlie", "Grace",
         "Thomas", "Lily", "William", "Freya", "Henry", "Poppy"]
LAST = ["Smith", "Jones", "Taylor", "Brown", "Williams", "Wilson", "Evans",
        "Davies", "Thomas", "Roberts", "Walker", "Wright", "Green", "Hall",
        "Clarke", "Patel", "Wood", "Turner", "Hughes", "Edwards"]
POSTCODE_AREAS = ["OX", "GL", "SN", "CB", "NN", "MK", "SO", "RG"]


def valid_postcode() -> str:
    area = random.choice(POSTCODE_AREAS)
    return f"{area}{random.randint(1, 49)} {random.randint(1, 9)}" \
           f"{random.choice('ABDEFGHJLNPQRSTUWXYZ')}" \
           f"{random.choice('ABDEFGHJLNPQRSTUWXYZ')}"


def valid_phone() -> str:
    return "07" + "".join(str(random.randint(0, 9)) for _ in range(9))


def make_email(first: str, last: str) -> str:
    domain = random.choice(["gmail.com", "outlook.com", "yahoo.co.uk", "btinternet.com"])
    return f"{first.lower()}.{last.lower()}{random.randint(1, 99)}@{domain}"


def make_row(i: int) -> dict:
    first, last = random.choice(FIRST), random.choice(LAST)
    pkg = random.choice(list(PACKAGES))
    base_charge = PACKAGES[pkg]
    tenure = random.randint(1, 72)
    return {
        "customer_id": 10000 + i,
        "full_name": f"{first} {last}",
        "email": make_email(first, last),
        "phone": valid_phone(),
        "postcode": valid_postcode(),
        "region": random.choice(REGIONS),
        "property_type": random.choice(PROPERTY_TYPES),
        "package": pkg,
        "monthly_charges": round(base_charge + random.uniform(-2, 5), 2),
        "contract_type": random.choice(CONTRACTS),
        "tenure_months": tenure,
        "install_date": f"20{random.randint(20, 25):02d}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        "payment_method": random.choice(PAYMENT),
        "nps_score": random.randint(0, 10),
        "support_tickets": random.choice([0, 0, 0, 1, 1, 2, 3, 5]),
        "churn": random.choice(["No", "No", "No", "Yes"]),
    }


def inject_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Plant the known data-quality problems. See module docstring for the list."""
    df = df.copy()
    n = len(df)

    # Allow mixed/string values to be planted into otherwise-numeric columns
    for col in ["monthly_charges", "nps_score", "tenure_months"]:
        df[col] = df[col].astype(object)

    def pick(frac):
        return random.sample(range(n), int(n * frac))

    # 1. missing values
    for idx in pick(0.06):
        df.at[idx, "email"] = None
    for idx in pick(0.05):
        df.at[idx, "postcode"] = None
    for idx in pick(0.04):
        df.at[idx, "nps_score"] = None
    for idx in pick(0.03):
        df.at[idx, "tenure_months"] = None

    # 3. invalid emails
    for idx in pick(0.04):
        df.at[idx, "email"] = random.choice(["john.smith", "no-at-sign.com", "a@b"])

    # 4. invalid postcodes
    for idx in pick(0.03):
        df.at[idx, "postcode"] = random.choice(["12345", "ABCDE", "OX 99"])

    # 5. invalid phones
    for idx in pick(0.03):
        df.at[idx, "phone"] = random.choice(["123", "not-a-number", "07xx"])

    # 6. out-of-range NPS
    for idx in pick(0.02):
        df.at[idx, "nps_score"] = random.choice([11, 15, 99])

    # 7. impossible numbers
    for idx in pick(0.02):
        df.at[idx, "tenure_months"] = random.choice([-5, -1])
    for idx in pick(0.02):
        df.at[idx, "monthly_charges"] = random.choice([-10.0, -29.99])

    # 8. inconsistent churn labels
    for idx in pick(0.10):
        df.at[idx, "churn"] = random.choice(["yes", "Y", "1", "no", "N", "0"])

    # 9. mixed-type monthly_charges (string with £)
    for idx in pick(0.03):
        df.at[idx, "monthly_charges"] = f"£{df.at[idx, 'monthly_charges']}"

    # 10. inconsistent region casing / spacing
    for idx in pick(0.05):
        r = str(df.at[idx, "region"])
        df.at[idx, "region"] = random.choice([r.lower(), f"  {r} ", r.upper()])

    # 11. inconsistent date formats
    for idx in pick(0.05):
        d = str(df.at[idx, "install_date"])
        y, m, day = d.split("-")
        df.at[idx, "install_date"] = random.choice([
            f"{day}/{m}/{y}",
            f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(m)-1]} {y}",
        ])

    # 2. duplicate rows (append exact copies of some rows)
    dupes = df.iloc[random.sample(range(n), int(n * 0.03))]
    df = pd.concat([df, dupes], ignore_index=True)

    return df


def main():
    rows = [make_row(i) for i in range(N_ROWS)]
    df = pd.DataFrame(rows)
    df = inject_issues(df)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")
    print("Reminder: this data is SYNTHETIC. Issues planted on purpose — see docstring.")


if __name__ == "__main__":
    main()
