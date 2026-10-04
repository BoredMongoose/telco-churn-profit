"""Download IBM's Telco Customer Churn sample data (7,043 customers of a fictional telecom company)."""
from pathlib import Path

import requests

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    OUT.write_bytes(r.content)
    print(f"saved {OUT} ({r.text.count(chr(10)) - 1:,} customers)")


if __name__ == "__main__":
    main()
