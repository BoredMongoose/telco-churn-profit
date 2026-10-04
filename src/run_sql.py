"""Run the SQL models in order, enforce the data-quality checks, and export the marts as CSV."""
import os
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
MARTS = ROOT / "data" / "marts"


def main():
    os.chdir(ROOT)                      # SQL files use paths relative to the project root
    MARTS.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ROOT / "data" / "telco.duckdb"))
    con.execute("SET threads = 1")   # same order and sums every run, so re-running reproduces the marts exactly
    for sql_file in sorted((ROOT / "sql").glob("*.sql")):
        con.execute(sql_file.read_text(encoding="utf-8"))
        print("ran", sql_file.name)

    dq = con.execute("SELECT * FROM dq_results ORDER BY passed, check_name").df()
    print(dq.to_string(index=False))
    failed = dq[~dq.passed]
    if len(failed):
        raise SystemExit(f"{len(failed)} data-quality check(s) failed - fix before using the marts")

    marts = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables "
                                       "WHERE table_name LIKE 'mart_%' ORDER BY 1").fetchall()]
    for table in marts:
        con.execute(f"COPY {table} TO '{(MARTS / table.removeprefix('mart_')).as_posix()}.csv' (HEADER)")
    print(f"{len(marts)} marts exported to", MARTS)


if __name__ == "__main__":
    main()
