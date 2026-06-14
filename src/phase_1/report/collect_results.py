from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.phase_1.common import ensure_output_dir, format_cell, timestamp


REQUIRED = {
    "search": {"1-A", "1-B", "1-C", "1-D", "1-E", "1-F", "1-G", "1-Regret"},
    "planning": {"2-A", "2-B", "2-C", "2-D", "2-E", "2-F", "2-G", "2-H"},
    "qlearning": {"3-A", "3-B", "3-C", "3-D"},
}


def latest_json(output_dir: str, prefix: str) -> str | None:
    paths = glob.glob(os.path.join(output_dir, f"{prefix}_*.json"))
    return max(paths, key=os.path.getmtime) if paths else None


def load_payload(path: str | None) -> dict | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_table(f, rows: list[dict]) -> None:
    if not rows:
        f.write("_No rows found._\n\n")
        return
    headers = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join("---" for _ in headers) + " |\n")
    for row in rows:
        f.write("| " + " | ".join(format_cell(row.get(header, "")) for header in headers) + " |\n")
    f.write("\n")


def collect(output_dir: str) -> str:
    ensure_output_dir(output_dir)
    payloads = {prefix: load_payload(latest_json(output_dir, prefix)) for prefix in REQUIRED}
    out_path = os.path.join(output_dir, f"phase_1_summary_{timestamp()}.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Phase 1 Summary\n\n")
        for prefix, required_ids in REQUIRED.items():
            payload = payloads[prefix]
            f.write(f"## {prefix}\n\n")
            if payload is None:
                f.write("_Missing result file._\n\n")
                continue
            rows = payload.get("summary", [])
            found = {row.get("experiment") for row in rows}
            missing = sorted(required_ids - found)
            if missing:
                f.write(f"Missing experiments: `{', '.join(missing)}`\n\n")
            else:
                f.write("All required experiment IDs found.\n\n")
            write_table(f, rows)

    print(f"Phase-1 summary saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Collect latest phase-1 result files into one report.")
    parser.add_argument("--output-dir", default=os.path.join("models", "eval_results"))
    args = parser.parse_args()
    collect(args.output_dir)


if __name__ == "__main__":
    main()
