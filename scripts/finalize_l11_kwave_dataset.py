"""Wait for raw workers, then atomically pack and validate all 64 shards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("/data/zhuangyang/NumerialBreastPhantoms/l11_neural_asp_64"))
    ap.add_argument("--config", type=Path,
                    default=PROJECT / "configs/l11_kwave.yaml")
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--timeout-hours", type=float, default=4.0)
    args = ap.parse_args()
    manifest = json.loads((args.root / "index.json").read_text())
    expected = len(manifest["samples"])
    deadline = time.monotonic() + args.timeout_hours * 3600
    last_count = -1
    while time.monotonic() < deadline:
        errors = sorted((args.root / "raw").glob("*.error.txt"))
        if errors:
            messages = [f"{p.name}: {p.read_text().strip()}" for p in errors]
            raise RuntimeError("raw generation failed: " + "; ".join(messages))
        complete = sum((args.root / r["raw_path"]).exists()
                       for r in manifest["samples"])
        if complete != last_count:
            print(f"raw complete {complete}/{expected}", flush=True)
            last_count = complete
        if complete == expected:
            break
        time.sleep(args.poll_seconds)
    else:
        raise TimeoutError(f"timed out with {last_count}/{expected} raw shards")

    packer = PROJECT / "scripts/pack_l11_kwave_dataset.py"
    subprocess.run([sys.executable, str(packer), "--config", str(args.config),
                    "--root", str(args.root), "--mode", "pack"], check=True)
    subprocess.run([sys.executable, str(packer), "--config", str(args.config),
                    "--root", str(args.root), "--mode", "validate"], check=True)
    shard_bytes = sum((args.root / r["path"]).stat().st_size
                      for r in manifest["samples"])
    result = {"ok": True, "raw_samples": expected, "packed_samples": expected,
              "shard_bytes": shard_bytes,
              "completed": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (args.root / "READY").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
