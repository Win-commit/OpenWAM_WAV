"""Materialize non-mutating sparse WAV value sidecars for RoboDojo.

Example:

    python scripts/prepare_robodojo_value_targets.py \
        --dataset-dir /data/robodojo \
        --sidecar-root /data/robodojo_wav_values

The source HDF5 files are read-only inputs.  The output tree mirrors episode
paths and contains per-episode values plus one global Z-score stats JSON.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openwam.dataloader.utils.robodojo_value import (  # noqa: E402
    DEFAULT_VALUE_GAMMA,
    build_robodojo_value_sidecars,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Formal RoboDojo dataset root")
    parser.add_argument("--sidecar-root", required=True, help="External directory for derived .npz sidecars")
    parser.add_argument("--embodiment", default="arx_x5")
    parser.add_argument("--variant", default="sim", choices=("sim", "real"))
    parser.add_argument("--gamma", type=float, default=DEFAULT_VALUE_GAMMA)
    parser.add_argument(
        "--value-dim",
        type=int,
        default=1,
        help="Target width; the scalar sparse-return proxy is broadcast to every channel when > 1",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Restrict conversion to one task (repeatable); default is the complete corpus",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate valid sidecars as well")
    parser.add_argument(
        "--fast-fingerprint",
        action="store_true",
        help="Skip full-file SHA-256; validate source size, mtime, and episode length instead",
    )
    args = parser.parse_args(argv)

    stats_path = build_robodojo_value_sidecars(
        args.dataset_dir,
        args.sidecar_root,
        tasks=args.tasks,
        embodiment=args.embodiment,
        variant=args.variant,
        gamma=args.gamma,
        value_dim=args.value_dim,
        overwrite=args.overwrite,
        include_source_sha256=not args.fast_fingerprint,
    )
    print(f"wrote RoboDojo WAV value sidecars; global Z-score stats: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
