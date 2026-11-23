#!/usr/bin/env python3
"""Compare two built Shockwave provers using the Table XI protocol on the RTX 5080."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def observe(binary, env, log, size, *, check_dispatch=False):
    env = env | {"SPARTAN_N": str(size)}
    env = env | {"KITRT_VERBOSE": "1" if check_dispatch else "0"}
    with log.open("x") as stream:
        subprocess.run(["taskset", "-c", "0-7", str(binary), "test", "Spartan proof"],
                       stdout=stream, stderr=subprocess.STDOUT, env=env, check=True,
                       cwd=binary.parent, timeout=180)
    output = log.read_text()
    proof = re.search(r"spartan_prove \(([\d.]+)s", output)
    if not (proof and "spartan_verify (" in output and "1 | 1 passed" in output):
        raise RuntimeError(f"Missing proof or verification in {log}")
    if check_dispatch and not re.search(
            r"^kitcuda: (?:nested )?kernel '_cuabi_kern_[^']+' launch parameters:$",
            output, re.MULTILINE):
        raise RuntimeError(f"No CUDA kernel launch observed in {log}")
    return float(proof[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path, help="new directory for logs, CSV and summaries")
    parser.add_argument("--sizes", type=int, nargs="+", choices=range(16, 22),
                        default=[18, 19, 20, 21], help="log2 input sizes (default 18–21)")
    parser.add_argument("--reference-library-path", help="LD_LIBRARY_PATH for the reference only")
    parser.add_argument("--candidate-library-path", help="LD_LIBRARY_PATH for the candidate only")
    args = parser.parse_args()
    if len(args.sizes) != len(set(args.sizes)):
        parser.error("each size may appear only once")
    builds = {"reference": args.reference.resolve(strict=True),
              "candidate": args.candidate.resolve(strict=True)}
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"gpu": subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True).strip(), "cpu_affinity": "0-7", "workers": 8, "sizes": args.sizes}
    environments = {}
    for label, binary in builds.items():
        env = os.environ | {"CILK_NWORKERS": "8", "CUDA_VISIBLE_DEVICES": "0"}
        library_path = getattr(args, label + "_library_path")
        if library_path is not None:
            env["LD_LIBRARY_PATH"] = library_path
        environments[label] = env
        libraries = subprocess.check_output(["ldd", binary], env=env, text=True)
        runtime = re.search(r"libkitrt\.so\S* => (.*?) \(", libraries)
        if not runtime:
            parser.error(f"{label} has no resolved kitrt library")
        metadata[label] = {"binary": str(binary),
                           "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                           "runtime_sha256": hashlib.sha256(Path(runtime[1]).read_bytes()).hexdigest(),
                           "ldd": libraries}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for label, binary in builds.items():
        observe(binary, environments[label], args.output / f"dispatch-{label}.log",
                min(args.sizes), check_dispatch=True)

    for rep in range(1, 4):
        for label, binary in builds.items():
            observe(binary, environments[label], args.output / f"warmup-{label}-{rep}.log", 20)

    summaries = []
    with (args.output / "samples.csv").open("x", newline="", buffering=1) as stream:
        writer = csv.writer(stream)
        writer.writerow(["log2_constraints", "pair", "order", "build", "proof_s"])
        for size in args.sizes:
            measurements = {label: [] for label in builds}
            for rep in range(1, 11):
                order = list(builds) if rep % 2 else list(reversed(builds))
                for position, label in enumerate(order, 1):
                    log = args.output / f"n{size}-{rep:02}-{label}.log"
                    proof = observe(builds[label], environments[label], log, size)
                    writer.writerow([size, rep, position, label, proof])
                    measurements[label].append(proof)
            reference = measurements["reference"]
            candidate = measurements["candidate"]
            ratio = statistics.median(c / r for c, r in zip(candidate, reference))
            mean_ratio = statistics.mean(candidate) / statistics.mean(reference)
            row = {"log2_constraints": size, "reference_mean_s": statistics.mean(reference),
                   "candidate_mean_s": statistics.mean(candidate),
                   "reference_sd_s": statistics.stdev(reference),
                   "candidate_sd_s": statistics.stdev(candidate),
                   "median_paired_ratio": ratio, "mean_ratio": mean_ratio,
                   "review": abs(ratio - 1) > 0.05 or abs(mean_ratio - 1) > 0.05}
            summaries.append(row)
            print(f"2^{size}: reference {row['reference_mean_s']:.4f} s, "
                  f"candidate {row['candidate_mean_s']:.4f} s, paired shift {ratio-1:+.2%}",
                  flush=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return int(any(row["review"] for row in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
