# Clang 19 artifact checkout

The repository root builds Kitsune. The `opencilk-project/` subtree builds the OpenCilk frontend and CPU runtime. The Shockwave path uses `cilk-kitsune/ock++` to pass Tapir IR between them. The tiled NTT uses Kitsune Clang directly with `-ftapir=cuda`.

Kitsune leaves the preferred location of shared CUDA allocations unset. It initially requests a GPU mapping for indirect access. When a GPU kernel directly uses a buffer with a dedicated allocation, the runtime clears its mapping hint and prefetches the buffer to the GPU. This lets CUDA migrate pages on later accesses by either processor. Small objects that share pool pages retain the mapping because neighboring objects can be in use on the CPU. The prefetch smoke test checks both allocation paths using CUDA's range attributes.

## Source revisions

| Component | Source |
| --- | --- |
| Kitsune | Mario's `85ae825fa669`, three commits beyond `049800bad40e` |
| CUDA lowering | Recovered block-shared allocation lowering and the optional per-thread atomic reduction ablation |
| OpenCilk | `b644df129866` merged with upstream `dev/19.x` at `dab8218b770b` |
| OpenCilk Cheetah and productivity tools | `4af05d4cfb7f` and `fc0af2000b79`, included in the frontend subtree |
| Kitsune Cheetah | `be3a3e74d391`, fetched by Kitsune's build |

## Build and smoke tests

From the checkout root on x86-64 Linux, with CMake, Ninja, Clang, and CUDA installed, run

```sh
export CUDA_HOME=/usr/local/cuda-12.9
cilk-kitsune/build-artifact.sh
cilk-kitsune/smoke-artifact.sh
```

`BUILD_DIR` overrides the default `build/` directory. `CMAKE_BUILD_PARALLEL_LEVEL` controls the requested build parallelism. `LLVM_USE_LINKER` can select a host linker such as `lld`. The first build needs access to Kitsune's pinned Cheetah and runtime dependencies.

The smoke script checks CPU and GPU reductions against a serial reference, confirms an actual CUDA launch in the lowered IR, and exercises a GPU loop inside a CPU Cilk loop. It writes intermediate IR, logs and executables under `build/smoke/`. GPU tests use `sm_80` with embedded PTX by default. Set `CUDA_ARCH` to override the target.

For the batched NTT, use the paper benchmark checkout at `224660e8e24d`. In a clean checkout of `ingonyama-zk/icicle-snark` at `bf00385db190`, apply `cilk-kitsune/icicle-snark-bf00385-headers.patch` from this repository. It preserves the three modified Icicle headers used in the evaluation environment. Then run

```sh
ZERA_BENCHMARKS=/absolute/path/to/zera-paper-benchmarks \
ICICLE_INCLUDE=/absolute/path/to/icicle-snark/icicle/include \
cilk-kitsune/smoke-artifact.sh
```

This also checks two BN254 transforms each at sizes `2^12` and `2^21`. The larger size selects the tiled implementation. The script checks the benchmark's reported comparison result because the benchmark can return zero on a mismatch.

## Regression and parity checks

Run the following from the repository root after building the artifact. The
Clang and Tapir suites run on the host. The CUDA lowering tests use NVPTX and
the configured CUDA toolkit to inspect host IR, device IR and PTX. The launch
cache CTest uses a stub CUDA driver. These suites can run without a GPU.

```sh
cmake --build build/opencilk --target FileCheck
build/opencilk/bin/llvm-lit -sv opencilk-project/clang/test/Cilk
cmake --build build/kitsune --target kitsune-test-depends llc llvm-as llvm-dis
build/kitsune/bin/llvm-lit -sv llvm/test/Transforms/Tapir
cmake --build build/kitsune --target check-kitsune check-kitrt
```

On the RTX 5080, run the smoke script above with both NTT dependencies set.
It exercises separate compilation and a source-plus-object link. It checks
hierarchical fused, unfused and naive-atomic GPU reducers, a
4,097-element CPU and GPU prefix scan, and the CUDA launch in the nested and
debug-info transpose fixtures. Intermediate frontend and lowered IR are saved
in the run directory. The cache test can also run under [ThreadSanitizer](../kitsune/runtime/cuda/README.md#launch-cache-test).

Build the same per-row Shockwave source (`60dd693`) in separate build
directories with the reference and candidate compilers. Keep their runtime
libraries separate when comparing changes to kitrt. On the idle RTX 5080 run

```sh
python3 cilk-kitsune/compare-shockwave.py \
  /path/to/reference/cu-zksnark /path/to/candidate/cu-zksnark /path/to/new-results
```

The harness runs three warmups per build, then ten interleaved pairs at each
Table XI size. It requires a verified proof and saves stdout, stderr, runtime
resolution, digests, individual proof times, means and median paired ratios.
A shift beyond ±5% in either the mean or the median paired ratio returns a
review status. Inspect outliers and rerun the affected sizes in a fresh
directory with e.g. `--sizes 18 19`. Use `--reference-library-path` and
`--candidate-library-path` to override each executable's library search
path for the comparison. Check the recorded library paths before attributing a
timing difference to the compiler.
