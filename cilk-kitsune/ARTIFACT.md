# Clang 19 artifact checkout

The Kitsune root builds Clang 19 with OpenCilk language support, the CPU and CUDA lowering passes, and their runtimes. Shockwave uses the root compiler through `cilk-kitsune/ock++`, which stages frontend IR emission, Tapir lowering and linking. Standalone CPU Cilk and the tiled CUDA NTT use the root driver directly. All three routes use resources from the same build.

Kitsune leaves the preferred location of shared CUDA allocations unset. It initially requests a GPU mapping for indirect access. When a GPU kernel directly uses a buffer with a dedicated allocation, the runtime clears its mapping hint and prefetches the buffer to the GPU. This lets CUDA migrate pages on later accesses by either processor. Small objects that share pool pages retain the mapping because neighboring objects can be in use on the CPU. The prefetch smoke test checks both allocation paths using CUDA's range attributes.

## Source revisions

| Component | Source |
| --- | --- |
| Kitsune | Mario's `85ae825fa669`, three commits beyond `049800bad40e` |
| CUDA lowering | Recovered block-shared allocation lowering and the optional per-thread atomic reduction ablation |
| OpenCilk language port | [Reference frontend and runtime revisions](OPENCILK.md) |
| Kitsune Cheetah | `be3a3e74d391`, fetched by Kitsune's build |

This branch descends from the Kitsune integration through `85ae825fa669`.
It carries the recovered CUDA lowering from `4002d71d3b05` and
`e2da16cc57cb`. The 17 replayed root commits through `232b79483332` record
their original revisions in `Source-commit` trailers.

The root Clang language port uses the OpenCilk subtree from the Stage 1
reference at `fc6a58b8d18c`, including its reducer-template fix and Cilk
characterization suite. The original Kitsune loop and spawn syntax remains
available alongside the OpenCilk constructs.

## Build and smoke tests

On x86-64 Linux the build needs CMake, Ninja, a host Clang and CUDA. Run from the Kitsune checkout root.

```sh
unset OPENCILK_SOURCE OPENCILK_PATH
export CUDA_HOME=/usr/local/cuda-12.9
export KITSUNE_PATH="$PWD/build/kitsune"
cilk-kitsune/build-artifact.sh
cilk-kitsune/smoke-artifact.sh
```

`BUILD_DIR` defaults to the checkout's `build/` directory and `KITSUNE_PATH` defaults to `$BUILD_DIR/kitsune`. The wrapper's frontend and backend both use this prefix. The optional `OPENCILK_PATH` override selects an external frontend for comparisons with the two-compiler reference. `CMAKE_BUILD_PARALLEL_LEVEL` controls build parallelism and `LLVM_USE_LINKER` selects the host linker. The first build fetches Kitsune's pinned Cheetah and runtime dependencies.

The smoke script checks CPU and GPU reductions against a serial reference, confirms an actual CUDA launch in the lowered IR, and exercises a GPU loop inside a CPU Cilk loop. It writes intermediate IR, logs and executables under `$BUILD_DIR/smoke/`. GPU tests use `sm_80` with embedded PTX by default. Set `CUDA_ARCH` to override the target.

For the batched NTT, use the paper benchmark checkout at `224660e8e24d`. In a clean checkout of `ingonyama-zk/icicle-snark` at `bf00385db190`, apply `cilk-kitsune/icicle-snark-bf00385-headers.patch` from this repository. It preserves the three modified Icicle headers used in the evaluation environment. Then run

```sh
ZERA_BENCHMARKS=/absolute/path/to/zera-paper-benchmarks \
ICICLE_INCLUDE=/absolute/path/to/icicle-snark/icicle/include \
cilk-kitsune/smoke-artifact.sh
```

This also checks two BN254 transforms each at sizes `2^12` and `2^21`. The larger size selects the tiled implementation. The script checks the benchmark's reported comparison result because the benchmark can return zero on a mismatch.

## Regression and parity checks

Run the following from the Kitsune repository root after building the artifact.
The Tapir suite runs on the host. The CUDA lowering tests use NVPTX and
the configured CUDA toolkit to inspect host IR, device IR and PTX. The launch
cache CTest uses a stub CUDA driver. These suites can run without a GPU.

```sh
cmake --build "$KITSUNE_PATH" --target kitsune-test-depends FileCheck count not llvm-config llc llvm-as llvm-dis
"$KITSUNE_PATH/bin/llvm-lit" -sv clang/test/Cilk
"$KITSUNE_PATH/bin/llvm-lit" -sv llvm/test/Transforms/Tapir
cmake --build "$KITSUNE_PATH" --target check-kitsune check-kitrt
```

On the RTX 5080, run the smoke script above with both NTT dependencies set.
It exercises separate compilation and a source-plus-object link. It checks
CPU exception and reducer cleanup at `-O0` and `-O2`, a CPU-to-GPU reducer
across the implicit sync, fused, unfused and naive-atomic GPU reductions, a
4,097-element CPU and GPU prefix scan, and the CUDA launch in the nested and
debug-info transpose fixtures. Intermediate frontend and lowered IR are saved
in the run directory. The cache test can also run under [ThreadSanitizer](../kitsune/runtime/cuda/README.md#launch-cache-test).

Build the same per-row Shockwave source (`60dd693`) in separate build
directories with the reference and candidate compilers. For the candidate,
keep the `KITSUNE_PATH` and `CUDA_HOME` set above and point
`ZERA_COMPILER` to the Kitsune checkout. Run from the Shockwave source directory

```sh
ZERA_COMPILER=/path/to/kitsune-checkout
launcher="cmake;-E;env;OPENCILK_PATH=$KITSUNE_PATH;KITSUNE_PATH=$KITSUNE_PATH;CUDA_HOME=$CUDA_HOME;PATH=$CUDA_HOME/bin:$PATH;$ZERA_COMPILER/cilk-kitsune/ock++"
cmake --preset artifact-kitsune -B build-stage3 \
  -DCMAKE_CXX_COMPILER="$KITSUNE_PATH/bin/clang++" \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$launcher" \
  -DCMAKE_CXX_LINKER_LAUNCHER="$launcher"
cmake --build build-stage3 --target cu-zksnark
```

Keep the builds' runtime libraries separate when comparing changes to kitrt.
On the idle RTX 5080 run

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

The hardware smokes require a named CUDA kernel launch in each GPU program’s runtime log. The Shockwave timing harness verifies a separate untimed proof with launch tracing for each binary before its warmups and timed pairs.
