# Clang 19 artifact checkout

The repository root builds Kitsune. The `opencilk-project/` subtree builds the OpenCilk frontend and CPU runtime. The Shockwave path uses `cilk-kitsune/ock++` to pass Tapir IR between them. The tiled NTT uses Kitsune Clang directly with `-ftapir=cuda`.

Kitsune leaves the preferred location of shared CUDA allocations unset. CPU and GPU phases can use the same allocation throughout its lifetime. When the compiler requests a prefetch, the runtime uses `cuMemPrefetchAsync` to move the requested range toward the next processor. The prefetch smoke test checks both destinations and queries CUDA's preferred-location attribute after each request.

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
