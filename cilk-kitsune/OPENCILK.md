# External OpenCilk frontend

The hybrid driver uses OpenCilk Clang 19.1.7 to emit unlowered Tapir IR with
`-fopencilk -fopencilk-kitsune -ftapir=none`. Kitsune Clang 19.1.2 and its
LLVM tools lower that IR and link the hybrid executable. The adapter takes
the external OpenCilk build or install prefix as `OPENCILK_PATH`. For Stage 2
comparisons, use the `opencilk/` prefix in the frozen Stage 1 snapshot.
[Adapter and build instructions](ARTIFACT.md)

## Source and runtime revisions

| Component | Revision and role |
| --- | --- |
| OpenCilk frontend merge | `f1e93f1b168f00a2008c9611612f0a42dcc2a197` combines the [Zera frontend fork](https://github.com/ma-chengyuan/opencilk-project/commit/b644df12986682b8dda843542a65839916798245) and [OpenCilk `dev/19.x`](https://github.com/OpenCilk/opencilk-project/commit/dab8218b770b293e29af22814dbc55ed5bdaedf0). The dual-tree commit `ff73da245ba8` imports this merge. |
| Reducer template fix | `ab069e19a53bb532c0d38b355037356044a34aa1` changes `clang/lib/Sema/SemaExpr.cpp` and adds `clang/test/Cilk/hyper-template-single-lookup.cpp`. It keeps an instantiated reducer lvalue tied to the first lookup's view. |
| Stage 1 frontend tests | `eeef165eaffe76e51af76a14266dfdfa691851c7` updates the checks in `cilkfor-bounds.cpp` and `tapirloopattrs.c`. `c894242a1a8ad1f53110074c3d66d081854cca68` adds `kitsune-loop-errors.cpp` and `kitsune-loop-targets.cpp` for attribute diagnostics and CPU/CUDA metadata at the IR handoff. |
| Standalone OpenCilk runtime | [Cheetah `4af05d4cfb7f693ed9fa05fa98126222204ab568`](https://github.com/OpenCilk/cheetah/commit/4af05d4cfb7f693ed9fa05fa98126222204ab568) and [cilktools `fc0af2000b7911fa1d8282ef7f910621c66e0921`](https://github.com/OpenCilk/productivity-tools/commit/fc0af2000b7911fa1d8282ef7f910621c66e0921) live in `cheetah/` and `cilktools/` under the external frontend source. The dual tree imported them in `06e1d74c278d` and `4085c34c67ab`. |
| Kitsune hybrid runtime | [Cheetah `be3a3e74d391913d6b58e60f2e65e48f084703c9`](https://github.com/neboat/cheetah-1/commit/be3a3e74d391913d6b58e60f2e65e48f084703c9) is pinned by `ff23e023ed1b` in [Kitsune's CMake build](../kitsune/CMakeLists.txt). The hybrid path uses this build's Cheetah ABI bitcode and libraries, along with `libkitrt.so` and `kitcuda.bc` from Kitsune. |

The two Cheetah revisions have different thread-local storage layouts. Keep
each `libopencilk-abi.bc` with the Cheetah libraries built from the same
revision. In the snapshot the mixed proof resolves Cheetah and kitrt under
`kitsune/lib/clang/19/lib/` and CUDA on the host. See its
`prover-libraries.txt` for the resolved paths.

The snapshot's OpenCilk `clang --version` reports parent-repository revision
`7311372f24553f930cd5af075fc0ae867cff157c`. Its OpenCilk subtree is identical
to the measured artifact revision `6841e22cf0c89ce5f47306b9d2ba21a4b51dd654`.
Check the snapshot's compiler and runtime file identities with
`sha256sum -c SHA256SUMS` from the snapshot root. The OpenCilk compiler is
`opencilk/bin/clang-19`, with SHA-256
`55b132f4880a0d79941d590028c199597e0c56f631fdd160f294ca4528dc4ee2`.

## Prepare the external source

If you have the frozen dual-tree checkout, export the Stage 1 frontend and
tests into an empty directory outside the Kitsune checkout. Set `REFERENCE`
to the dual-tree checkout and `OPENCILK_SOURCE` to the external directory.

```sh
mkdir -p "$OPENCILK_SOURCE"
git -C "$REFERENCE" archive fc6a58b8d18c81969759cbf9e5d39489b95f30c0:opencilk-project \
  | tar -x -C "$OPENCILK_SOURCE"
```

You can also reconstruct the same source from the pinned public repositories.
Run these commands from the clean Kitsune checkout with `OPENCILK_SOURCE`
set to an empty directory outside it. The
[frontend patch](patches/opencilk-frontend.patch) reproduces the merged frontend
and the reducer-template fix. The [test patch](patches/opencilk-stage1-tests.patch)
adds the Stage 1 characterization changes.

```sh
patch_dir="$PWD/cilk-kitsune/patches"
git init "$OPENCILK_SOURCE"

git -C "$OPENCILK_SOURCE" fetch --depth 1 https://github.com/OpenCilk/opencilk-project.git \
  dab8218b770b293e29af22814dbc55ed5bdaedf0
git -C "$OPENCILK_SOURCE" checkout --detach FETCH_HEAD
git -C "$OPENCILK_SOURCE" apply --index "$patch_dir/opencilk-frontend.patch"

git -C "$OPENCILK_SOURCE" fetch --depth 1 https://github.com/OpenCilk/cheetah.git \
  4af05d4cfb7f693ed9fa05fa98126222204ab568
git -C "$OPENCILK_SOURCE" read-tree -u --prefix=cheetah/ FETCH_HEAD
git -C "$OPENCILK_SOURCE" fetch --depth 1 https://github.com/OpenCilk/productivity-tools.git \
  fc0af2000b7911fa1d8282ef7f910621c66e0921
git -C "$OPENCILK_SOURCE" read-tree -u --prefix=cilktools/ FETCH_HEAD

test "$(git -C "$OPENCILK_SOURCE" write-tree)" = 80fa3e6be62ad8c787fef455a44c98493c3695ee
git -C "$OPENCILK_SOURCE" apply --index "$patch_dir/opencilk-stage1-tests.patch"
test "$(git -C "$OPENCILK_SOURCE" write-tree)" = 37c7cfac7490f924a2a7ff88ce30e7a3e7300733
```

The first tree ID is the artifact OpenCilk subtree. The second includes the
Stage 1 tests and is the source of the characterization suite below.

## Run frontend characterization

Use OpenCilk's Clang lit configuration on an x86-64 Linux host. With
`OPENCILK_PATH` set to the frozen snapshot's `opencilk/` prefix, set
`OPENCILK_TEST_BUILD` to a separate build directory for the test harness and
host tools. The snapshot itself stays read-only for this procedure.

```sh
cmake -S "$OPENCILK_SOURCE/llvm" -B "$OPENCILK_TEST_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS=clang -DLLVM_TARGETS_TO_BUILD=X86
cmake --build "$OPENCILK_TEST_BUILD" --target FileCheck count not llvm-config
CLANG="$OPENCILK_PATH/bin/clang" "$OPENCILK_TEST_BUILD/bin/llvm-lit" \
  -sv "$OPENCILK_SOURCE/clang/test/Cilk"
```

The suite reports 107 passes with frozen OpenCilk. It exercises Cilk syntax,
diagnostics, exception handling, reducer lookup, and the mixed-target IR
handoff. The prebuilt compiler, its resource headers, and the standalone
LLVM test tools are sufficient for this host-only gate.
