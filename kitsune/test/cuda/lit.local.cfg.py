import os

if 'kitsune-cuda' not in config.available_features:
    config.unsupported = True
else:
    config.substitutions.append(
        ("%kitcuda_bc", os.path.join(config.clang_resource_dir, "lib", "kitcuda.bc"))
    )
