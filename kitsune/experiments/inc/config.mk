# Kitsune support
kitsune_install_prefix:=/home/mrowkam/.local/kitsune

# Cuda support
kitsune_cuda_enable:="ON"
ifeq ($(kitsune_cuda_enable),"ON")
  $(info config: cuda target enabled.)
  KITSUNE_CUDA_ENABLED:=true
endif

# Hip support
kitsune_hip_enable:="OFF"
ifeq ($(kitsune_hip_enable),"ON")
  $(info config: hip target enabled.)
  KITSUNE_HIP_ENABLED:=true
  ROCM_PATH:=
endif

# Kokkos support
kitsune_kokkos_enable:="OFF"
ifeq ($(kitsune_kokkos_enable),"ON")
  $(info config: kokkos codegen enabled.)
  KITSUNE_KOKKOS_ENABLED:=true
endif

