#include <cstdio>
#include <cuda_runtime_api.h>

unsigned char *allocate_buffer();
void release_buffer(unsigned char *buffer);

int main() {
  unsigned char *buffer = allocate_buffer();
  cudaPointerAttributes attributes{};
  cudaError_t status = cudaPointerGetAttributes(&attributes, buffer);
  release_buffer(buffer);

  if (status != cudaSuccess) {
    std::fprintf(stderr, "Could not query shared library allocation: %s\n",
                 cudaGetErrorString(status));
    return 1;
  }
  if (attributes.type != cudaMemoryTypeManaged) {
    std::fprintf(stderr, "Shared library allocation has CUDA memory type %d\n",
                 attributes.type);
    return 1;
  }
  return 0;
}
