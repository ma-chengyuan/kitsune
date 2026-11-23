#include <cstdio>

void update_a(int *out, int count);
void update_b(int *out, int count);

int main() {
  constexpr int count = 65536;
  constexpr int index = count / 2;
  int *out = new int[count]{};
  update_a(out, count);
  int first = out[index] - index;
  update_b(out, count);
  int second = out[index] - index;
  delete[] out;
  std::printf("Kernel offsets %d and %d (expected 17 and 29)\n", first, second);
  return first == 17 && second == 29 ? 0 : 1;
}
