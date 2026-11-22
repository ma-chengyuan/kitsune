#include <cilk/cilk.h>
#include <cstdint>
#include <cstdio>

using Word = std::uint64_t;
struct Pair {
  Word sum;
  Word parity;
};

static void zero_word(void *p) { *static_cast<Word *>(p) = 0; }
static void add_word(void *left, void *right) {
  *static_cast<Word *>(left) += *static_cast<Word *>(right);
}
static void zero_pair(void *p) { *static_cast<Pair *>(p) = {}; }
static void merge_pair(void *left, void *right) {
  auto &a = *static_cast<Pair *>(left);
  const auto &b = *static_cast<Pair *>(right);
  a.sum += b.sum;
  a.parity ^= b.parity;
}

static bool check(Word n) {
  Word cilk_reducer(zero_word, add_word) total = 19;
  Pair cilk_reducer(zero_pair, merge_pair) pair = {23, 31};
#ifdef ZERA_SMOKE_CUDA
  [[tapir::target("cuda")]]
#endif
  cilk_for (Word i = 0; i < n; ++i) {
    total += (i << 36) + 1;
    pair.sum += i * i;
    pair.parity ^= (i + 1) * 0x9e3779b97f4a7c15ULL;
  }
  Word expected_total = 19;
  Pair expected_pair{23, 31};
  for (Word i = 0; i < n; ++i) {
    expected_total += (i << 36) + 1;
    expected_pair.sum += i * i;
    expected_pair.parity ^= (i + 1) * 0x9e3779b97f4a7c15ULL;
  }
  return total == expected_total && pair.sum == expected_pair.sum &&
         pair.parity == expected_pair.parity;
}

int main() {
  const Word sizes[] = {0, 1, 31, 32, 33, 255, 256, 257, 65537};
  for (Word n : sizes) {
    if (!check(n)) {
      std::fprintf(stderr, "Reducer mismatch at n=%llu\n",
                   static_cast<unsigned long long>(n));
      return 1;
    }
  }
  std::puts("Reducer smoke test passed");
}
