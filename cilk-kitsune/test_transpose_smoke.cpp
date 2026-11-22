#include <algorithm>
#include <cilk/cilk.h>
#include <cstddef>
#include <vector>

// Inlining puts the helper's source locations in the caller's debug scope.
[[gnu::always_inline]] static inline void
transpose(const int *input, int *output, std::size_t rows, std::size_t cols) {
  cilk_for (std::size_t row = 0; row < rows; row += 32) {
    [[tapir::target("cuda"), tapir::grain_size(1)]]
    cilk_for (std::size_t col = 0; col < cols; col += 32) {
      for (auto i = row; i < std::min(row + 32, rows); ++i)
        for (auto j = col; j < std::min(col + 32, cols); ++j)
          output[j * rows + i] = input[i * cols + j];
    }
  }
}

// Keep the loop bounds dynamic through optimization.
[[gnu::noinline]] static void run(const int *input, int *output,
                                  std::size_t rows, std::size_t cols) {
  transpose(input, output, rows, cols);
}

int main(int argc, char **) {
  const std::size_t rows = 64 + argc, cols = 96 + argc;
  std::vector<int> input(rows * cols), output(rows * cols, -1);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<int>(i);

  run(input.data(), output.data(), rows, cols);
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      if (output[j * rows + i] != input[i * cols + j])
        return 1;
  return 0;
}
