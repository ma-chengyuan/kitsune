
#ifndef CLANGD
#include <cilk/cilk.h>
#else
#define cilk_for for
#define cilk_reducer(id, merge)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>

void id_sum(void *data) { *reinterpret_cast<size_t *>(data) = 0; }

void merge_sum(void *data, void *other) {
    *reinterpret_cast<size_t *>(data) += *reinterpret_cast<size_t *>(other);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return 1;
    }
    int N = strtol(argv[1], NULL, 10);
    size_t cilk_reducer(id_sum, merge_sum) sum = 0;
    cilk_for (size_t j = 0; j < N; j++) {
        [[tapir::target("cuda"), tapir::grain_size(1)]] cilk_for (size_t i = 0; i < N; i++) {
            sum += i | j;
        }
    }
    size_t ref = 0;
    for (size_t j = 0; j < N; j++) {
        for (size_t i = 0; i < N; i++) {
            ref += i | j;
        }
    }
    printf("Sum of first %d numbers: %zu vs %zu\n", N, sum, ref);
    return 0;
}
