#include <cstdio>
#include <new>

struct TaskError {};

[[clang::noinline]] int task(int value) {
  if (value < 0)
    throw TaskError{};
  return value;
}

[[clang::noinline]] int spawn_and_catch(int value) {
  int result = 0;
  try {
    _Cilk_spawn { result = task(value); }
    _Cilk_sync;
  } catch (const TaskError &) {
    return 17;
  }
  return result;
}

struct SumView {
  int value = 0;
  int *destroyed_value = nullptr;

  ~SumView() {
    if (destroyed_value)
      *destroyed_value = value;
  }
};

void identity(void *view) { new (view) SumView; }

void reduce(void *left, void *right) {
  auto &other = *static_cast<SumView *>(right);
  static_cast<SumView *>(left)->value += other.value;
  other.~SumView();
}

int main() {
  int destroyed_value = -1;
  int sum = 0;
  {
    SumView _Hyperobject(identity, reduce) reducer{0, &destroyed_value};
    _Cilk_for (int i = 0; i != 1024; ++i)
      reducer.value += i;
    sum = reducer.value;
  }

  int normal = spawn_and_catch(12);
  int caught = spawn_and_catch(-1);
  if (normal == 12 && caught == 17 && sum == 523776 && destroyed_value == sum)
    return 0;
  std::fprintf(stderr, "CPU Cilk smoke got normal=%d caught=%d sum=%d destroyed=%d\n",
               normal, caught, sum, destroyed_value);
  return 1;
}
