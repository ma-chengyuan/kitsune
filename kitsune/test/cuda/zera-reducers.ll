; REQUIRES: nvptx-registered-target
; DEFINE: %{lower} = "opt" -passes='tapir-lowering<O2>' -tapir-target=cuda -use-kitcuda-runtime-bc=false -cuabi-arch=sm_80 -cuabi-keep-files -verify-each -S "%s" -o host.ll
; DEFINE: %{check-device} = "FileCheck" "%s" --implicit-check-not='call ptr @llvm.hyper.lookup'
; RUN: rm -rf "%t"
; RUN: mkdir -p "%t/fused" "%t/unfused" "%t/naive"
; RUN: cd "%t/fused" && %{lower}
; RUN: cd "%t/unfused" && %{lower} -cuabi-dont-fuse-reducers
; RUN: cd "%t/naive" && %{lower} -cuabi-naive-atomics
; RUN: "FileCheck" "%s" --check-prefix=HOST < "%t/fused/host.ll"
; RUN: "FileCheck" "%s" --check-prefix=HOST < "%t/unfused/host.ll"
; RUN: "FileCheck" "%s" --check-prefix=HOST < "%t/naive/host.ll"
; RUN: cd "%t/fused" && "opt" -passes=verify -disable-output *-ptx.ll
; RUN: cd "%t/unfused" && "opt" -passes=verify -disable-output *-ptx.ll
; RUN: cd "%t/naive" && "opt" -passes=verify -disable-output *-ptx.ll
; RUN: cd "%t/fused" && %{check-device} --check-prefixes=DEVICE,TREE < *-ptx.ll
; RUN: cd "%t/unfused" && %{check-device} --check-prefixes=DEVICE,TREE < *-ptx.ll
; RUN: cd "%t/naive" && %{check-device} --check-prefix=DEVICE --implicit-check-not='call i32 @llvm.nvvm.shfl' --implicit-check-not='call void @llvm.nvvm.bar.sync' < *-ptx.ll
; RUN: cd "%t/fused" && "FileCheck" "%s" --check-prefix=FUSED --implicit-check-not='call void @llvm.nvvm.bar.sync' < *-ptx.ll
; RUN: cd "%t/unfused" && "FileCheck" "%s" --check-prefix=UNFUSED --implicit-check-not='call void @llvm.nvvm.bar.sync' < *-ptx.ll

; Reduction forces synchronization even with the deferred-sync hint. The
; staging buffer includes the aggregate mutex and padding for the scalar.
; HOST-LABEL: define void @reduce(
; HOST: call ptr @__kitcuda_mem_alloc_and_copy_to_device(ptr{{[^,]*}} [[STAGING:%[0-9a-zA-Z_.]+]], i64 32,
; HOST: [[STREAM:%[0-9a-zA-Z_.]+]] = call ptr @__kitcuda_launch_kernel(
; HOST: call void @__kitcuda_sync_thread_stream(ptr [[STREAM]])
; HOST: call ptr @__kitcuda_mem_copy_and_free_from_device(ptr{{[^,]*}} [[STAGING]], i64 32, ptr {{[^,]+}}, ptr [[STREAM]])
; HOST-DAG: call void @merge_pair(
; HOST-DAG: call void @add_word(
; HOST: ret void

; Both reducers update global memory atomically. Hierarchical modes also
; shuffle within warps and use one block barrier per reduction tree.
; DEVICE-LABEL: define void @_cuabi_kern_reduce_
; TREE: call i32 @llvm.nvvm.shfl.sync.down.i32(
; DEVICE-DAG: cmpxchg ptr %word{{[^,]*}}, i64
; DEVICE-DAG: [[MUTEX:%[^ ]+]] = getelementptr i32, ptr %pair{{[^,]*}}, i32 4
; DEVICE-DAG: cmpxchg ptr [[MUTEX]], i32 0, i32 1
; DEVICE: }

; FUSED: call void @llvm.nvvm.bar.sync(i32 0)
; UNFUSED: call void @llvm.nvvm.bar.sync(i32 0)
; UNFUSED: call void @llvm.nvvm.bar.sync(i32 0)

target triple = "x86_64-unknown-linux-gnu"

declare token @llvm.syncregion.start()
declare ptr @llvm.hyper.lookup.i64(ptr, i64, ptr, ptr)

; Use a scalar and an aggregate to exercise both forms of the final atomic
; update. The aggregate reserves a mutex after its two i64 fields.
define void @reduce(ptr %input, i64 %n, ptr %word, ptr %pair) {
entry:
  %region = call token @llvm.syncregion.start()
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  detach within %region, label %body, label %latch

body:
  %element = getelementptr inbounds i64, ptr %input, i64 %i
  %value = load i64, ptr %element, align 8
  %word.view = call ptr @llvm.hyper.lookup.i64(ptr %word, i64 8, ptr @zero_word, ptr @add_word)
  %pair.view = call ptr @llvm.hyper.lookup.i64(ptr %pair, i64 16, ptr @zero_pair, ptr @merge_pair)
  %word.old = load i64, ptr %word.view, align 8
  %word.new = add i64 %word.old, %value
  store i64 %word.new, ptr %word.view, align 8
  %sum.old = load i64, ptr %pair.view, align 8
  %sum.new = add i64 %sum.old, %i
  store i64 %sum.new, ptr %pair.view, align 8
  %parity = getelementptr inbounds i64, ptr %pair.view, i64 1
  %parity.old = load i64, ptr %parity, align 8
  %parity.new = xor i64 %parity.old, %value
  store i64 %parity.new, ptr %parity, align 8
  reattach within %region, label %latch

latch:
  %next = add nuw i64 %i, 1
  %last = icmp eq i64 %next, %n
  br i1 %last, label %sync, label %loop, !llvm.loop !0

sync:
  sync within %region, label %done

done:
  ret void
}

define internal void @zero_word(ptr %p) {
  store i64 0, ptr %p, align 8
  ret void
}

define internal void @add_word(ptr %left, ptr %right) {
  %a = load i64, ptr %left, align 8
  %b = load i64, ptr %right, align 8
  %sum = add i64 %a, %b
  store i64 %sum, ptr %left, align 8
  ret void
}

define internal void @zero_pair(ptr %p) {
  store i64 0, ptr %p, align 8
  %parity = getelementptr inbounds i64, ptr %p, i64 1
  store i64 0, ptr %parity, align 8
  ret void
}

define internal void @merge_pair(ptr %left, ptr %right) {
  call void @add_word(ptr %left, ptr %right)
  %pa = getelementptr inbounds i64, ptr %left, i64 1
  %pb = getelementptr inbounds i64, ptr %right, i64 1
  %a = load i64, ptr %pa, align 8
  %b = load i64, ptr %pb, align 8
  %parity = xor i64 %a, %b
  store i64 %parity, ptr %pa, align 8
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"tapir.loop.grainsize", i32 1}
!3 = !{!"tapir.loop.target", i32 2}
!4 = !{!"tapir.loop.deferred_sync", i1 true}
