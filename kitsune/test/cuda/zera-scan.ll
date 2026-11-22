; REQUIRES: nvptx-registered-target
; RUN: rm -rf "%t" && mkdir -p "%t"
; RUN: cd "%t" && "opt" -passes='tapir-lowering<O2>' -tapir-target=cuda -kitcuda-runtime-bc-path="%kitcuda_bc" -cuabi-arch=sm_80 -cuabi-keep-files -verify-each -S "%s" -o host.ll
; RUN: "FileCheck" "%s" --check-prefix=HOST < "%t/host.ll"
; RUN: cd "%t" && "opt" -passes=verify -disable-output *-ptx.ll
; RUN: cd "%t" && "FileCheck" "%s" --check-prefix=DEVICE --implicit-check-not='call ptr @__kitcuda_get_scan_view' < *-ptx.ll

; The runtime allocates scratch for the same kernel and trip count as the
; launch, and frees the returned buffers after the stream completes.
; HOST-LABEL: define void @scan(
; HOST: call void @__kitcuda_alloc_scanner(ptr{{[^,]*}} [[FATBIN:@[^,]+]], ptr{{[^,]*}} [[KERNEL:@[^,]+]], i64 %n, ptr{{[^,]*}} [[MIX:%[^,]+]], i64 4, ptr{{[^,]*}} [[AGG_SLOT:%[^,]+]], ptr{{[^,]*}} [[PREFIX_SLOT:%[^,]+]], ptr{{[^,]*}} [[STATE_SLOT:%[^)]+]])
; HOST-DAG: [[AGG:%[^ ]+]] = load ptr, ptr [[AGG_SLOT]],
; HOST-DAG: [[PREFIX:%[^ ]+]] = load ptr, ptr [[PREFIX_SLOT]],
; HOST-DAG: [[STATE:%[^ ]+]] = load ptr, ptr [[STATE_SLOT]],
; HOST-DAG: store ptr [[AGG]], ptr
; HOST-DAG: store ptr [[PREFIX]], ptr
; HOST-DAG: store ptr [[STATE]], ptr
; HOST: [[STREAM:%[^ ]+]] = call ptr @__kitcuda_launch_kernel(ptr{{[^,]*}} [[FATBIN]], ptr{{[^,]*}} [[KERNEL]], ptr {{[^,]+}}, i64 %n, i32 {{[0-9]+}}, ptr{{[^,]*}} [[MIX]],
; HOST: call void @__kitcuda_sync_thread_stream(ptr [[STREAM]])
; HOST: call void @__kitcuda_free_scanner(ptr [[AGG]], ptr [[PREFIX]], ptr [[STATE]])

; The scan uses one shared i32 per warp and combines the initialized view
; with the supplied identity and merge functions.
; DEVICE: [[SHARED:@[^ ]+]] = internal addrspace(3) global [32 x i32] undef
; DEVICE-LABEL: define void @_cuabi_kern_scan_
; DEVICE: call void @zero(ptr [[VIEW:%[^)]+]])
; DEVICE: call void @__kitcuda_scan(ptr [[VIEW]], {{.*}}ptr addrspacecast (ptr addrspace(3) [[SHARED]] to ptr), {{.*}}i64 4, {{.*}}ptr @zero, ptr @add)

target triple = "x86_64-unknown-linux-gnu"

declare token @llvm.syncregion.start()
declare ptr @__kitcuda_null()
declare ptr @__kitcuda_get_scan_view(i64, ptr, ptr, ptr, ptr, ptr, ptr)

; The three placeholder arguments are replaced with runtime scratch buffers.
define void @scan(ptr %input, ptr %result, i64 %n) {
entry:
  %region = call token @llvm.syncregion.start()
  %aggregate = call ptr @__kitcuda_null()
  %prefix = call ptr @__kitcuda_null()
  %state = call ptr @__kitcuda_null()
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  detach within %region, label %body, label %latch

body:
  %view = call ptr @__kitcuda_get_scan_view(i64 4, ptr %aggregate, ptr %prefix, ptr %state, ptr %result, ptr @zero, ptr @add)
  %element = getelementptr inbounds i32, ptr %input, i64 %i
  %value = load i32, ptr %element, align 4
  %old = load i32, ptr %view, align 4
  %new = add i32 %old, %value
  store i32 %new, ptr %view, align 4
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

define internal void @zero(ptr %p) {
  store i32 0, ptr %p, align 4
  ret void
}

define internal void @add(ptr %left, ptr %right) {
  %a = load i32, ptr %left, align 4
  %b = load i32, ptr %right, align 4
  %sum = add i32 %a, %b
  store i32 %sum, ptr %left, align 4
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"tapir.loop.grainsize", i32 1}
!3 = !{!"tapir.loop.target", i32 2}
!4 = !{!"tapir.loop.deferred_sync", i1 true}
