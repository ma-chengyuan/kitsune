; REQUIRES: nvptx-registered-target
; RUN: rm -rf "%t" && mkdir -p "%t"
; RUN: cd "%t" && "opt" -passes='tapir-lowering<O2>' -tapir-target=cuda -use-kitcuda-runtime-bc=false -cuabi-arch=sm_80 -verify-each -S "%s" -o host.ll
; RUN: "FileCheck" "%s" < "%t/host.ll"

; The default loop waits on its launch stream before consuming the result.
; CHECK-LABEL: define void @sync_on_launch(
; CHECK: [[STREAM:%[0-9a-zA-Z_.]+]] = call ptr @__kitcuda_launch_kernel(
; CHECK: call void @__kitcuda_sync_thread_stream(ptr [[STREAM]])
; CHECK: call void @consume(ptr %out)

; The deferred loop lets the host do work before its explicit stream wait.
; CHECK-LABEL: define void @sync_explicitly(
; CHECK: call ptr @__kitcuda_launch_kernel(
; CHECK-NOT: @__kitcuda_sync_thread_stream
; CHECK: call void @host_work()
; CHECK-NEXT: call void @__kitcuda_sync_current_stream()
; CHECK-NEXT: call void @consume(ptr %out)
; CHECK-NOT: @__kitcuda_sync_thread_stream
; CHECK: ret void

target triple = "x86_64-unknown-linux-gnu"

declare token @llvm.syncregion.start()
declare void @__kitcuda_sync_current_stream()
declare void @host_work()
declare void @consume(ptr)

define void @sync_on_launch(ptr %out, i64 %n) {
entry:
  %region = call token @llvm.syncregion.start()
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  detach within %region, label %body, label %latch

body:
  %dst = getelementptr inbounds i64, ptr %out, i64 %i
  store i64 %i, ptr %dst, align 8
  reattach within %region, label %latch

latch:
  %next = add nuw i64 %i, 1
  %last = icmp eq i64 %next, %n
  br i1 %last, label %sync, label %loop, !llvm.loop !0

sync:
  sync within %region, label %done

done:
  call void @consume(ptr %out)
  ret void
}

define void @sync_explicitly(ptr %out, i64 %n) {
entry:
  %region = call token @llvm.syncregion.start()
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %done, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %latch ]
  detach within %region, label %body, label %latch

body:
  %dst = getelementptr inbounds i64, ptr %out, i64 %i
  store i64 %i, ptr %dst, align 8
  reattach within %region, label %latch

latch:
  %next = add nuw i64 %i, 1
  %last = icmp eq i64 %next, %n
  br i1 %last, label %sync, label %loop, !llvm.loop !4

sync:
  sync within %region, label %done

done:
  call void @host_work()
  call void @__kitcuda_sync_current_stream()
  call void @consume(ptr %out)
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"tapir.loop.spawn.strategy", i32 1}
!2 = !{!"tapir.loop.grainsize", i32 1}
!3 = !{!"tapir.loop.target", i32 2}
!4 = distinct !{!4, !1, !2, !3, !5}
!5 = !{!"tapir.loop.deferred_sync", i1 true}
