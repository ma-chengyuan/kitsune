; RUN: opt -passes=tapir-memops-replacement -verify-each -S %s | FileCheck %s

; The host path dispatches nonvolatile bulk operations through kitrt. Preserve
; both the full unsigned byte count and the volatile form of the intrinsics.
declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)

define void @bulk(ptr %dst, ptr %src, i32 %count32, i64 %count64, i8 %value) {
; CHECK-LABEL: define void @bulk(
; CHECK:      [[WIDE:%.*]] = zext i32 %count32 to i64
; CHECK-NEXT: call void @__kitcuda_memcpy(ptr %dst, ptr %src, i64 [[WIDE]])
; CHECK-NEXT: call void @__kitcuda_memmove(ptr %dst, ptr %src, i64 %count64)
; CHECK-NEXT: tail call void @__kitcuda_memset(ptr %dst, i8 %value, i64 %count64)
; CHECK-NEXT: ret void
  call void @llvm.memcpy.p0.p0.i32(ptr %dst, ptr %src, i32 %count32, i1 false)
  call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 %count64, i1 false)
  tail call void @llvm.memset.p0.i64(ptr %dst, i8 %value, i64 %count64, i1 false)
  ret void
}

define void @volatile_bulk(ptr %dst, ptr %src, i32 %count32, i64 %count64) {
; CHECK-LABEL: define void @volatile_bulk(
; CHECK:      call void @llvm.memcpy.p0.p0.i32(ptr %dst, ptr %src, i32 %count32, i1 true)
; CHECK-NEXT: call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 %count64, i1 true)
; CHECK-NEXT: call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 %count64, i1 true)
; CHECK-NEXT: ret void
  call void @llvm.memcpy.p0.p0.i32(ptr %dst, ptr %src, i32 %count32, i1 true)
  call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 %count64, i1 true)
  call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 %count64, i1 true)
  ret void
}
