// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -fopencilk -ftapir=none -fcxx-exceptions -fexceptions -verify -emit-llvm -o /dev/null
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -fopencilk -ftapir=none -fcxx-exceptions -fexceptions -verify -emit-llvm -o /dev/null -DTHROW

namespace std { class type_info; }

void identity(void *);
void second_identity(void *);
void reduce(void *, void *);

#ifndef THROW
using First = int _Hyperobject(identity, reduce) *; // expected-error {{RTTI for hyperobject types with reducer callbacks is unsupported}}
using Second = int _Hyperobject(second_identity, reduce) *; // expected-error {{RTTI for hyperobject types with reducer callbacks is unsupported}}

bool same_type() { return &typeid(First) == &typeid(Second); }

#else
using Exception = long _Hyperobject(identity, reduce) **; // expected-error {{RTTI for hyperobject types with reducer callbacks is unsupported}}
void raise(Exception e) { throw e; }
#endif
