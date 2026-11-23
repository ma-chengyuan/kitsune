// -*- C++ -*-
/*
 * OpenCilk productivity-tools revision
 * fc0af2000b7911fa1d8282ef7f910621c66e0921.
 *
 * SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2020 Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef INCLUDED_CILK_CILKSCALE_H
#define INCLUDED_CILK_CILKSCALE_H

#ifdef __cplusplus
#include <cstdint>
#else // __cplusplus
#include <stdint.h>
#endif // __cplusplus

typedef int64_t raw_duration_t;
typedef struct wsp_t {
  raw_duration_t work;
  raw_duration_t span;
  raw_duration_t bspan;
} wsp_t;

#ifdef __cplusplus

#include <fstream>
#include <iostream>

#define CILKSCALE_EXTERN_C extern "C"
#define CILKSCALE_NOTHROW noexcept

wsp_t &operator+=(wsp_t &lhs, const wsp_t &rhs) noexcept;
wsp_t &operator-=(wsp_t &lhs, const wsp_t &rhs) noexcept;
std::ostream &operator<<(std::ostream &os, const wsp_t &pt);
std::ofstream &operator<<(std::ofstream &os, const wsp_t &pt);

#ifndef __cilkscale__
// Default implementations when the program is not compiled with Cilkscale.
inline wsp_t &operator+=(wsp_t &lhs, const wsp_t &rhs) noexcept { return lhs; }

inline wsp_t &operator-=(wsp_t &lhs, const wsp_t &rhs) noexcept { return lhs; }

inline std::ostream &operator<<(std::ostream &os, const wsp_t &pt) {
  return os;
}

inline std::ofstream &operator<<(std::ofstream &os, const wsp_t &pt) {
  return os;
}
#endif // #ifndef __cilkscale__

#else // #ifdef __cplusplus

#define CILKSCALE_EXTERN_C
#define CILKSCALE_NOTHROW __attribute__((nothrow))

#endif // #ifdef __cplusplus

#ifdef __cplusplus
inline wsp_t operator+(wsp_t lhs, const wsp_t &rhs) noexcept {
  lhs += rhs;
  return lhs;
}
inline wsp_t operator-(wsp_t lhs, const wsp_t &rhs) noexcept {
  lhs -= rhs;
  return lhs;
}

extern "C" {
#endif // #ifdef __cplusplus
inline wsp_t wsp_zero(void) CILKSCALE_NOTHROW {
  wsp_t res = {0, 0, 0};
  return res;
}
#ifdef __cplusplus
}
#endif // #ifdef __cplusplus

#ifndef __cilkscale__

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Default implementations when the program is not compiled with Cilkscale.
static inline wsp_t wsp_getworkspan() CILKSCALE_NOTHROW {
  wsp_t res = {0, 0, 0};
  return res;
}

static inline wsp_t wsp_add(wsp_t lhs, wsp_t rhs) CILKSCALE_NOTHROW {
  wsp_t res = {0, 0, 0};
  return res;
}

static inline wsp_t wsp_sub(wsp_t lhs, wsp_t rhs) CILKSCALE_NOTHROW {
  wsp_t res = {0, 0, 0};
  return res;
}

static inline void wsp_dump(wsp_t wsp, const char *tag) { return; }

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#else // #ifndef __cilkscale__

CILKSCALE_EXTERN_C wsp_t wsp_getworkspan() CILKSCALE_NOTHROW;

CILKSCALE_EXTERN_C
wsp_t wsp_add(wsp_t lhs, wsp_t rhs) CILKSCALE_NOTHROW;

CILKSCALE_EXTERN_C
wsp_t wsp_sub(wsp_t lhs, wsp_t rhs) CILKSCALE_NOTHROW;

CILKSCALE_EXTERN_C
void wsp_dump(wsp_t wsp, const char *tag);

#endif // #ifndef __cilkscale__

#endif // INCLUDED_CILK_CILKSCALE_H
