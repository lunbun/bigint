/*
 * Copyright (c) 2025 lunbun
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifndef BIGINT_H
#define BIGINT_H

#ifndef BIGINT_INLINE
#ifdef __GNUC__
#define BIGINT_INLINE inline __attribute__((always_inline))
#else
#define BIGINT_INLINE inline
#endif // __GNUC__
#endif // BIGINT_INLINE

#ifdef __GNUC__
#define BIGINT_LIKELY(x) __builtin_expect(!!(x), 1)
#define BIGINT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define BIGINT_LIKELY(x) (x)
#define BIGINT_UNLIKELY(x) (x)
#endif // __GNUC__

#define BIGINT_ASSERT(x) assert(x)

#ifdef __clang__
#define BIGINT_ASSUME(x) __builtin_assume(x)
#elifdef __GNUC__
// clang-format off
#define BIGINT_ASSUME(x) do { if (!(x)) __builtin_unreachable(); } while (0)
// clang-format on
#else
#define BIGINT_ASSUME(x) ((void)0)
#endif // __clang__

#ifdef NDEBUG
#define BIGINT_ASSUME_ASSERT(x) BIGINT_ASSUME(x)
#else
#define BIGINT_ASSUME_ASSERT(x) BIGINT_ASSERT(x)
#endif // NDEBUG

#define BIGINT_UNIMPL()                                                        \
  do {                                                                         \
    std::cerr << "Unimplemented: " << __FILE__ << ":" << __LINE__              \
              << std::endl;                                                    \
    std::abort();                                                              \
  } while (0)

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <string>

class bigint_t {
private:
  // Some conventions about this bigint implementation:
  // - The limbs are stored in little-endian order.
  // - "true" in the sign-bit corresponds to negative numbers.
  // - Every bigint will always have at least one limb--including the number
  //    zero.
  // - Negative zero is not allowed.
  // - Sign-bit is stored as flags in the most significant bit of the size_
  //    field. See the comment in size_ for more details.
  // - Local-buffer optimization shall always be in use if the size of the
  //    number is less than the size of the local buffer.

  // A limb refers to a single machine word used to store a part of the number.
  // The term is borrowed from the GMP library
  // (https://gmplib.org/manual/Nomenclature-and-Types).
  using LimbT = uint64_t;

  static constexpr size_t kSizeBitCount = sizeof(size_t) * 8;

  static constexpr size_t kLocalBufSize = 16 / sizeof(LimbT);
  static constexpr bool kLocalBufEnabled = kLocalBufSize > 0;

  static constexpr size_t kSignBit = 1 << (kSizeBitCount - 1);

  // kFlagBitsMask is used to mask out any flag bits in the size_ field.
  // Currently, only the sign bit is used, but this is here in case any other
  // flag bits are needed in the future.
  static constexpr size_t kFlagBitsMask = kSignBit;

  BIGINT_INLINE static constexpr bool AddOverflow(LimbT a, LimbT b,
                                                  LimbT &res) {
#ifdef __GNUC__
    return __builtin_add_overflow(a, b, &res);
#else
    res = a + b;
    return (res < a) || (res < b);
#endif // __GNUC__
  }

public:
  bigint_t() : bigint_t(0) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  BIGINT_INLINE /* implicit */ bigint_t(LimbT n, bool sign = false) {
    InitByCopy(&n, 1, sign);
  }

  BIGINT_INLINE bigint_t(const bigint_t &other) {
    InitByCopy(other.Data(), other.Size(), other.Sign());
  }

  BIGINT_INLINE bigint_t(bigint_t &&other) noexcept {
    InitByMove(other.Data(), other.Size(), other.Capacity(), other.Sign());

    // Invalidate the moved-from object.
    //
    // If we are using a heap buffer, we need to set the data pointer to
    // nullptr. No need to check if we are using the local buffer, as this is
    // still a valid statement.
    other.u_.data_ = nullptr;
    other.u_.capacity_ = 0;
    other.size_ = 0;
  }

  bigint_t &operator=(bigint_t other) {
    swap(*this, other);
    return *this;
  }

  ~bigint_t() noexcept {
    if (!UseLocalBuf()) {
      delete[] u_.data_;
    }
  }

  BIGINT_INLINE explicit operator LimbT() const { return At(0); }

  [[nodiscard]] std::string ToString() const {
    if (Size() == 1) {
      // Fast path: single-limb number.
      return (Sign() ? "-" : "") + std::to_string(At(0));
    } else {
      BIGINT_UNIMPL();
    }
  }

  [[nodiscard]] std::string ToHexString() const {
    std::stringstream ss;
    ss << std::hex;
    if (Sign()) {
      ss << "-";
    }
    ss << "0x";

    size_t size = Size();
    for (size_t i = 0; i < size; ++i) {
      ss << std::hex << At(size - 1 - i);
    }
    return ss.str();
  }

private:
  // Most significant bit is used to indicate the sign of the number.
  size_t size_;

  union {
    LimbT local_buf_[kLocalBufSize];
    struct {
      LimbT *data_;
      size_t capacity_;
    };
  } u_;

  BIGINT_INLINE bigint_t(std::initializer_list<LimbT> l, bool sign) {
    InitByCopy(l.begin(), l.size(), sign);
  }

  // Initializes the bigint_t object by copying the data from the given pointer.
  void InitByCopy(const LimbT *data, size_t size, bool sign) {
    size_ = size | (sign ? kSignBit : 0);
    if (size < kLocalBufSize) {
      std::copy_n(data, size, u_.local_buf_);
    } else {
      u_.data_ = new LimbT[size];
      u_.capacity_ = size;
      std::copy_n(data, size, u_.data_);
    }
    DebugSanityCheck();
  }

  // Initializes the bigint_t object by moving the data from the given pointer.
  //
  // NB: bigint_t takes ownership of the data pointer.
  void InitByMove(LimbT *data, size_t size, size_t capacity, bool sign) {
    size_ = size | (sign ? kSignBit : 0);
    if (size < kLocalBufSize) {
      std::copy_n(data, size, u_.local_buf_);

      // We have taken ownership of the data pointer, but we are using the local
      // buffer. We need to delete the data pointer.
      delete[] data;
    } else {
      u_.data_ = data;
      u_.capacity_ = capacity;
    }
    DebugSanityCheck();
  }

  [[nodiscard]] BIGINT_INLINE constexpr bool Sign() const {
    return (size_ & kSignBit) != 0;
  }

  [[nodiscard]] BIGINT_INLINE constexpr bool UseLocalBuf() const {
    return Size() < kLocalBufSize;
  }

  [[nodiscard]] BIGINT_INLINE constexpr size_t Capacity() const {
    if (UseLocalBuf()) {
      return kLocalBufSize;
    } else {
      size_t capacity = u_.capacity_;
      BIGINT_ASSUME_ASSERT(capacity >= kLocalBufSize);
      return capacity;
    }
  }

  [[nodiscard]] BIGINT_INLINE constexpr const LimbT *Data() const {
    return UseLocalBuf() ? u_.local_buf_ : u_.data_;
  }

  [[nodiscard]] BIGINT_INLINE constexpr LimbT *Data() {
    return UseLocalBuf() ? u_.local_buf_ : u_.data_;
  }

  [[nodiscard]] BIGINT_INLINE constexpr size_t Size() const {
    return size_ & ~kFlagBitsMask;
  }

  [[nodiscard]] BIGINT_INLINE constexpr const LimbT &At(size_t i) const {
    BIGINT_ASSUME_ASSERT(i < Size());
    return Data()[i];
  }

  [[nodiscard]] BIGINT_INLINE constexpr LimbT &At(size_t i) {
    BIGINT_ASSUME_ASSERT(i < Size());
    return Data()[i];
  }

  BIGINT_INLINE void Swap(bigint_t &other) noexcept {
    using std::swap;
    swap(size_, other.size_);
    swap(u_, other.u_);
  }

  friend void swap(bigint_t &a, bigint_t &b) noexcept { a.Swap(b); }

  BIGINT_INLINE void DebugSanityCheck() const {
    // Disallow empty limb arrays.
    BIGINT_ASSERT(Size() > 0);

    // Disallow negative zero.
    BIGINT_ASSERT(!(Size() == 1 && At(0) == 0 && Sign()));

    // Disallow leading zeroes.
    if (Size() > 1) {
      BIGINT_ASSERT(At(Size() - 1) != 0);
    }
  }
};

#endif // BIGINT_H
