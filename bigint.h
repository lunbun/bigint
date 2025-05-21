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

#define BIGINT_UNIMPL() \
  do { \
    std::cerr << "Unimplemented: " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::abort(); \
  } while (0)

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <string>

class bigint_t {
private:
  // A limb refers to a single machine word used to store a part of the number.
  // The term is borrowed from the GMP library
  // (https://gmplib.org/manual/Nomenclature-and-Types).
  using LimbT = uint64_t;

  static constexpr size_t kLocalBufSize = 16 / sizeof(LimbT);
  static constexpr bool kLocalBufEnabled = kLocalBufSize > 0;

  // See the comment in capacity_ for the meaning of these values.
  static constexpr size_t kSignBit = 1 << 0;
  static constexpr size_t kUseLocalBufBit = 1 << 1;
  static constexpr size_t kFlagBitsCount = 2;
  static constexpr size_t kFlagBitsMask = kSignBit | kUseLocalBufBit;
  static constexpr size_t kCapacityGranularity = 4;

  // Aligns the given size to the next multiple of kCapacityGranularity.
  BIGINT_INLINE static constexpr size_t AlignUp(size_t n) {
    static_assert((kCapacityGranularity & (kCapacityGranularity - 1)) == 0);
    return (n + kCapacityGranularity - 1) & ~(kCapacityGranularity - 1);
  }

  BIGINT_INLINE static constexpr bool AddOverflow(LimbT a, LimbT b, LimbT &res) {
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
    capacity_ = sign ? kSignBit : 0;
    if constexpr (kLocalBufEnabled) {
      capacity_ |= (1 << kFlagBitsCount) | kUseLocalBufBit;
    } else {
      capacity_ |= kCapacityGranularity;
      data_ = new LimbT[kCapacityGranularity];
      size_ = 1;
    }
    At(0) = n;
  }

  bigint_t(std::initializer_list<LimbT> l, bool sign) {
    capacity_ = sign ? kSignBit : 0;
    if (l.size() < kLocalBufSize) {
      capacity_ |= (l.size() << kFlagBitsCount) | kUseLocalBufBit;
    } else {
      size_t alignedCapacity = bigint_t::AlignUp(l.size());
      capacity_ |= alignedCapacity;
      data_ = new LimbT[alignedCapacity];
      size_ = l.size();
    }

    size_t i = 0;
    for (LimbT n : l) {
      At(i++) = n;
    }
  }

  BIGINT_INLINE explicit operator LimbT() const { return At(0); }

  bigint_t operator+(const bigint_t &other) const {
    if (Size() == 1 && other.Size() == 1) {
      // Fast path: both numbers have a single limb.
      LimbT a = At(0);
      LimbT b = other.At(0);
      if (Sign() == other.Sign()) {
        LimbT res;
        bool overflow = AddOverflow(a, b, res);
        if (BIGINT_LIKELY(!overflow)) {
          return bigint_t{res, Sign()};
        } else {
          return bigint_t{{1, res}, Sign()};
        }
      } else if (a > b) {
        return bigint_t{a - b, Sign()};
      } else if (b > a) {
        return bigint_t{b - a, other.Sign()};
      } else {
        return bigint_t{0};
      }
    } else {
      BIGINT_UNIMPL();
    }
  }

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
  // Bit 0 (least significant) is used to indicate the sign of the number.
  // Bit 1 is used to indicate whether local-buffer optimization is used.
  //  If local-buffer optimization is used, then the upper remaining bits of
  //  capacity_ are used to store size_.
  // NB: Capacity must always be a multiple of kCapacityGranularity.
  size_t capacity_;

  union {
    LimbT local_buf_[kLocalBufSize];
    struct {
      LimbT *data_;
      size_t size_;
    };
  };

  [[nodiscard]] BIGINT_INLINE constexpr bool Sign() const {
    return capacity_ & kSignBit;
  }

  [[nodiscard]] BIGINT_INLINE constexpr bool UseLocalBuf() const {
    if constexpr (!kLocalBufEnabled) {
      BIGINT_ASSERT(!(capacity_ & kUseLocalBufBit));
      return false;
    } else {
      return capacity_ & kUseLocalBufBit;
    }
  }

  [[nodiscard]] BIGINT_INLINE constexpr size_t Capacity() const {
    if (UseLocalBuf()) {
      return kLocalBufSize;
    } else {
      size_t capacity = capacity_ & ~kFlagBitsMask;
      BIGINT_ASSUME_ASSERT(capacity >= kLocalBufSize);
      return capacity;
    }
  }

  [[nodiscard]] BIGINT_INLINE constexpr const LimbT *Data() const {
    return UseLocalBuf() ? local_buf_ : data_;
  }

  [[nodiscard]] BIGINT_INLINE constexpr LimbT *Data() {
    return UseLocalBuf() ? local_buf_ : data_;
  }

  [[nodiscard]] BIGINT_INLINE constexpr size_t Size() const {
    if (UseLocalBuf()) {
      size_t size = capacity_ >> kFlagBitsCount;
      BIGINT_ASSUME_ASSERT(0 < size && size < kLocalBufSize);
      return size;
    } else {
      size_t size = size_;
      BIGINT_ASSUME_ASSERT(size >= kLocalBufSize);
      return size;
    }
  }

  [[nodiscard]] BIGINT_INLINE constexpr const LimbT &At(size_t i) const {
    BIGINT_ASSUME_ASSERT(i < Size());
    return Data()[i];
  }

  [[nodiscard]] BIGINT_INLINE constexpr LimbT &At(size_t i) {
    BIGINT_ASSUME_ASSERT(i < Size());
    return Data()[i];
  }
};

#endif // BIGINT_H
