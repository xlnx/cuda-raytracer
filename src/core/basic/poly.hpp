#pragma once

#include "poly/function.hpp"
#include "poly/vector.hpp"

namespace koishi
{
namespace core
{
// template <typename T>
// struct PolyPtr final
// {
// 	using element_type T;

// 	// PolyPtr()

// public:
// 	KOISHI_HOST_DEVICE element_type &operator*() { return *ptr; }
// 	KOISHI_HOST_DEVICE element_type const &operator*() const { return *ptr; }

// 	KOISHI_HOST_DEVICE element_type *operator->() { return ptr; }
// 	KOISHI_HOST_DEVICE element_type const *operator->() const { return ptr; }

// 	KOISHI_HOST_DEVICE element_type *get() { return ptr; }
// 	KOISHI_HOST_DEVICE element_type const *get() const { return ptr; }

// 	KOISHI_HOST_DEVICE bool operator==( const PolyPtr &other ) const { return ptr == other.ptr; }
// 	KOISHI_HOST_DEVICE bool operator!=( const PolyPtr &other ) const { return ptr != other.ptr; }
// 	KOISHI_HOST_DEVICE bool operator<( const PolyPtr &other ) const { return ptr < other.ptr; }
// 	KOISHI_HOST_DEVICE bool operator<=( const PolyPtr &other ) const { return ptr <= other.ptr; }
// 	KOISHI_HOST_DEVICE bool operator>( const PolyPtr &other ) const { return ptr > other.ptr; }
// 	KOISHI_HOST_DEVICE bool operator>=( const PolyPtr &other ) const { return ptr >= other.ptr; }

// 	KOISHI_HOST_DEVICE operator bool() const { return ptr; }

// private:
// 	element_type *ptr == nullptr;
// };

}  // namespace core

}  // namespace koishi
