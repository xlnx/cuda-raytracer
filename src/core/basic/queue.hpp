#pragma once

#include <utility>
#include <vec/vmath.hpp>
#include "allocator.hpp"

namespace koishi
{
namespace core
{
template <typename T>
struct CyclicQueue
{
	template <typename Alloc>
	KOISHI_HOST_DEVICE CyclicQueue( Alloc &pool, uint alloc_size = 0 )
	{
		front_ptr = back_ptr = base = alloc_uninitialized<T>( pool, alloc_size );
		finish = base + alloc_size;
	}

	template <typename... Args>
	KOISHI_HOST_DEVICE void emplace( Args &&... args )
	{
		of |= back_ptr == front_ptr;
		new ( back_ptr++ ) T( std::forward<Args>( args )... );
		if ( back_ptr == finish ) back_ptr = base;
	}

	KOISHI_HOST_DEVICE void pop()
	{
		front_ptr++->~T();
		if ( front_ptr == finish ) front_ptr = base;
	}

	KOISHI_HOST_DEVICE T &front() { return *front_ptr; }
	KOISHI_HOST_DEVICE const T &front() const { return *front_ptr; }

	KOISHI_HOST_DEVICE bool overflow() const { return of; }

	KOISHI_HOST_DEVICE uint capacity() const { return finish - base; }

	KOISHI_HOST_DEVICE bool empty() const { return front_ptr == back_ptr; }

private:
	T *base, *finish, *front_ptr, *back_ptr;
	bool of = false;
};

}  // namespace core

}  // namespace koishi