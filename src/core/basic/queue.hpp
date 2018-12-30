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
	KOISHI_HOST_DEVICE CyclicQueue( Allocator &pool, uint alloc_size = 0 ) :
	  pool( pool )
	{
		if ( !alloc_size ) alloc_size = pool.size() / sizeof( T );
		front_ptr = back_ptr = base = alloc_uninitialized<T>( pool, alloc_size );
		finish = base + alloc_size;
	}

	KOISHI_HOST_DEVICE ~CyclicQueue()
	{
		dealloc( pool, base );
	}

	template <typename... Args>
	KOISHI_HOST_DEVICE void emplace( Args &&... args )
	{
		new ( back_ptr++ ) T( std::forward<Args>( args )... );
		if ( back_ptr == finish ) back_ptr = base;
		of |= back_ptr == front_ptr;
	}

	KOISHI_HOST_DEVICE void pop()
	{
		front_ptr++->~T();
		if ( front_ptr == finish ) front_ptr = base;
	}

	KOISHI_HOST_DEVICE T &front() { return *front_ptr; }
	KOISHI_HOST_DEVICE const T &front() const { return *front_ptr; }

	KOISHI_HOST_DEVICE void clear() { front_ptr = back_ptr, of = false; }

	KOISHI_HOST_DEVICE bool overflow() const { return of; }

	KOISHI_HOST_DEVICE uint capacity() const { return finish - base; }

	KOISHI_HOST_DEVICE bool empty() const { return front_ptr == back_ptr; }

private:
	Allocator &pool;
	T *base, *finish, *front_ptr, *back_ptr;
	bool of = false;
};

template <typename T>
struct Stack
{
	KOISHI_HOST_DEVICE Stack( Allocator &pool, uint alloc_size = 0 ) :
	  pool( pool )
	{
		if ( !alloc_size ) alloc_size = pool.size() / sizeof( T );
		top_ptr = base = alloc_uninitialized<T>( pool, alloc_size );
		full_ptr = base + alloc_size;
	}

	KOISHI_HOST_DEVICE ~Stack()
	{
		dealloc( pool, base );
	}

	template <typename... Args>
	KOISHI_HOST_DEVICE void emplace( Args &&... args )
	{
		new ( top_ptr++ ) T( std::forward<Args>( args )... );
	}

	KOISHI_HOST_DEVICE void pop()
	{
		( --top_ptr )->~T();
	}

	KOISHI_HOST_DEVICE T &top() { return top_ptr[ -1 ]; }
	KOISHI_HOST_DEVICE const T &top() const { return top_ptr[ -1 ]; }

	KOISHI_HOST_DEVICE void clear() { top_ptr = base; }

	KOISHI_HOST_DEVICE uint capacity() const { return full_ptr - base; }

	KOISHI_HOST_DEVICE bool empty() const { return top_ptr == base; }

private:
	Allocator &pool;
	T *base, *top_ptr, *full_ptr;
};

}  // namespace core

}  // namespace koishi
