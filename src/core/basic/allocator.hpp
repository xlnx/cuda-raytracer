#pragma once

#include <type_traits>
#include <algorithm>
#include <cstddef>
#include <algorithm>
#include <memory>
#include <vector>
#include <util/debug.hpp>
#include <vec/vmath.hpp>

namespace koishi
{
namespace core
{
struct Allocator
{
	KOISHI_HOST_DEVICE Allocator() = default;
	KOISHI_HOST_DEVICE virtual ~Allocator() = default;

	KOISHI_HOST_DEVICE virtual char *alloc( std::size_t size ) = 0;
	KOISHI_HOST_DEVICE virtual void dealloc( char *ptr ) = 0;

	KOISHI_HOST_DEVICE virtual void clear() = 0;

	KOISHI_HOST_DEVICE virtual std::size_t size() const = 0;

	KOISHI_HOST_DEVICE Allocator( const Allocator & ) = delete;
	KOISHI_HOST_DEVICE Allocator &operator=( const Allocator & ) = delete;
};

template <typename T, typename... Args>
KOISHI_HOST_DEVICE inline T *create( Allocator &al, Args &&... args )
{
	auto ptr = reinterpret_cast<T *>( al.alloc( sizeof( T ) ) );
	new ( ptr ) T( std::forward<Args>( args )... );
	return ptr;
}

template <typename T>
KOISHI_HOST_DEVICE inline T *alloc_uninitialized( Allocator &al, std::size_t count )
{
	return reinterpret_cast<T *>( al.alloc( sizeof( T ) * count ) );
}

template <typename T>
KOISHI_HOST_DEVICE inline T *alloc( Allocator &al, std::size_t count )
{
	auto ptr = alloc_uninitialized<T>( al, count );
	if ( !std::is_pod<T>::value )
	{
		for ( auto q = ptr; q != ptr + count; ++q )
		{
			new ( q ) T();
		}
	}
	return ptr;
}

template <typename T>
KOISHI_HOST_DEVICE inline void dealloc( Allocator &al, T *ptr )
{
	al.dealloc( reinterpret_cast<char *>( ptr ) );
}

struct HybridAllocator : Allocator
{
	KOISHI_HOST_DEVICE HybridAllocator( char *block, uint block_size ) :
	  base( block ),
	  finish( block + block_size ),
	  curr( block )
	{
	}

	KOISHI_HOST_DEVICE char *alloc( std::size_t size ) override
	{
		KASSERT( finish - curr >= size );
		auto ptr = curr;
		curr += size;
		return ptr;
	}
	KOISHI_HOST_DEVICE void dealloc( char *ptr ) override
	{
		if ( ptr < curr ) curr = ptr;
	}
	KOISHI_HOST_DEVICE void clear() override
	{
		curr = base;
	}
	KOISHI_HOST_DEVICE std::size_t size() const override
	{
		return finish - curr;
	}

private:
	char *base, *finish, *curr;
};

}  // namespace core

}  // namespace koishi
