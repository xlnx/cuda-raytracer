#pragma once

#include <type_traits>
#include <cstddef>
#include <algorithm>
#include <memory>
#include <vector>
#include <util/debug.hpp>
#include <vec/vmath.hpp>
#include "poly.hpp"

namespace koishi
{
namespace core
{
struct Allocator
{
	KOISHI_HOST_DEVICE Allocator() = default;
	KOISHI_HOST_DEVICE virtual ~Allocator() = default;

	KOISHI_HOST_DEVICE virtual char *alloc( std::size_t size ) = 0;

	KOISHI_HOST_DEVICE virtual void clear() = 0;

	KOISHI_HOST_DEVICE virtual std::size_t size() const = 0;

	KOISHI_HOST_DEVICE Allocator( const Allocator & ) = delete;
	KOISHI_HOST_DEVICE Allocator &operator=( const Allocator & ) = delete;
};

template <typename T, typename Alloc, typename... Args>
KOISHI_HOST_DEVICE inline T *create( Alloc &al, Args &&... args )
{
	auto ptr = reinterpret_cast<T *>( al.alloc( sizeof( T ) ) );
	new ( ptr ) T( std::forward<Args>( args )... );
	return ptr;
}

template <typename T, typename Alloc>
KOISHI_HOST_DEVICE inline T *alloc_uninitialized( Alloc &al, std::size_t count )
{
	return reinterpret_cast<T *>( al.alloc( sizeof( T ) * count ) );
}

template <typename T, typename Alloc>
KOISHI_HOST_DEVICE inline T *alloc( Alloc &al, std::size_t count )
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

struct HostAllocator : core::Allocator, Require<Host>
{
	KOISHI_HOST HostAllocator() = default;

	KOISHI_HOST_DEVICE ~HostAllocator()
	{
		destroy();
	}

	KOISHI_HOST void destroy()
	{
		for ( auto &block : blocks ) { delete[] block.second; }
	}

	KOISHI_HOST_DEVICE char *alloc( std::size_t size ) override { return do_alloc( size ); }
	KOISHI_HOST_DEVICE void clear() override { return do_clear(); }
	KOISHI_HOST_DEVICE std::size_t size() const override { return do_size(); }

private:
	KOISHI_HOST char *do_alloc( std::size_t size )
	{
		static constexpr std::size_t align =
#if __GNUC__ == 4 && __GNUC_MINOR__ < 9
		  alignof( ::max_align_t );
		while ( !( data = reinterpret_cast<char_type *>(
					 align_impl( align, size, reinterpret_cast<void *&>( data ), rest_size ) ) ) )
#else
		  alignof( std::max_align_t );
		while ( !( data = reinterpret_cast<char_type *>(
					 std::align( align, size, reinterpret_cast<void *&>( data ), rest_size ) ) ) )
#endif
		{
			if ( blocks.size() > curr_block + 1 )
			{
				auto &block = blocks[ ++curr_block ];
				rest_size = block.first;
				data = block.second;
			}
			else
			{
				rest_size = std::max( block_size, size );
				++curr_block;
				data = new char_type[ rest_size ];
				blocks.emplace_back( rest_size, data );
			}
		}
		auto ptr = data;
		data += size, rest_size -= size;
		return reinterpret_cast<char *>( ptr );
	}
	KOISHI_HOST void do_clear()
	{
		rest_size = block_size;
		curr_block = 0;
		data = blocks[ 0 ].second;
	}
	KOISHI_HOST std::size_t do_size() const
	{
		return rest_size;
	}

private:
	static void *align_impl( std::size_t __align, std::size_t __size, void *&__ptr, std::size_t &__space ) noexcept
	{
		const auto __intptr = reinterpret_cast<std::uintptr_t>( __ptr );
		const auto __aligned = ( __intptr - 1u + __align ) & -__align;
		const auto __diff = __aligned - __intptr;
		if ( ( __size + __diff ) > __space )
			return nullptr;
		else
		{
			__space -= __diff;
			return __ptr = reinterpret_cast<void *>( __aligned );
		}
	}

private:
	const std::size_t block_size = 2048u;

	using char_type = typename std::aligned_storage<1, 64u>::type;

	std::size_t rest_size = block_size, curr_block = 0;
	char_type *data = new char_type[ rest_size ];
	std::vector<std::pair<std::size_t, char_type *>> blocks{ { rest_size, data } };
};

struct HybridAllocator : core::Allocator, Require<Device>, Require<Host>
{
	KOISHI_DEVICE HybridAllocator( char *block, uint block_size ) :
	  base( block ), finish( block + block_size ), curr( block )
	{
	}

	KOISHI_HOST_DEVICE char *alloc( std::size_t size ) override
	{
		KASSERT( finish - curr >= size );
		auto ptr = curr;
		curr += size;
		return ptr;
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
};  // namespace core

// #if defined( KOISHI_USE_CUDA )

// struct DeviceAllocator : core::Allocator, Require<Device>
// {
// 	KOISHI_DEVICE DeviceAllocator()
// 	{
// 	}

// 	KOISHI_HOST_DEVICE char *alloc( std::size_t size ) override { return do_alloc( size ); }
// 	KOISHI_HOST_DEVICE void clear() override { return do_clear(); }

// private:
// 	KOISHI_HOST_DEVICE char *do_alloc( std::size_t size )
// 	{
// 	}
// 	KOISHI_HOST_DEVICE void do_clear()
// 	{
// 	}
// };

// #endif

}  // namespace core

}  // namespace koishi
