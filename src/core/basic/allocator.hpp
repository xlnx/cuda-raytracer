#pragma once

#include <type_traits>
#include <cstddef>
#include <algorithm>
#include <memory>
#include <vector>
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

	KOISHI_HOST_DEVICE Allocator( const Allocator & ) = delete;
	KOISHI_HOST_DEVICE Allocator &operator=( const Allocator & ) = delete;
};

template <typename T, typename... Args>
KOISHI_HOST_DEVICE inline T *alloc( Allocator &al, Args &&... args )
{
	auto ptr = reinterpret_cast<T *>( al.alloc( sizeof( T ) ) );
	new ( ptr ) T( std::forward<Args>( args )... );
	return ptr;
}

KOISHI_HOST_DEVICE inline void clear( Allocator &al )
{
	al.clear();
}

struct HostAllocator : core::Allocator, Require<Host>
{
	KOISHI_HOST HostAllocator() = default;

	KOISHI_HOST ~HostAllocator()
	{
		for ( auto &block : blocks ) { delete[] block.second; }
	}

	KOISHI_HOST_DEVICE char *alloc( std::size_t size ) override
	{
		return do_alloc( size );
	}
	KOISHI_HOST_DEVICE void clear() override { return do_clear(); }

private:
	KOISHI_HOST char *do_alloc( std::size_t size )
	{
		static constexpr std::size_t align =
#if __GNUC__ == 4 && __GNUC_MINOR__ < 9
		  alignof( ::max_align_t );
#else
		  alignof( std::max_align_t );
#endif
		while ( !( data = reinterpret_cast<char_type *>(
					 std::align( align, size, reinterpret_cast<void *&>( data ), rest_size ) ) ) )
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

private:
	const std::size_t block_size = 2048u;

	using char_type = typename std::aligned_storage<1, 64u>::type;

	std::size_t rest_size = block_size, curr_block = 0;
	char_type *data = new char_type[ rest_size ];
	std::vector<std::pair<std::size_t, char_type *>> blocks{ { rest_size, data } };
};

#if defined( KOISHI_USE_CUDA )

struct DeviceAllocator : core::Allocator, Require<Device>
{
	KOISHI_DEVICE DeviceAllocator()
	{
	}

	KOISHI_HOST_DEVICE char *alloc( std::size_t size ) override { return do_alloc( size ); }
	KOISHI_HOST_DEVICE void clear() override { return do_clear(); }

private:
	KOISHI_DEVICE char *do_alloc( std::size_t size )
	{
	}
	KOISHI_DEVICE void do_clear()
	{
	}
};

#endif

}  // namespace core

}  // namespace koishi