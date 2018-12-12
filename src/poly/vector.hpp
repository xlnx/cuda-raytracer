#pragma once

#include <utility>
#include <vector>
#include <initializer_list>
#include <type_traits>
#include <vec/trait.hpp>
#include <vec/vmath.hpp>
#include <util/debug.hpp>
#include "kernel.hpp"

#define MIN_SIZE std::size_t( 4 )

namespace koishi
{
namespace poly
{
// vector holds a read-only data vector for either cpu or gpu
// use std::move to make
template <typename T>
struct vector final : emittable
{
	using value_type = T;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using reference = T &;
	using const_reference = const T &;
	using pointer = T *;
	using const_pointer = const T *;
	using iterator = T *;
	using const_iterator = const T *;

	// private:
	// 	static constexpr size_type MIN_SIZE = 4;

public:
	vector() = default;
	vector( size_type count, const T &val = T() )
	{
		resize( count, val );
	}
	vector( std::initializer_list<T> l ) :
	  total( std::max( l.size(), MIN_SIZE ) ),
	  curr( l.size() )
	{
		auto p = value;
		auto q = &*l.begin();
		for ( ; p != value + curr; ++p, ++q )
		{
			new ( p ) T( std::move( *q ) );
		}
	}
	vector &operator=( std::initializer_list<T> l )
	{
		destroy();
		total = std::max( l.size(), MIN_SIZE );
		curr = l.size();
		value = (T *)std::malloc( sizeof( T ) * total );
		auto p = value;
		auto q = &*l.begin();
		for ( ; p != value + curr; ++p, ++q )
		{
			new ( p ) T( std::move( *q ) );
		}
		is_device_ptr = false;
		return *this;
	}
	vector( const std::vector<T> &other ) :
	  total( std::max( other.capacity(), MIN_SIZE ) ),
	  curr( other.size() )
	{
		for ( auto p = value, q = &other[ 0 ]; p != value + curr; ++p, ++q )
		{
			new ( p ) T( *q );
		}
	}
	vector( std::vector<T> &&other ) :
	  total( std::max( other.capacity(), MIN_SIZE ) ),
	  curr( other.size() )
	{
		for ( auto p = value, q = &other[ 0 ]; p != value + curr; ++p, ++q )
		{
			new ( p ) T( std::move( *q ) );
		}
	}
	vector &operator=( const std::vector<T> &other )
	{
		destroy();
		total = std::max( other.capacity(), MIN_SIZE );
		curr = other.size();
		value = (T *)std::malloc( sizeof( T ) * total );
		for ( auto p = value, q = &other[ 0 ]; p != value + curr; ++p, ++q )
		{
			new ( p ) T( *q );
		}
		is_device_ptr = false;
		return *this;
	}
	vector &operator=( std::vector<T> &&other )
	{
		destroy();
		total = std::max( other.capacity(), MIN_SIZE );
		curr = other.size();
		value = (T *)std::malloc( sizeof( T ) * total );
		for ( auto p = value, q = &other[ 0 ]; p != value + curr; ++p, ++q )
		{
			new ( p ) T( std::move( *q ) );
		}
		is_device_ptr = false;
		return *this;
	}
	void swap( T *&other, std::size_t &count )
	{
		auto tmp = curr;
		total = curr = count;
		count = tmp;
		auto old = value;
		value = other;
		other = old;
	}

public:
	KOISHI_HOST_DEVICE vector( vector &&other ) :
	  emittable( std::move( other ) ),
	  total( other.total ),
	  curr( other.curr ),
	  value( other.value ),
	  device_value( other.device_value ),
	  is_device_ptr( other.is_device_ptr )
	{
		other.value = nullptr;
	}
	KOISHI_HOST_DEVICE vector &operator=( vector &&other )
	{
		destroy();
		total = other.total;
		curr = other.curr;
		value = other.value;
		device_value = other.device_value;
		is_device_ptr = other.is_device_ptr;
		other.value = nullptr;
		return *this;
	}
	~vector()
	{
		destroy();
	}

	vector( const vector &other )
#ifdef KOISHI_USE_CUDA
	  :
	  emittable( other ),
	  total( other.total ),
	  curr( other.curr ),
	  value( other.value ),
	  device_value( other.device_value ),
	  is_device_ptr( other.is_device_ptr )
	{
		copyBetweenDevice( other );
	}
#else
	{
		KTHROW( "invalid use of vector( const & )" );
	}
#endif
	vector &operator=( const vector &other )
#ifdef KOISHI_USE_CUDA
	{
		// emittable::operator=( std::move( const_cast<vector &>( other ) ) );
		total = other.total;
		curr = other.curr;
		value = other.value;
		device_value = other.device_value;
		is_device_ptr = other.is_device_ptr;
		copyBetweenDevice( other );
		return *this;
	}
#else
	{
		KTHROW( "invalid use of vector( const & )" );
	}
#endif

private:
#ifdef KOISHI_USE_CUDA
	void copyBetweenDevice( const vector &other )
	{
		if ( !__impl::Emittable::isTransferring() )
		{
			KTHROW( "invalid use of vector( const & )" );
		}
		if ( is_device_ptr )
		{
			__impl::Mover<T>::device_to_host( value, device_value, curr );
			cudaFree( device_value );
		}
		else
		{
			auto alloc_size = sizeof( T ) * total;
			T *ptr;
			if ( auto err = cudaMalloc( &ptr, alloc_size ) )
			{
				KTHROW( "cudaMalloc on device failed" );
			}
			device_value = ptr;
			__impl::Mover<T>::host_to_device( device_value, value, curr );
		}
		is_device_ptr = !is_device_ptr;
		KLOG3( "value", value );
	}

#endif
	void destroy()
	{
		if ( value != nullptr )
		{
			KLOG3( "destroy()", typeid( T ).name(), this );
			if ( is_device_ptr )
			{
				KTHROW( "invalid internal state" );
			}
			else
			{
				__impl::Destroyer<T>::destroy_host( value, curr );
				std::free( value );
			}
		}
	}

public:
#ifdef __CUDA_ARCH__
#  define KOISHI_DATA_PTR device_value
#else
#  define KOISHI_DATA_PTR value
#endif
	KOISHI_HOST_DEVICE reference operator[]( size_type idx ) { return KOISHI_DATA_PTR[ idx ]; }
	KOISHI_HOST_DEVICE const_reference operator[]( size_type idx ) const { return KOISHI_DATA_PTR[ idx ]; }

	KOISHI_HOST_DEVICE reference front() { return *KOISHI_DATA_PTR; }
	KOISHI_HOST_DEVICE const_reference front() const { return *KOISHI_DATA_PTR; }

	KOISHI_HOST_DEVICE reference back() { return KOISHI_DATA_PTR[ curr - 1 ]; }
	KOISHI_HOST_DEVICE const_reference back() const { return KOISHI_DATA_PTR[ curr - 1 ]; }

	KOISHI_HOST pointer data() { return KOISHI_DATA_PTR; }
	KOISHI_HOST const_pointer data() const { return KOISHI_DATA_PTR; }

	KOISHI_HOST_DEVICE iterator begin() { return KOISHI_DATA_PTR; }
	KOISHI_HOST_DEVICE const_iterator begin() const { return KOISHI_DATA_PTR; }

	KOISHI_HOST_DEVICE iterator end() { return KOISHI_DATA_PTR + curr; }
	KOISHI_HOST_DEVICE const_iterator end() const { return KOISHI_DATA_PTR + curr; }

	KOISHI_HOST_DEVICE bool empty() const { return curr == 0; }

	KOISHI_HOST_DEVICE size_type size() const { return curr; }

	KOISHI_HOST_DEVICE size_type capacity() const { return total; }
#undef KOISHI_DATA_PTR

public:
	template <typename... Args>
	void emplace_back( Args &&... args )
	{
		KASSERT( !is_device_ptr );
		if ( curr >= total )
		{
			auto newData = (pointer)std::malloc( sizeof( T ) * ( total *= 2 ) );
			for ( auto p = value, q = newData; p != value + curr; ++p, ++q )
			{
				new ( q ) T( std::move( *p ) );
				p->~T();
			}
			std::free( value );
			value = newData;
		}
		auto ptr = value + curr++;
		new ( ptr ) T( std::forward<Args>( args )... );
	}

	void resize( size_type count )
	{
		KASSERT( !is_device_ptr );
		if ( curr >= count )
		{
			for ( auto p = value + count; p != value + curr; ++p )
			{
				p->~T();
			}
			curr = count;
		}
		else if ( total >= count )
		{
			for ( auto p = value + curr; p != value + count; ++p )
			{
				new ( p ) T();
			}
			curr = count;
		}
		else
		{
			total = std::max( MIN_SIZE, count );
			auto new_ptr = (pointer)std::malloc( sizeof( T ) * total );
			for ( auto p = value, q = new_ptr; p != value + curr; ++p, ++q )
			{
				new ( q ) T( std::move( *p ) );
				p->~T();
			}
			destroy();
			for ( auto p = new_ptr + curr; p != new_ptr + count; ++p )
			{
				new ( p ) T();
			}
			curr = count;
			value = new_ptr;
		}
	}

	void resize( size_type count, const value_type &val )
	{
		static_assert( !std::is_base_of<emittable, value_type>::value,
					   "emittable type is not copyable" );
		KASSERT( !is_device_ptr );
		if ( curr >= count )
		{
			for ( auto p = value + count; p != value + curr; ++p )
			{
				p->~T();
			}
			curr = count;
		}
		else if ( total >= count )
		{
			for ( auto p = value + curr; p != value + count; ++p )
			{
				new ( p ) T( val );
			}
			curr = count;
		}
		else
		{
			total = std::max( MIN_SIZE, count );
			auto new_ptr = (pointer)std::malloc( sizeof( T ) * total );
			for ( auto p = value, q = new_ptr; p != value + curr; ++p, ++q )
			{
				new ( q ) T( std::move( *p ) );
				p->~T();
			}
			destroy();
			for ( auto p = new_ptr + curr; p != new_ptr + count; ++p )
			{
				new ( p ) T( val );
			}
			curr = count;
			value = new_ptr;
		}
	}

private:
	std::size_t total = MIN_SIZE;
	size_type curr = 0;
	T *KOISHI_RESTRICT value = (pointer)std::malloc( sizeof( T ) * total );
	T *KOISHI_RESTRICT device_value = nullptr;
	bool is_device_ptr = false;
};

}  // namespace poly

}  // namespace koishi
