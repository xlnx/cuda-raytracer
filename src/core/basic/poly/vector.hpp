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
namespace core
{
// PolyVector holds a read-only data vector for either cpu or gpu
// use std::move to make
template <typename T>
struct PolyVector final : Emittable<PolyVector<T>>
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
	PolyVector() = default;
	PolyVector( size_type count, const T &val = T() )
	{
		resize( count, val );
	}
	PolyVector( std::initializer_list<T> l ) :
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
	PolyVector &operator=( std::initializer_list<T> l )
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
	PolyVector( const std::vector<T> &other ) :
	  total( std::max( other.capacity(), MIN_SIZE ) ),
	  curr( other.size() )
	{
		for ( auto p = value, q = &other[ 0 ]; p != value + curr; ++p, ++q )
		{
			new ( p ) T( *q );
		}
	}
	PolyVector( std::vector<T> &&other ) :
	  total( std::max( other.capacity(), MIN_SIZE ) ),
	  curr( other.size() )
	{
		for ( auto p = value, q = &other[ 0 ]; p != value + curr; ++p, ++q )
		{
			new ( p ) T( std::move( *q ) );
		}
	}
	PolyVector &operator=( const std::vector<T> &other )
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
	PolyVector &operator=( std::vector<T> &&other )
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

public:
	KOISHI_HOST_DEVICE PolyVector( PolyVector &&other ) :
	  total( other.total ),
	  curr( other.curr ),
	  value( other.value ),
	  is_device_ptr( other.is_device_ptr )
	{
		other.value = nullptr;
	}
	KOISHI_HOST_DEVICE PolyVector &operator=( PolyVector &&other )
	{
		destroy();
		total = other.total;
		curr = other.curr;
		value = other.value;
		is_device_ptr = other.is_device_ptr;
		other.value = nullptr;
		return *this;
	}
	~PolyVector()
	{
		destroy();
	}

	PolyVector( const PolyVector &other )
#ifdef KOISHI_USE_CUDA
	  :
	  total( other.total ),
	  curr( other.curr ),
	  value( other.value ),
	  is_device_ptr( other.is_device_ptr )
	{
		copyBetweenDevice( other );
	}
#else
	{
		KTHROW( invalid use of PolyVector( const & ) );
	}
#endif
	PolyVector &operator=( const PolyVector &other )
#ifdef KOISHI_USE_CUDA
	{
		Emittable<PolyVector<T>>::operator=( std::move( const_cast<PolyVector &>( other ) ) );
		total = other.total;
		curr = other.curr;
		value = other.value;
		is_device_ptr = other.is_device_ptr;
		copyBetweenDevice( other );
		return *this;
	}
#else
	{
		KTHROW( invalid use of PolyVector( const & ) );
	}
#endif

private:
#ifdef KOISHI_USE_CUDA
	void copyBetweenDevice( const PolyVector &other )
	{
		if ( !__impl::Emittable::isTransferring() )
		{
			KTHROW( invalid use of PolyVector( const & ) );
		}
		pointer new_ptr;
		auto alloc_size = sizeof( T ) * total;
		if ( is_device_ptr )
		{
			new_ptr = (pointer)std::malloc( alloc_size );
			__impl::Mover<T>::device_to_host( new_ptr, value, curr );
		}
		else
		{
			if ( auto err = cudaMalloc( &new_ptr, alloc_size ) )
			{
				KTHROW( cudaMalloc on device failed );
			}
			__impl::Mover<T>::host_to_device( new_ptr, value, curr );
		}
		if ( &other == this )
		{
			destroy();
		}
		value = new_ptr;
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
#ifdef KOISHI_USE_CUDA
				__impl::Destroyer<T>::destroy_device( value, curr );
				cudaFree( value );
#else
				KTHROW( invalid internal state );
#endif
			}
			else
			{
				__impl::Destroyer<T>::destroy_host( value, curr );
				std::free( value );
			}
		}
	}

public:
	KOISHI_HOST_DEVICE reference operator[]( size_type idx ) { return value[ idx ]; }
	KOISHI_HOST_DEVICE const_reference operator[]( size_type idx ) const { return value[ idx ]; }

	KOISHI_HOST_DEVICE reference front() { return *value; }
	KOISHI_HOST_DEVICE const_reference front() const { return *value; }

	KOISHI_HOST_DEVICE reference back() { return value[ curr - 1 ]; }
	KOISHI_HOST_DEVICE const_reference back() const { return value[ curr - 1 ]; }

	KOISHI_HOST pointer data() { return value; }
	KOISHI_HOST const_pointer data() const { return value; }

	KOISHI_HOST_DEVICE iterator begin() { return value; }
	KOISHI_HOST_DEVICE const_iterator begin() const { return value; }

	KOISHI_HOST_DEVICE iterator end() { return value + curr; }
	KOISHI_HOST_DEVICE const_iterator end() const { return value + curr; }

	KOISHI_HOST_DEVICE bool empty() const { return curr == 0; }

	KOISHI_HOST_DEVICE size_type size() const { return curr; }

	KOISHI_HOST_DEVICE size_type capacity() const { return total; }

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

	void resize( size_type count, const value_type &val = T() )
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
	T * KOISHI_RESTRICT value = (pointer)std::malloc( sizeof( T ) * total );
	bool is_device_ptr = false;
};

}  // namespace core

}  // namespace koishi
