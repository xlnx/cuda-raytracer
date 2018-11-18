#pragma once

#include <utility>
#include <type_traits>
#include <vec/trait.hpp>
#include <vec/vmath.hpp>

#include "kernel.hpp"
#include "debug.hpp"

namespace koishi
{
namespace core
{
// PolyVectorView holds a read-only data vector for either cpu or gpu
// use std::move to make
template <typename T>
struct PolyVectorView final : Emittable<PolyVectorView<T>>
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
	using buffer_type = PolyVector<T>;

public:
	PolyVectorView() = default;
	PolyVectorView( size_type count ) :
	  value( (pointer)std::malloc( sizeof( T ) * count ) ),
	  curr( count )
	{
	}
	KOISHI_HOST_DEVICE PolyVectorView( PolyVectorView &&other ) :
	  value( other.value ),
	  curr( other.curr ),
	  is_device_ptr( other.is_device_ptr )
	{
		other.value = nullptr;
	}
	KOISHI_HOST_DEVICE PolyVectorView &operator=( PolyVectorView &&other )
	{
		destroy();
		value = other.value;
		curr = other.curr;
		is_device_ptr = other.is_device_ptr;
		other.value = nullptr;
		return *this;
	}
	~PolyVectorView()
	{
		destroy();
	}

	PolyVectorView( const PolyVectorView &other )
#ifdef KOISHI_USE_CUDA
	  :
	  value( other.value ),
	  curr( other.curr ),
	  is_device_ptr( other.is_device_ptr )
	{
		copyBetweenDevice( other );
	}
#else
	{
		THROW( invalid use of PolyVectorView( const & ) );
	}
#endif
	PolyVectorView &operator=( const PolyVectorView &other )
#ifdef KOISHI_USE_CUDA
	{
		Emittable<PolyVectorView<T>>::operator=( std::move( const_cast<PolyVectorView &>( other ) ) );
		value = other.value;
		curr = other.curr;
		is_device_ptr = other.is_device_ptr;
		copyBetweenDevice( other );
		return *this;
	}
#else
	{
		THROW( invalid use of PolyVectorView( const & ) );
	}
#endif

private:
#ifdef KOISHI_USE_CUDA
	void copyBetweenDevice( const PolyVectorView &other )
	{
		if ( !__impl::Emittable::isTransferring() )
		{
			THROW( invalid use of PolyVectorView( const & ) );
		}
		pointer new_ptr;
		auto alloc_size = sizeof( T ) * curr;
		if ( is_device_ptr )
		{
			new_ptr = (pointer)std::malloc( alloc_size );
			__impl::Mover<T>::device_to_host( new_ptr, value, curr );
		}
		else
		{
			if ( auto err = cudaMalloc( &new_ptr, alloc_size ) )
			{
				THROW( cudaMalloc on device failed );
			}
			__impl::Mover<T>::host_to_device( new_ptr, value, curr );
		}
		if ( &other == this )
		{
			destroy();
		}
		value = new_ptr;
		is_device_ptr = !is_device_ptr;
		LOG( "value", value );
	}

#endif
	void destroy()
	{
		if ( value != nullptr )
		{
			LOG( "destroy()", typeid( T ).name(), this );
			if ( is_device_ptr )
			{
#ifdef KOISHI_USE_CUDA
				__impl::Destroyer<T>::destroy_device( value, curr );
				cudaFree( value );
#else
				THROW( invalid internal state );
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
	PolyVectorView( const buffer_type &other ) = delete;
	PolyVectorView( buffer_type &&other ) :
	  value( other.value ),
	  curr( other.curr )
	{
		other.value = nullptr;
	}
	PolyVectorView &operator=( const buffer_type &other ) = delete;
	PolyVectorView &operator=( buffer_type &&other )
	{
		destroy();
		value = other.value;
		curr = other.curr;
		is_device_ptr = false;
		other.value = nullptr;
		return *this;
	}

public:
	KOISHI_HOST_DEVICE reference operator[]( size_type idx ) { return value[ idx ]; }
	KOISHI_HOST_DEVICE const_reference operator[]( size_type idx ) const { return value[ idx ]; }

	KOISHI_HOST_DEVICE reference front() { return *value; }
	KOISHI_HOST_DEVICE const_reference front() const { return *value; }

	KOISHI_HOST_DEVICE reference back() { return value[ curr - 1 ]; }
	KOISHI_HOST_DEVICE const_reference back() const { return value[ curr - 1 ]; }

	KOISHI_HOST_DEVICE pointer data() { return value; }
	KOISHI_HOST_DEVICE const_pointer data() const { return value; }

	KOISHI_HOST_DEVICE iterator begin() { return value; }
	KOISHI_HOST_DEVICE const_iterator begin() const { return value; }

	KOISHI_HOST_DEVICE iterator end() { return value + curr; }
	KOISHI_HOST_DEVICE const_iterator end() const { return value + curr; }

	KOISHI_HOST_DEVICE bool empty() const { return curr == 0; }

	KOISHI_HOST_DEVICE size_type size() const { return curr; }

private:
	T *value = nullptr;
	size_type curr = 0;
	bool is_device_ptr = false;
};

}  // namespace core

}  // namespace koishi