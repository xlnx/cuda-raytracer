#pragma once

#include <utility>
#include <typeinfo>
#include <type_traits>
#include <vec/trait.hpp>
#include <vec/vmath.hpp>
#include <util/debug.hpp>
#include "kernel.hpp"

namespace koishi
{
namespace poly
{
template <typename T>
struct object;

template <typename T, typename... Args>
object<T> &&make_object( Args &&... args );

template <typename T, typename U>
KOISHI_HOST_DEVICE object<T> &&static_object_cast( object<U> &&other );

template <typename T>
struct object final : emittable<object<T>>
{
	using value_type = T;
	using reference = T &;
	using const_reference = const T &;
	using pointer = T *;
	using const_pointer = const T *;

	template <typename U, typename... Args>
	friend object<U> &&make_object( Args &&... args );

	template <typename V, typename U>
	friend KOISHI_HOST_DEVICE object<V> &&static_object_cast( object<U> &&other );

	template <typename U>
	friend struct object;

public:
	object() = default;
	object( object &&other ) :
	  value( other.value ),
	  preserved( other.preserved ),
	  alloc_size( other.alloc_size ),
	  is_device_ptr( other.is_device_ptr )
	{
		other.value = nullptr;
	}
	template <typename U, typename = typename std::enable_if<std::is_base_of<T, U>::value>::type>
	object( object<U> &&other ) :
	  value( static_cast<T *>( other.value ) ),
	  preserved( static_Cast<T *>( other.preserved ),
	  alloc_size( other.alloc_size ),
	  is_device_ptr( other.is_device_ptr )
	{
		other.value = nullptr;
	}
	object &operator=( object &&other )
	{
		destroy();
		value = other.value;
		preserved = other.preserved;
		alloc_size = other.alloc_size;
		is_device_ptr = other.is_device_ptr;
		other.value = nullptr;
		return *this;
	}
	template <typename U, typename = typename std::enable_if<std::is_base_of<T, U>::value>::type>
	object &operator=( object<U> &&other )
	{
		destroy();
		value = static_cast<T *>( other.value );
		preserved = static_cast<T *>( other.preserved );
		alloc_size = other.alloc_size;
		is_device_ptr = other.is_device_ptr;
		other.value = nullptr;
		return *this;
	}

	object( const object &other )
#ifdef KOISHI_USE_CUDA
	  :
	  value( other.value ),
	  preserved( other.preserved ),
	  alloc_size( other.alloc_size ),
	  is_device_ptr( other.is_device_ptr )
	{
		copyBetweenDevice( other );
	}
#else
	{
		KTHROW( invalid use of object( const & ) );
	}
#endif
	object &operator=( const object &other )
#ifdef KOISHI_USE_CUDA
	{
		value = other.value;
		preserved = other.preserved;
		alloc_size = other.alloc_size;
		is_device_ptr = other.is_device_ptr;
		copyBetweenDevice( other );
		return *this;
	}
#else
	{
		KTHROW( invalid use of object( const & ) );
	}
#endif

	~object()
	{
		destroy();
	}

private:
#ifdef KOISHI_USE_CUDA
	void copyBetweenDevice( const object &other )
	{
		if ( !__impl::Emittable::isTransferring() )
		{
			KTHROW( invalid use of object( const & ) );
		}
		pointer new_ptr;
		if ( is_device_ptr )
		{
			new_ptr = preserved;
			__impl::Mover<T>::device_to_host( preserved, new_ptr, value );
		}
		else
		{
			if ( auto err = cudaMalloc( &new_ptr, alloc_size ) )
			{
				KTHROW( cudaMalloc on device failed );
			}
			__impl::Mover<T>::host_to_device( value, new_ptr, value );
		}
		preserved = value;
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
//#ifdef KOISHI_USE_CUDA
//				__impl::Destroyer<T>::destroy_device( value );
//				cudaFree( value );
//#else
				KTHROW( invalid internal state );
//#endif
			}
			else
			{
				__impl::Destroyer<T>::destroy_host( value );
				std::free( value );
			}
		}
	}

public:
	KOISHI_HOST_DEVICE pointer operator->() { return value; }
	KOISHI_HOST_DEVICE const_pointer operator->() const { return value; }

	KOISHI_HOST_DEVICE reference operator*() { return *value; }
	KOISHI_HOST_DEVICE const_reference operator*() const { return *value; }

private:
	T *KOISHI_RESTRICT value = nullptr;
	T *preserved;
	std::size_t alloc_size = 0;
	bool is_device_ptr = false;
};

template <typename T, typename... Args>
object<T> &&make_object( Args &&... args )
{
	static typename std::aligned_storage<sizeof( object<T> ),
										 alignof( object<T> )>::type buffer;
	static object<T> &ptr = reinterpret_cast<object<T> &>( buffer );
	new ( &ptr ) object<T>;
	auto val = (T *)std::malloc( sizeof( T ) );
	new ( val ) T( std::forward<Args>( args )... );
	ptr.value = val;
	ptr.alloc_size = sizeof( T );
	return std::move( ptr );
}

template <typename T, typename U>
KOISHI_HOST_DEVICE object<T> &&static_object_cast( object<U> &&other )
{
	static typename std::aligned_storage<sizeof( object<T> ),
										 alignof( object<T> )>::type buffer;
	static object<T> &ptr = reinterpret_cast<object<T> &>( buffer );
	new ( &ptr ) object<T>;
	ptr.value = static_cast<T *>( other.value );
	ptr.alloc_size = other.alloc_size;
	ptr.is_device_ptr = other.is_device_ptr;
	other.value = nullptr;
	return std::move( ptr );
}

}  // namespace poly

}  // namespace koishi
