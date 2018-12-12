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

template <typename T, typename U>
KOISHI_HOST_DEVICE object<T> &&dynamic_object_cast( object<U> &&other );

namespace __impl
{
using cc_erased_t = void ( * )( Emittable *, Emittable * );

template <typename T>
inline void move_construct_erased( Emittable *dst, Emittable *src )
{
#ifdef KOISHI_USE_CUDA
	move_construct( static_cast<T *>( dst ), static_cast<T *>( src ), 1u );
#endif
}

template <typename T>
inline void copy_construct_erased( Emittable *dst, Emittable *src )
{
#ifdef KOISHI_USE_CUDA
	copy_construct( static_cast<T *>( dst ), static_cast<T *>( src ), 1u );
#endif
}

struct type_desc
{
	std::size_t alloc_size;
	cc_erased_t mvctor, cpctor;
};

template <typename T>
inline type_desc get_type_desc()
{
	return type_desc{
		sizeof( T ),
		move_construct_erased<T>,
		copy_construct_erased<T>
	};
}

#ifdef KOISHI_USE_CUDA

struct TypeErasedMover
{
	using T = Emittable;

	static void host_to_device( T *device_ptr, T *host_ptr, const type_desc &desc )
	{
		KLOG3( "move from host to device type erased" );
		T *union_ptr;
		if ( auto err = cudaMallocManaged( &union_ptr, desc.alloc_size ) )
		{
			KTHROW( "cudaMallocManaged failed" );
		}
		host_to_union( union_ptr, host_ptr, desc );
		union_to_device( device_ptr, union_ptr, desc );
		cudaFree( union_ptr );
	}

	static void device_to_host( T *host_ptr, T *device_ptr, const type_desc &desc )
	{
		KLOG3( "move from device to host type erased" );
		T *union_ptr;
		if ( auto err = cudaMallocManaged( &union_ptr, desc.alloc_size ) )
		{
			KTHROW( "cudaMallocManaged failed" );
		}
		device_to_union( union_ptr, device_ptr, desc );
		union_to_host( host_ptr, union_ptr, desc );
		cudaFree( union_ptr );
	}

private:
	static void union_to_device( T *device_ptr, T *union_ptr, const type_desc &desc )
	{
		desc.mvctor( device_ptr, union_ptr );
	}

	static void device_to_union( T *union_ptr, T *device_ptr, const type_desc &desc )
	{
		desc.mvctor( union_ptr, device_ptr );
	}

	static void union_to_host( T *host_ptr, T *union_ptr, const type_desc &desc )
	{
		desc.cpctor( host_ptr, union_ptr );
	}

	static void host_to_union( T *union_ptr, T *host_ptr, const type_desc &desc )
	{
		desc.cpctor( union_ptr, host_ptr );
	}
};

#endif

}  // namespace __impl

template <typename T>
struct object final : emittable
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

	template <typename V, typename U>
	friend KOISHI_HOST_DEVICE object<V> &&dynamic_object_cast( object<U> &&other );

	template <typename U>
	friend struct object;

public:
	object() = default;
	KOISHI_HOST_DEVICE object( object &&other ) :
	  emittable( std::move( other ) ),
	  value( other.value ),
	  device_value( other.device_value ),
	  desc( other.desc ),
	  is_device_ptr( other.is_device_ptr )
	{
		other.value = nullptr;
	}
	template <typename U, typename = typename std::enable_if<std::is_base_of<T, U>::value>::type>
	KOISHI_HOST_DEVICE object( object<U> &&other ) :
	  emittable( std::move( other ) ),
	  value( static_cast<T *>( other.value ) ),
	  device_value( static_cast<T *>( other.device_value ) ),
	  desc( other.desc ),
	  is_device_ptr( other.is_device_ptr )
	{
		other.value = nullptr;
	}
	KOISHI_HOST_DEVICE object &operator=( object &&other )
	{
		destroy();
		value = other.value;
		device_value = other.device_value;
		desc = other.desc;
		is_device_ptr = other.is_device_ptr;
		other.value = nullptr;
		return *this;
	}
	template <typename U, typename = typename std::enable_if<std::is_base_of<T, U>::value>::type>
	KOISHI_HOST_DEVICE object &operator=( object<U> &&other )
	{
		destroy();
		value = static_cast<T *>( other.value );
		device_value = static_cast<T *>( other.device_value );
		desc = other.desc;
		is_device_ptr = other.is_device_ptr;
		other.value = nullptr;
		return *this;
	}

	object( const object &other )
#ifdef KOISHI_USE_CUDA
	  :
	  emittable( other ),
	  value( other.value ),
	  device_value( other.device_value ),
	  desc( other.desc ),
	  is_device_ptr( other.is_device_ptr )
	{
		copyBetweenDevice( other );
	}
#else
	{
		KTHROW( "invalid use of object( const & )" );
	}
#endif
	object &operator=( const object &other )
#ifdef KOISHI_USE_CUDA
	{
		value = other.value;
		device_value = other.device_value;
		desc = other.desc;
		is_device_ptr = other.is_device_ptr;
		copyBetweenDevice( other );
		return *this;
	}
#else
	{
		KTHROW( "invalid use of object( const & )" );
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
			KTHROW( "invalid use of object( const & )" );
		}
		if ( is_device_ptr )
		{
			__impl::TypeErasedMover::device_to_host( value, device_value, *desc );
			cudaFree( value );
		}
		else
		{
			T *ptr;
			if ( auto err = cudaMalloc( &ptr, desc->alloc_size ) )
			{
				KTHROW( "cudaMalloc on device failed" );
			}
			device_value = ptr;
			__impl::TypeErasedMover::host_to_device( device_value, value, *desc );
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
				__impl::Destroyer<T>::destroy_host( value );
				std::free( value );
			}
		}
	}

public:
#ifdef __CUDA_ARCH__
#define KOISHI_DATA_PTR device_value
#else
#define KOISHI_DATA_PTR value
#endif
	KOISHI_HOST_DEVICE pointer operator->()
	{
		return KOISHI_DATA_PTR;
	}
	KOISHI_HOST_DEVICE const_pointer operator->() const
	{
		return KOISHI_DATA_PTR;
	}

	KOISHI_HOST_DEVICE reference operator*()
	{
		return *KOISHI_DATA_PTR;
	}
	KOISHI_HOST_DEVICE const_reference operator*() const
	{
		return *KOISHI_DATA_PTR;
	}
#undef KOISHI_DATA_PTR

private:
	static __impl::type_desc &getDesc()
	{
		static __impl::type_desc desc = __impl::get_type_desc<T>();
		return desc;
	}

private:
	T *KOISHI_RESTRICT value = nullptr;
	T *KOISHI_RESTRICT device_value;
	__impl::type_desc *desc;
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
	ptr.desc = &object<T>::getDesc();
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
	ptr.desc = other.desc;
	ptr.is_device_ptr = other.is_device_ptr;
	other.value = nullptr;
	return std::move( ptr );
}

template <typename T, typename U>
KOISHI_HOST_DEVICE object<T> &&dynamic_object_cast( object<U> &&other )
{
	static typename std::aligned_storage<sizeof( object<T> ),
										 alignof( object<T> )>::type buffer;
	static object<T> &ptr = reinterpret_cast<object<T> &>( buffer );
	new ( &ptr ) object<T>;
	if ( !( ptr.value = dynamic_cast<T *>( other.value ) ) )
	{
		KTHROW( "dynamic object cast failed" );
	}
	ptr.desc = other.desc;
	ptr.is_device_ptr = other.is_device_ptr;
	other.value = nullptr;
	return std::move( ptr );
}

}  // namespace poly

}  // namespace koishi
