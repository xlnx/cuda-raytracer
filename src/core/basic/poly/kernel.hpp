#pragma once

#include <utility>
#include <type_traits>
#include <vec/trait.hpp>
#include <vec/vmath.hpp>

#include "debug.hpp"

namespace koishi
{
namespace core
{
namespace __impl
{
#ifdef KOISHI_USE_CUDA

template <typename... Args>
struct arguments;

#endif

struct Emittable
{
#ifdef KOISHI_USE_CUDA

	template <typename... Args>
	friend struct arguments;

#endif

#ifdef KOISHI_USE_CUDA
	void emit()
	{
		isTransferring() = true;
		__copy_construct();
		isTransferring() = false;
	}
	void fetch()
	{
		isTransferring() = true;
		__copy_construct();
		isTransferring() = false;
	}

#endif
protected:
	static bool &isTransferring()
	{
		static bool is_transferring = false;
		return is_transferring;
	}

protected:
	KOISHI_HOST_DEVICE virtual void __copy_construct() = 0;
	KOISHI_HOST_DEVICE virtual void __move_construct() = 0;
};

#ifdef KOISHI_USE_CUDA

template <typename T>
__global__ inline void move_construct( T *dest, T *src )
{
	//LOG("move_construct()", typeid(T).name(), dest, src);
	auto idx = threadIdx.x;
	new ( dest + idx ) T( std::move( src[ idx ] ) );
}

template <typename T>
struct Mover
{
	static void union_to_device( T *device_ptr, T *union_ptr, uint count = 1 )
	{
		move_construct<<<1, count>>>( device_ptr, union_ptr );
		cudaDeviceSynchronize();
	}

	static void device_to_union( T *union_ptr, T *device_ptr, uint count = 1 )
	{
		move_construct<<<1, count>>>( union_ptr, device_ptr );
		cudaDeviceSynchronize();
	}

	static void union_to_host( T *host_ptr, T *union_ptr, uint count = 1 )
	{
		for ( auto q = host_ptr, p = union_ptr; p != union_ptr + count; ++p, ++q )
		{
			new ( q ) T( static_cast<const T &>( *p ) );
		}
	}

	static void host_to_union( T *union_ptr, T *host_ptr, uint count = 1 )
	{
		for ( auto q = host_ptr, p = union_ptr; p != union_ptr + count; ++p, ++q )
		{
			new ( p ) T( static_cast<const T &>( *q ) );
		}
	}

	static void host_to_device( T *device_ptr, T *host_ptr, uint count = 1 )
	{
		LOG( "move from host to device", typeid( T ).name(), count );
		std::size_t alloc_size = sizeof( T ) * count;
		if ( !std::is_pod<T>::value )
		{
			// #if __GNUC__ == 4 && __GNUC_MINOR__ < 9
			// 		if ( !std::is_standard_layout<T>::value )
			// #else
			// 		if ( !std::is_trivially_copyable<T>::value )
			// #endif
			// 		{
			LOG( "using non-trival copy for class", typeid( T ).name() );
			T *union_ptr;
			if ( auto err = cudaMallocManaged( &union_ptr, alloc_size ) )
			{
				THROW( cudaMallocManaged failed );
			}
			host_to_union( union_ptr, host_ptr, count );
			union_to_device( device_ptr, union_ptr, count );
			cudaFree( union_ptr );
			// }
		}
		else
		{
			LOG( "using plain copy for class", typeid( T ).name() );
			if ( auto err = cudaMemcpy( device_ptr, host_ptr, alloc_size, cudaMemcpyHostToDevice ) )
			{
				THROW( cudaMemcpy to device failed );
			}
		}
	}

	static void device_to_host( T *host_ptr, T *device_ptr, uint count = 1 )
	{
		LOG( "move from device to host", typeid( T ).name(), count );
		std::size_t alloc_size = sizeof( T ) * count;
		if ( !std::is_pod<T>::value )
		{
			// #if __GNUC__ == 4 && __GNUC_MINOR__ < 9
			// 			if ( !std::is_standard_layout<T>::value )
			// #else
			// 			if ( !std::is_trivially_copyable<T>::value )
			// #endif
			// 			{
			LOG( "using non-trival copy for class", typeid( T ).name() );
			T *union_ptr;
			if ( auto err = cudaMallocManaged( &union_ptr, alloc_size ) )
			{
				THROW( cudaMallocManaged failed );
			}
			device_to_union( union_ptr, device_ptr, count );
			union_to_host( host_ptr, union_ptr, count );
			cudaFree( union_ptr );
			// }
		}
		else
		{
			LOG( "using plain copy for class", typeid( T ).name() );
			if ( auto err = cudaMemcpy( host_ptr, device_ptr, alloc_size, cudaMemcpyDeviceToHost ) )
			{
				THROW( cudaMemcpy to host failed );
			}
		}
	}
};

#endif

template <typename T>
struct Destroyer
{
#ifdef KOISHI_USE_CUDA
	static void destroy_device( T *device_ptr, uint count = 1 )
	{
		LOG( "destroying device objects", typeid( T ).name(), count );
		std::size_t alloc_size = sizeof( T ) * count;
		if ( !std::is_pod<T>::value )
		{
			T *union_ptr;
			if ( auto err = cudaMallocManaged( &union_ptr, alloc_size ) )
			{
				THROW( cudaMallocManaged failed );
			}
			Mover<T>::device_to_union( union_ptr, device_ptr, count );
			for ( auto p = union_ptr; p != union_ptr + count; ++p )
			{
				p->~T();
			}
			cudaFree( union_ptr );
		}
	}
#endif

	static void destroy_host( T *host_ptr, uint count = 1 )
	{
		for ( auto p = host_ptr; p != host_ptr + count; ++p )
		{
			p->~T();
		}
	}
};

#ifdef KOISHI_USE_CUDA

template <typename T, typename... Args>
struct arguments<T, Args...> : arguments<Args...>
{
	using value_type = typename std::remove_reference<
	  typename std::remove_cv<T>::type>::type;

	arguments( T &&x, Args &&... args ) :
	  arguments<Args...>( std::forward<Args>( args )... ),
	  param_ptr( const_cast<value_type *>( &x ) )
	{
		cudaMalloc( &data, sizeof( value_type ) );
		Emittable::isTransferring() = true;
		Mover<value_type>::host_to_device( data, param_ptr );
		Emittable::isTransferring() = false;
	}
	~arguments()
	{
		Emittable::isTransferring() = true;
		Mover<value_type>::device_to_host( param_ptr, data );
		Emittable::isTransferring() = false;
		cudaFree( data );
	}

	template <typename X, std::size_t N, std::size_t Id>
	const X &forward()
	{
		LOG( "forwarding", typeid( X ).name() );
		static_assert( N - Id - 1 != sizeof...( Args ) ||
						 std::is_same<X, T>::value,
					   "wrong parameter type" );
		if ( N - Id - 1 == sizeof...( Args ) )
		{
			return reinterpret_cast<const X &>( *data );
		}
		else
		{
			return arguments<Args...>::template forward<X, N, Id>();
		}
	}

private:
	value_type *data, *param_ptr;
};

template <typename T>
struct arguments<T>
{
	using value_type = typename std::remove_reference<
	  typename std::remove_cv<T>::type>::type;

	arguments( const T &x ) :
	  param_ptr( const_cast<value_type *>( &x ) )
	{
		cudaMalloc( &data, sizeof( value_type ) );
		Emittable::isTransferring() = true;
		Mover<value_type>::host_to_device( data, param_ptr );
		Emittable::isTransferring() = false;
	}
	~arguments()
	{
		Emittable::isTransferring() = true;
		Mover<value_type>::device_to_host( param_ptr, data );
		Emittable::isTransferring() = false;
		cudaFree( data );
	}

	template <typename X, std::size_t N, std::size_t Id>
	const X &forward()
	{
		LOG( "forwarding", typeid( X ).name() );
		return reinterpret_cast<const X &>( *data );
	}

private:
	value_type *data, *param_ptr;
};

template <std::size_t... Is>
struct indices
{
	using type = indices<Is...>;
};

template <std::size_t N, std::size_t... Is>
struct build_indices : build_indices<N - 1, N - 1, Is...>
{
};

template <std::size_t... Is>
struct build_indices<0, Is...> : indices<Is...>
{
};

template <typename F>
struct callable;

template <typename... Args>
struct callable<void ( * )( Args... )>
{
	using F = void ( * )( Args... );

	callable( F f, int a, int b, int c ) :
	  f( f ), a( a ), b( b ), c( c )
	{
	}
	template <typename... Given>
	void operator()( Given &&... given )
	{
		do_call( build_indices<sizeof...( Given )>{}, std::forward<Given>( given )... );
	}

private:
	template <typename... Given, std::size_t... Is>
	void do_call( indices<Is...>, Given &&... given )
	{
		LOG( "calling" );
		arguments<Given...> argval( std::forward<Given>( given )... );
		f<<<a, b, c>>>( argval.template forward<Given, sizeof...( Given ), Is>()... );
		cudaDeviceSynchronize();
	}

private:
	F f;
	int a, b, c;
};

template <typename F>
callable<F> kernel( F f, int a, int b, int c = 0 )
{
	return callable<F>( f, a, b, c );
}

#endif

}  // namespace __impl

#ifdef KOISHI_USE_CUDA

using __impl::kernel;

#endif

template <typename T>
struct Emittable : __impl::Emittable
{
private:
	KOISHI_HOST_DEVICE void __copy_construct() override
	{
		auto p = static_cast<T *>( this );
		new ( p ) T( *p );
	}
	KOISHI_HOST_DEVICE void __move_construct() override
	{
		auto p = static_cast<T *>( this );
		new ( p ) T( *p );
	}
};

}  // namespace core

}  // namespace koishi