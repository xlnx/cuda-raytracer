#pragma once

#include <memory>
#include <functional>
#include <utility>
#include <chrono>
#include <type_traits>
#include <vec/trait.hpp>
#include <vec/vmath.hpp>
#include <util/debug.hpp>

namespace koishi
{
namespace poly
{
namespace __impl
{
#ifdef KOISHI_USE_CUDA

template <typename... Args>
struct arguments;

#endif

struct Emittable
{
	virtual ~Emittable() = default;

#ifdef KOISHI_USE_CUDA

	template <typename... Args>
	friend struct arguments;

#endif

protected:
	static bool &isTransferring()
	{
		static bool is_transferring = false;
		return is_transferring;
	}
};

#ifdef KOISHI_USE_CUDA

template <typename T>
__global__ inline void move_construct_glob( T *dest, T *src )
{
	auto idx = threadIdx.x;
	new ( dest + idx ) T( std::move( src[ idx ] ) );
}

template <typename T>
inline void move_construct( T *dst, T *src, uint count )
{
	move_construct_glob<<<1, count>>>( dst, src );
	cudaDeviceSynchronize();
}

template <typename T>
inline void copy_construct( T *dst, T *src, uint count )
{
	T *q = dst;
	const T *p = src;
	for ( ; p != src + count; ++q, ++p )
	{
		new ( q ) T( *p );
	}
}

template <typename T>
struct Mover
{
	static void host_to_device( T *device_ptr, T *host_ptr, uint count )
	{
		KLOG3( "move from host to device", typeid( T ).name(), count );
		if ( !count ) return;
		std::size_t alloc_size = sizeof( T ) * count;
		if ( !std::is_pod<T>::value )
		{
			KLOG3( "using non-trival copy for class", typeid( T ).name() );
			T *union_ptr;
			if ( auto err = cudaMallocManaged( &union_ptr, alloc_size ) )
			{
				KTHROW( "cudaMallocManaged failed to allocate", alloc_size, "bytes on device" );
			}
			host_to_union( union_ptr, host_ptr, count );
			union_to_device( device_ptr, union_ptr, count );
			cudaFree( union_ptr );
		}
		else
		{
			KLOG3( "using plain copy for class", typeid( T ).name() );
			if ( auto err = cudaMemcpy( device_ptr, host_ptr, alloc_size, cudaMemcpyHostToDevice ) )
			{
				KTHROW( "cudaMemcpy to device failed" );
			}
		}
	}

	static void device_to_host( T *host_ptr, T *device_ptr, uint count )
	{
		KLOG3( "move from device to host", typeid( T ).name(), count );
		if ( !count ) return;
		std::size_t alloc_size = sizeof( T ) * count;
		if ( !std::is_pod<T>::value )
		{
			KLOG3( "using non-trival copy for class", typeid( T ).name() );
			T *union_ptr;
			if ( auto err = cudaMallocManaged( &union_ptr, alloc_size ) )
			{
				KTHROW( "cudaMallocManaged failed to allocate", alloc_size, "bytes on device" );
			}
			device_to_union( union_ptr, device_ptr, count );
			union_to_host( host_ptr, union_ptr, count );
			cudaFree( union_ptr );
		}
		else
		{
			KLOG3( "using plain copy for class", typeid( T ).name() );
			if ( auto err = cudaMemcpy( host_ptr, device_ptr, alloc_size, cudaMemcpyDeviceToHost ) )
			{
				KTHROW( "cudaMemcpy to host failed" );
			}
		}
	}

private:
	static void union_to_device( T *device_ptr, T *union_ptr, uint count )
	{
		move_construct( device_ptr, union_ptr, count );
	}

	static void device_to_union( T *union_ptr, T *device_ptr, uint count )
	{
		move_construct( union_ptr, device_ptr, count );
	}

	static void union_to_host( T *host_ptr, T *union_ptr, uint count )
	{
		copy_construct( host_ptr, union_ptr, count );
	}

	static void host_to_union( T *union_ptr, T *host_ptr, uint count )
	{
		copy_construct( union_ptr, host_ptr, count );
	}
};

#endif

template <typename T>
struct Destroyer
{
	static void destroy_host( T *host_ptr, uint count = 1 )
	{
		KLOG3( "destroy", host_ptr );
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
		cudaMallocManaged( &data, sizeof( value_type ) );
		Emittable::isTransferring() = true;
		Mover<value_type>::host_to_device( data, param_ptr, 1 );
		Emittable::isTransferring() = false;
	}
	~arguments()
	{
		Emittable::isTransferring() = true;
		Mover<value_type>::device_to_host( param_ptr, data, 1 );
		Emittable::isTransferring() = false;
		cudaFree( data );
	}

	template <typename X, std::size_t N, std::size_t Id>
	const X &forward()
	{
		static_assert( N - Id - 1 != sizeof...( Args ) ||
						 std::is_same<X, T>::value,
					   "wrong parameter type" );
		if ( N - Id - 1 == sizeof...( Args ) )
		{
			KLOG3( "forwarding", typeid( X ).name() );
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

struct argument_base
{
	virtual ~argument_base() = default;
};

template <typename T>
struct arguments<T> : argument_base
{
	using value_type = typename std::remove_reference<
	  typename std::remove_cv<T>::type>::type;

	arguments( const T &x ) :
	  param_ptr( const_cast<value_type *>( &x ) )
	{
		cudaMallocManaged( &data, sizeof( value_type ) );
		Emittable::isTransferring() = true;
		Mover<value_type>::host_to_device( data, param_ptr, 1 );
		Emittable::isTransferring() = false;
	}
	~arguments()
	{
		Emittable::isTransferring() = true;
		Mover<value_type>::device_to_host( param_ptr, data, 1 );
		Emittable::isTransferring() = false;
		cudaFree( data );
	}

	template <typename X, std::size_t N, std::size_t Id>
	const X &forward()
	{
		KLOG3( "forwarding", typeid( X ).name() );
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

template <typename... Args>
struct concurrent final
{
	using F = void ( * )( Args... );

	concurrent &pardo( const std::function<void()> &host_job )
	{
		host_job();
		return *this;
	}

	template <typename... Given, std::size_t... Is>
	concurrent( F f, dim3 a, dim3 b, uint c, indices<Is...>, Given &&... given ) :
	  f( f ), a( a ), b( b ), c( c )
	{
		using namespace std::chrono;
		KINFO( cuda, "Transmitting data..." );
		util::tick();
		auto argval = std::make_shared<arguments<Given...>>( std::forward<Given>( given )... );
		KINFO( cuda, "Transmission finished in", util::tick(), "seconds" );

		KINFO( cuda, "Executing cuda kernel..." );
		util::tick();
		f<<<a, b, c>>>( argval->template forward<Given, sizeof...( Given ), Is>()... );
		args = std::dynamic_pointer_cast<argument_base>( argval );
	}

	~concurrent()
	{
		cudaDeviceSynchronize();
		KINFO( cuda, "Kernel finished in", util::tick(), "seconds" );
	}

private:
	F f;
	dim3 a, b;
	uint c;
	std::shared_ptr<argument_base> args;
};

template <typename F>
struct callable;

template <typename... Args>
struct callable<void ( * )( Args... )>
{
	using F = void ( * )( Args... );

	callable( F f, dim3 a, dim3 b, uint c ) :
	  f( f ), a( a ), b( b ), c( c )
	{
	}
	template <typename... Given>
	concurrent<Args...> operator()( Given &&... given )
	{
		return concurrent<Args...>( f, a, b, c, build_indices<sizeof...( Given )>{}, std::forward<Given>( given )... );
	}

private:
	F f;
	dim3 a, b;
	uint c;
};

template <typename F>
callable<F> kernel( F f, dim3 a, dim3 b, uint c = 0u )
{
	return callable<F>( f, a, b, c );
}

#endif

}  // namespace __impl

#ifdef KOISHI_USE_CUDA

using __impl::kernel;

#endif

struct emittable : public __impl::Emittable
{
	KOISHI_HOST_DEVICE emittable() = default;
	KOISHI_HOST_DEVICE emittable( emittable && ) = default;
	KOISHI_HOST_DEVICE emittable &operator=( emittable && ) = default;
	emittable( const emittable & ) = default;
	emittable &operator=( const emittable & ) = default;
};

}  // namespace poly

using poly::emittable;

}  // namespace koishi
