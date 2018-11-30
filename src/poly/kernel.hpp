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

template <typename T, typename = void>
struct MoverImpl;

#endif

struct Emittable
{
	virtual ~Emittable() = default;

#ifdef KOISHI_USE_CUDA

	template <typename... Args>
	friend struct arguments;

	template <typename T, typename>
	friend struct MoverImpl;

#endif

protected:
	static bool &isTransferring()
	{
		static bool is_transferring = false;
		return is_transferring;
	}

protected:
	virtual void __copy_construct( void *dst, const void *src, uint count ) const = 0;
	virtual void __move_construct( void *dst, void *src, uint count ) const = 0;
};

#ifdef KOISHI_USE_CUDA

template <typename T>
__global__ inline void move_construct( T *dest, T *src )
{
	auto idx = threadIdx.x;
	new ( dest + idx ) T( std::move( src[ idx ] ) );
}

template <typename T, typename>
struct MoverImpl
{
	static void mvc( T *preserved, T *dst, T *src, uint count )
	{
		move_construct<<<1, count>>>( dst, src );
		cudaDeviceSynchronize();
	}

	static void cc( T *preserved, T *dst, const T *src, uint count )
	{
		auto q = dst;
		auto p = src;
		for ( ; p != src + count; ++p, ++q )
		{
			new ( q ) T( *p );
		}
	}
};

template <typename T>
struct MoverImpl<T, typename std::enable_if<std::is_base_of<Emittable, T>::value>::type>
{
	static void mvc( T *preserved, T *dst, T *src, uint count )
	{
		preserved->__move_construct( dst, src, count );
	}

	static void cc( T *preserved, T *dst, const T *src, uint count )
	{
		preserved->__copy_construct( dst, src, count );
	}
};

template <typename T>
struct Mover : MoverImpl<T>
{
	static void union_to_device( T *preserved, T *device_ptr, T *union_ptr, uint count = 1 )
	{
		MoverImpl<T>::mvc( preserved, device_ptr, union_ptr, count );
	}

	static void device_to_union( T *preserved, T *union_ptr, T *device_ptr, uint count = 1 )
	{
		MoverImpl<T>::mvc( preserved, union_ptr, device_ptr, count );
	}

	static void union_to_host( T *preserved, T *host_ptr, T *union_ptr, uint count = 1 )
	{
		MoverImpl<T>::cc( preserved, host_ptr, union_ptr, count );
	}

	static void host_to_union( T *preserved, T *union_ptr, T *host_ptr, uint count = 1 )
	{
		MoverImpl<T>::cc( preserved, union_ptr, host_ptr, count );
	}

	static void host_to_device( T *preserved, T *device_ptr, T *host_ptr, uint count = 1 )
	{
		KLOG3( "move from host to device", typeid( T ).name(), count );
		if ( !count ) return;
		std::size_t alloc_size = sizeof( T ) * count;
		if ( !std::is_pod<T>::value )
		{
			// #if __GNUC__ == 4 && __GNUC_MINOR__ < 9
			// 		if ( !std::is_standard_layout<T>::value )
			// #else
			// 		if ( !std::is_trivially_copyable<T>::value )
			// #endif
			// 		{
			KLOG3( "using non-trival copy for class", typeid( T ).name() );
			T *union_ptr;
			if ( auto err = cudaMallocManaged( &union_ptr, alloc_size ) )
			{
				KTHROW( cudaMallocManaged failed );
			}
			host_to_union( preserved, union_ptr, host_ptr, count );
			union_to_device( preserved, device_ptr, union_ptr, count );
			cudaFree( union_ptr );
			// }
		}
		else
		{
			KLOG3( "using plain copy for class", typeid( T ).name() );
			if ( auto err = cudaMemcpy( device_ptr, host_ptr, alloc_size, cudaMemcpyHostToDevice ) )
			{
				KTHROW( cudaMemcpy to device failed );
			}
		}
	}

	static void device_to_host( T *preserved, T *host_ptr, T *device_ptr, uint count = 1 )
	{
		KLOG3( "move from device to host", typeid( T ).name(), count );
		if ( !count ) return;
		std::size_t alloc_size = sizeof( T ) * count;
		if ( !std::is_pod<T>::value )
		{
			// #if __GNUC__ == 4 && __GNUC_MINOR__ < 9
			// 			if ( !std::is_standard_layout<T>::value )
			// #else
			// 			if ( !std::is_trivially_copyable<T>::value )
			// #endif
			// 			{
			KLOG3( "using non-trival copy for class", typeid( T ).name() );
			T *union_ptr;
			if ( auto err = cudaMallocManaged( &union_ptr, alloc_size ) )
			{
				KTHROW( cudaMallocManaged failed );
			}
			device_to_union( preserved, union_ptr, device_ptr, count );
			union_to_host( preserved, host_ptr, union_ptr, count );
			cudaFree( union_ptr );
			// }
		}
		else
		{
			KLOG3( "using plain copy for class", typeid( T ).name() );
			if ( auto err = cudaMemcpy( host_ptr, device_ptr, alloc_size, cudaMemcpyDeviceToHost ) )
			{
				KTHROW( cudaMemcpy to host failed );
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
		KLOG3( "destroying device objects", typeid( T ).name(), count );
		std::size_t alloc_size = sizeof( T ) * count;
		if ( !std::is_pod<T>::value )
		{
			T *union_ptr;
			if ( auto err = cudaMallocManaged( &union_ptr, alloc_size ) )
			{
				KTHROW( cudaMallocManaged failed );
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
		cudaMallocManaged( &data, sizeof( value_type ) );
		Emittable::isTransferring() = true;
		Mover<value_type>::host_to_union( data, param_ptr );
		Emittable::isTransferring() = false;
	}
	~arguments()
	{
		Emittable::isTransferring() = true;
		Mover<value_type>::union_to_host( param_ptr, data );
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
		Mover<value_type>::host_to_union( data, param_ptr );
		Emittable::isTransferring() = false;
	}
	~arguments()
	{
		Emittable::isTransferring() = true;
		Mover<value_type>::union_to_host( param_ptr, data );
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

template <typename T>
using mvfn = void ( * )( T *, T * );

template <typename T>
__global__ inline void move_construct_fnptr( mvfn<T> mv, T *dst, T *src )
{
	auto idx = threadIdx.x;
	mv( dst + idx, src + idx );
}

template <typename T>
__global__ inline void get_move_function( mvfn<T> *fn )
{
	*fn = T::move_constructor;
}

template <typename T>
struct Mvfn
{
	static mvfn<T> &value()
	{
		static mvfn<T> fn;
		return fn;
	}
};

#endif

}  // namespace __impl

#ifdef KOISHI_USE_CUDA

using __impl::kernel;

#endif

template <typename Type, typename Base = __impl::Emittable>
struct emittable : public Base
{
	static_assert( std::is_base_of<__impl::Emittable, Base>::value,
				   "'emittable' must derive from a emittable type" );

#ifdef KOISHI_USE_CUDA
	template <typename T>
	friend __global__ inline void __impl::get_move_function( __impl::mvfn<T> *fn );
#endif

private:
#ifdef KOISHI_USE_CUDA
	KOISHI_HOST_DEVICE static void move_constructor( Type *dst, Type *src )
	{
		new ( dst ) Type( std::move( *src ) );
	}
	static int getMvfn()
	{
		static volatile int k = [&] {
			__impl::mvfn<Type> *p;
			cudaMallocManaged( &p, sizeof( p ) );
			__impl::get_move_function<Type><<<1, 1>>>( p );
			cudaDeviceSynchronize();
			__impl::Mvfn<Type>::value() = *p;
			cudaFree( p );
			return 0;
		}();
		return 0;
	}
#endif
	void __move_construct( void *dst, void *src, uint count ) const override
	{
#ifdef KOISHI_USE_CUDA
		static volatile int invoke = getMvfn();
		auto d = static_cast<Type *>( dst );
		auto s = static_cast<Type *>( src );
		__impl::move_construct_fnptr<<<1, count>>>( __impl::Mvfn<Type>::value(), d, s );
		cudaDeviceSynchronize();
#endif
	}
	void __copy_construct( void *dst, const void *src, uint count ) const override
	{
#ifdef KOISHI_USE_CUDA
		auto d = static_cast<Type *>( dst );
		auto s = static_cast<const Type *>( src );
		auto q = d;
		auto p = s;
		for ( ; p != s + count; ++p, ++q )
		{
			new ( q ) Type( *p );
		}
#endif
	}
};

template <typename Type, typename Base = __impl::Emittable>
struct abstract_emittable : public Base
{
	static_assert( std::is_base_of<__impl::Emittable, Base>::value,
				   "'emittable' must derive from a emittable type" );
};

}  // namespace poly

using poly::abstract_emittable;
using poly::emittable;

}  // namespace koishi
