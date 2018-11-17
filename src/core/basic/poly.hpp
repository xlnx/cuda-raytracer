#pragma once

#include <new>
#include <memory>
#include <cstdlib>
#include <utility>
#include <type_traits>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <vec/trait.hpp>
#include <vec/vmath.hpp>

//#define KOISHI_DEBUG
#ifdef KOISHI_DEBUG
#define LOG( ... ) println( __VA_ARGS__ )
#else
#define LOG( ... )
#endif
#define THROW( ... ) throw std::logic_error( #__VA_ARGS__ )

namespace koishi
{
namespace core
{
struct Host;
struct Device;

struct Host
{
	template <typename F, typename... Args>
	KOISHI_HOST static auto call( Args &&... args )
	{
		return F::__priv::template func<F, Host, Host, Device>::fn( std::forward<Args>( args )... );
	}
};

struct Device
{
	template <typename F, typename... Args>
	KOISHI_DEVICE static auto call( Args &&... args )
	{
		return F::__priv::template func<F, Device, Host, Device>::fn( std::forward<Args>( args )... );
	}
};

inline void println()
{
	std::cout << std::endl
			  << std::flush;
}

template <typename X, typename... Args>
void println( const X &x, Args &&... args )
{
	std::cout << x << " ";
	println( std::forward<Args>( args )... );
}

namespace trait
{
struct dummy
{
};

template <typename T, typename = void>
struct is_host_callable : std::integral_constant<bool, false>
{
};
template <typename T>
struct is_host_callable<T, typename std::enable_if<
							 std::is_base_of<Host, T>::value>::type> : std::integral_constant<bool, true>
{
};

template <typename T, typename = void>
struct is_device_callable : std::integral_constant<bool, false>
{
};
template <typename T>
struct is_device_callable<T, typename std::enable_if<
							   std::is_base_of<Device, T>::value>::type> : std::integral_constant<bool, true>
{
};

template <bool X, bool... Args>
struct make_and : std::integral_constant<bool, X && make_and<Args...>::value>
{
};
template <bool X>
struct make_and<X> : std::integral_constant<bool, X>
{
};

}  // namespace trait

template <typename... Args>
struct Require : std::conditional<trait::make_and<
									std::is_base_of<Host, Args>::value...>::value,
								  Host, trait::dummy>::type,
				 std::conditional<trait::make_and<
									std::is_base_of<Device, Args>::value...>::value,
								  Device, trait::dummy>::type
{
};

#define __PolyFunctionImpl( ... )                                                                                          \
	struct __priv                                                                                                          \
	{                                                                                                                      \
		template <typename _M_Self, typename _M_T, typename _M_Host, typename _M_Device>                                   \
		struct func;                                                                                                       \
		template <typename _M_Self, typename _M_Host, typename _M_Device>                                                  \
		struct func<_M_Self, _M_Host, _M_Host, _M_Device>                                                                  \
		{                                                                                                                  \
			static_assert( std::is_base_of<Host, _M_Self>::value, "this function is not callable on host" );               \
			using call_type = Host;                                                                                        \
			template <typename _M_F, typename... _M_Args>                                                                  \
			KOISHI_HOST static auto call( _M_Args &&... args )                                                             \
			{                                                                                                              \
				return _M_F::__priv::template func<_M_F, call_type, Host, Device>::fn( std::forward<_M_Args>( args )... ); \
			}                                                                                                              \
			KOISHI_HOST static auto fn __VA_ARGS__                                                                         \
		};                                                                                                                 \
		template <typename _M_Self, typename _M_Host, typename _M_Device>                                                  \
		struct func<_M_Self, _M_Device, _M_Host, _M_Device>                                                                \
		{                                                                                                                  \
			static_assert( std::is_base_of<Device, _M_Self>::value, "this function is not callable on device" );           \
			using call_type = Device;                                                                                      \
			template <typename _M_F, typename... _M_Args>                                                                  \
			KOISHI_DEVICE static auto call( _M_Args &&... args )                                                           \
			{                                                                                                              \
				return _M_F::__priv::template func<_M_F, call_type, Host, Device>::fn( std::forward<_M_Args>( args )... ); \
			}                                                                                                              \
			KOISHI_DEVICE static auto fn __VA_ARGS__                                                                       \
		};                                                                                                                 \
		template <typename _M_T>                                                                                           \
		struct return_type_of;                                                                                             \
		template <typename _M_T, typename... _M_Args>                                                                      \
		struct return_type_of<_M_T( _M_Args... )>                                                                          \
		{                                                                                                                  \
			using type = _M_T;                                                                                             \
		};                                                                                                                 \
	};                                                                                                                     \
	}

#define PolyFunction( name, ... ) \
	struct name : __VA_ARGS__     \
	{                             \
		using Self = name;        \
		__PolyFunctionImpl

template <typename T>
struct PolyVectorView;

template <typename T>
struct PolyVector final
{
	friend class PolyVectorView<T>;

	using value_type = T;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using reference = T &;
	using const_reference = const T &;
	using pointer = T *;
	using const_pointer = const T *;
	using iterator = T *;
	using const_iterator = const T *;

public:
	PolyVector() = default;
	PolyVector( size_type count, const T &val = T() )
	{
		resize( count, val );
	}
	PolyVector( const PolyVector &other ) :
	  total( other.total ),
	  curr( other.curr ),
	  value( (pointer)std::malloc( sizeof( T ) * other.total ) )
	{
		copy( other );
	}
	PolyVector( PolyVector &&other ) :
	  total( other.total ),
	  curr( other.curr ),
	  value( other.value )
	{
		other.value = nullptr;
	}
	PolyVector &operator=( const PolyVector &other )
	{
		destroy();
		value = (pointer)std::malloc( sizeof( T ) * other.total );
		total = other.total;
		curr = other.curr;
		copy( other );
		return *this;
	}
	PolyVector &operator=( PolyVector &&other )
	{
		destroy();
		value = other.value;
		total = other.total;
		curr = other.curr;
		other.value = nullptr;
		return *this;
	}
	~PolyVector()
	{
		destroy();
	}

public:
	reference operator[]( size_type idx ) { return value[ idx ]; }
	const_reference operator[]( size_type idx ) const { return value[ idx ]; }

	reference front() { return *value; }
	const_reference front() const { return *value; }

	reference back() { return value[ curr - 1 ]; }
	const_reference back() const { return value[ curr - 1 ]; }

	pointer data() { return value; }
	const_pointer data() const { return value; }

	iterator begin() { return value; }
	const_iterator begin() const { return value; }

	iterator end() { return value + curr; }
	const_iterator end() const { return value + curr; }

	bool empty() const { return curr == 0; }

	size_type size() const { return curr; }

	size_type capacity() const { return total; }

public:
	template <typename... Args>
	void emplace_back( Args &&... args )
	{
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
			destroy();
			value = (pointer)std::malloc( sizeof( T ) * count );
			total = curr = count;
			for ( auto p = value; p != value + curr; ++p )
			{
				new ( p ) T( val );
			}
		}
	}

private:
	void destroy()
	{
		if ( value != nullptr )
		{
			for ( auto p = value; p != value + curr; ++p )
			{
				p->~T();
			}
			std::free( value );
		}
	}

	void copy( const PolyVector &other )
	{
		for ( auto p = value, q = other.value; p != value + curr; ++p, ++q )
		{
			new ( p ) T( *q );
		}
	}

private:
	std::size_t total = 4;
	std::size_t curr = 0;
	T *value = (pointer)std::malloc( sizeof( T ) * total );
};

namespace __impl
{
struct Emittable
{
protected:
	Emittable() = default;
	virtual ~Emittable() = default;
	KOISHI_HOST_DEVICE Emittable( Emittable &&other ) = default;
	KOISHI_HOST_DEVICE Emittable &operator=( Emittable &&other ) = default;
	Emittable( const Emittable &other ) :
	  is_device_ptr( !other.is_device_ptr )
	{
	}
	Emittable &operator=( const Emittable &other )
	{
		is_device_ptr = !other.is_device_ptr;
		return *this;
	}

public:
#ifdef KOISHI_USE_CUDA
	void emit()
	{
		LOG( "emit&replacing", this );
		if ( is_device_ptr )
		{
			THROW( unable to emit device ptr );
		}
		isTransferring() = true;
		__copy_construct();
		isTransferring() = false;
	}
	void fetch()
	{
		LOG( "fetch&replacing", this );
		if ( !is_device_ptr )
		{
			THROW( unable to fetch host ptr );
		}
		isTransferring() = true;
		__copy_construct();
		isTransferring() = false;
	}
	KOISHI_HOST_DEVICE bool space() const
	{
		return is_device_ptr;
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

	bool is_device_ptr = false;
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
			pointer union_ptr;
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
			pointer union_ptr;
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
			pointer union_ptr;
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

template <typename... Args>
struct arguments;

template <typename T, typename... Args>
struct arguments<T, Args...> : arguments<Args...>
{
	using value_type = typename std::remove_reference<
	  typename std::remove_cv<T>::type>::type;

	arguments( T &&x, Args &&... args ) :
	  arguments<Args...>( std::forward<Args>( args )... )
	{
		cudaMalloc( &data, sizeof( value_type ) );
		Mover<value_type>::host_to_device( data, &x );
	}
	~arguments()
	{
		Mover<value_type>::device_to_host( &x, data );
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
	value_type *data;
};

template <typename T>
struct arguments<T>
{
	using value_type = typename std::remove_reference<
	  typename std::remove_cv<T>::type>::type;

	arguments( const T &x )
	{
		cudaMalloc( &data, sizeof( value_type ) );
		Mover<value_type>::host_to_device( data, &x );
	}
	~arguments()
	{
		Mover<value_type>::device_to_host( &x, data );
		cudaFree( data );
	}

	template <typename X, std::size_t N, std::size_t Id>
	const X &forward()
	{
		LOG( "forwarding", typeid( X ).name() );
		return reinterpret_cast<const X &>( *data );
	}

private:
	value_type *data;
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
		LOG( "calling" );
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
struct Emittable : virtual __impl::Emittable
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
	PolyVectorView() :
	  value( nullptr ),
	  curr( 0 )
	{
	}
	PolyVectorView( size_type count ) :
	  value( (pointer)std::malloc( sizeof( T ) * count ) ),
	  curr( count )
	{
	}
	KOISHI_HOST_DEVICE PolyVectorView( PolyVectorView &&other ) :
	  __impl::Emittable( std::forward<PolyVectorView>( other ) ),
	  value( other.value ),
	  curr( other.curr )
	{
		other.value = nullptr;
	}
	KOISHI_HOST_DEVICE PolyVectorView &operator=( PolyVectorView &&other )
	{
		destroy();
		__impl::Emittable::operator=( std::forward<PolyVectorView>( other ) );
		value = other.value;
		curr = other.curr;
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
	  __impl::Emittable( other )
	{
		if ( !__impl::Emittable::isTransferring() )
		{
			THROW( invalid use of PolyVectorView( const & ) );
		}
		copyBetweenDevice();
	}
#else
	{
		THROW( invalid use of PolyVectorView( const & ) );
	}
#endif
	PolyVectorView &operator=( const PolyVectorView &other )
#ifdef KOISHI_USE_CUDA
	{
		__impl::Emittable::operator=( other );
		if ( !__impl::Emittable::isTransferring() )
		{
			THROW( invalid use of PolyVectorView( const & ) );
		}
		copyBetweenDevice();
		return *this;
	}
#else
	{
		THROW( invalid use of PolyVectorView( const & ) );
	}
#endif

private:
#ifdef KOISHI_USE_CUDA
	void copyBetweenDevice( const PolyVectorView & )
	{
		pointer new_ptr;
		auto alloc_size = sizeof( T ) * curr;
		else if ( !this->is_device_ptr )
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
		destroy();
		value = new_ptr;
		LOG( "value", value );
	}

#endif
	void destroy()
	{
		if ( value != nullptr )
		{
			LOG( "destroy()", typeid( T ).name(), this );
			if ( this->is_device_ptr )
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
		this->is_device_ptr = false;
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
	T *value;
	size_type curr;
};

// template <typename T>
// struct PolyPtr final
// {
// 	using element_type T;

// 	// PolyPtr()

// public:
// 	KOISHI_HOST_DEVICE element_type &operator*() { return *ptr; }
// 	KOISHI_HOST_DEVICE element_type const &operator*() const { return *ptr; }

// 	KOISHI_HOST_DEVICE element_type *operator->() { return ptr; }
// 	KOISHI_HOST_DEVICE element_type const *operator->() const { return ptr; }

// 	KOISHI_HOST_DEVICE element_type *get() { return ptr; }
// 	KOISHI_HOST_DEVICE element_type const *get() const { return ptr; }

// 	KOISHI_HOST_DEVICE bool operator==( const PolyPtr &other ) const { return ptr == other.ptr; }
// 	KOISHI_HOST_DEVICE bool operator!=( const PolyPtr &other ) const { return ptr != other.ptr; }
// 	KOISHI_HOST_DEVICE bool operator<( const PolyPtr &other ) const { return ptr < other.ptr; }
// 	KOISHI_HOST_DEVICE bool operator<=( const PolyPtr &other ) const { return ptr <= other.ptr; }
// 	KOISHI_HOST_DEVICE bool operator>( const PolyPtr &other ) const { return ptr > other.ptr; }
// 	KOISHI_HOST_DEVICE bool operator>=( const PolyPtr &other ) const { return ptr >= other.ptr; }

// 	KOISHI_HOST_DEVICE operator bool() const { return ptr; }

// private:
// 	element_type *ptr == nullptr;
// };

}  // namespace core

}  // namespace koishi
