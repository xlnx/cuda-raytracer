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
template <typename T>
struct PolyStructImpl;

template <typename T, typename = void>
struct copyConstructor;

template <typename T>
struct Emittable
{
public:
	T emit() const
	{
		if ( is_device_ptr )
		{
			THROW( unable to emit device ptr );
			//			throw ::std::bad_alloc();
		}
		typename std::aligned_storage<sizeof( T ), alignof( T )>::type mem;
		auto ptr = reinterpret_cast<T *>( &mem );
		const T *self = static_cast<const T *>( self );
		copyConstructor<T>::apply( ptr, self );
		ptr->is_device_ptr = !ptr->is_device_ptr;
		return std::move( *ptr );
	}
	void emitAndReplace()
	{
		if ( is_device_ptr )
		{
			THROW( unable to emit device ptr );
			//			throw ::std::bad_alloc();
		}
		T *self = static_cast<T *>( self );
		copyConstructor<T>::apply( self, self );
		is_device_ptr = !is_device_ptr;
	}
	T fetch() const
	{
		if ( !is_device_ptr )
		{
			THROW( unable to fetch host ptr );
			//			throw ::std::bad_alloc();
		}
		typename std::aligned_storage<sizeof( T ), alignof( T )>::type mem;
		auto ptr = reinterpret_cast<T *>( &mem );
		const T *self = static_cast<const T *>( self );
		copyConstructor<T>::apply( ptr, *self );
		ptr->is_device_ptr = !ptr->is_device_ptr;
		return std::move( *ptr );
	}
	void fetchAndReplace()
	{
		if ( !is_device_ptr )
		{
			THROW( unable to fetch host ptr );
			//			throw ::std::bad_alloc();
		}
		T *self = static_cast<T *>( self );
		copyConstructor<T>::apply( self, self );
		is_device_ptr = !is_device_ptr;
	}

protected:
	bool is_device_ptr = false;
};

template <typename T>
struct PolyStructImpl : __impl::Emittable<T>
{
	template <typename U>
	friend struct PolyVectorView;
	template <typename U>
	friend struct PolyStructImpl;
	template <typename U, typename X>
	friend struct copyConstructor;

	PolyStructImpl() = default;
	KOISHI_HOST_DEVICE PolyStructImpl( PolyStructImpl && ) = default;
	KOISHI_HOST_DEVICE PolyStructImpl &operator=( PolyStructImpl && ) = default;

protected:
	PolyStructImpl( const PolyStructImpl & ) = default;
	PolyStructImpl &operator=( const PolyStructImpl & ) = default;

private:
	void copyConstruct( const T &other )
	{
	//	LOG("copyConstruct(const&)", typeid(T).name(), this);
		new ( static_cast<T *>( this ) ) T( other );
	}
};

#ifdef KOISHI_USE_CUDA

template <typename T>
__global__ void move_construct( T *dest, T *src )
{
	//LOG("move_construct()", typeid(T).name(), dest, src);
	auto idx = threadIdx.x;
	new ( dest + idx ) T( std::move( src[ idx ] ) );
}

#endif

template <typename T>
struct copyConstructor<T, typename std::enable_if<!std::is_base_of<PolyStructImpl<T>, T>::value>::type>
{
	inline static void apply( T *q, const T *p )
	{
		LOG( "copyConstructor<T>()", typeid( T ).name(), q, p );
		new ( q ) T( *p );
	}
};

template <typename T>
struct copyConstructor<T, typename std::enable_if<std::is_base_of<PolyStructImpl<T>, T>::value>::type>
{
	inline static void apply( T *q, const T *p )
	{
		LOG( "copyConstructor<PolyStruct>()", typeid( T ).name(), q, p );
		static_cast<PolyStructImpl<T> *>( q )->copyConstruct( *p );
	}
};

template <typename T>
struct Poly;

}  // namespace __impl

#define PolyStruct( type )                                   \
	__##type##_tag;                                          \
	using type = koishi::core::__impl::Poly<__##type##_tag>; \
	template <>                                              \
	struct koishi::core::__impl::Poly<__##type##_tag> : __impl::PolyStructImpl<type>

// PolyVectorView holds a read-only data vector for either cpu or gpu
// use std::move to make
template <typename T>
struct PolyVectorView final
{
	template <typename U>
	friend class __impl::Poly;
	template <typename U>
	friend class PolyVectorView;

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
	  curr( count ),
	  is_device_ptr( false )
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
		value = other.value;
		curr = other.curr;
		is_device_ptr = other.is_device_ptr;
		other.value = nullptr;
		return *this;
	}
	~PolyVectorView()
	{
		LOG( "~PolyVectorView()", typeid( T ).name(), this );
		destroy();
	}

private:
	PolyVectorView( const PolyVectorView &other )
#ifdef KOISHI_USE_CUDA
	{
		LOG( "PolyVectorView(const&)", typeid( T ).name(), this, &other );
		copyBetweenDevice( other );
	}
#else
	  = delete;
#endif
	PolyVectorView &operator=( const PolyVectorView &other )
#ifdef KOISHI_USE_CUDA
	{
		LOG( "PolyVectorView=(const&)", typeid( T ).name(), this, &other );
		copyBetweenDevice( other );
		return *this;
	}
#else
	  = delete;
#endif
#ifdef KOISHI_USE_CUDA
	void copyBetweenDevice( const PolyVectorView &other )
	{
		LOG( "copyBetweenDevice(const&)", typeid( T ).name(), this, &other, other.curr );
		auto alloc_size = sizeof( T ) * other.curr;
		if ( alloc_size == 0 )
		{
			destroy();
			value = nullptr;
			curr = other.curr;
			is_device_ptr = !other.is_device_ptr;
		}
		else if ( other.is_device_ptr )
		{
			LOG( "copy from device to host" );
			if ( &other != this )
			{
				destroy();
				value = other.value;
				curr = other.curr;
			}
			pointer host_value = (pointer)std::malloc( alloc_size );
			if ( !std::is_pod<T>::value )
			{
				LOG( "using trivial copy for class", typeid( T ).name() );
				auto buf = (pointer)std::malloc( alloc_size );
				if ( auto err = cudaMemcpy( buf, value, alloc_size, cudaMemcpyDeviceToHost ) )
				{
					THROW( cudaMemcpy to host failed );
				}
				for ( auto p = buf, q = host_value; p != buf + curr; ++p, ++q )
				{
					__impl::copyConstructor<T>::apply( q, p );
				}
				std::free( buf );  // don't call dtor because they exist on device
			}
			else
			{
				LOG( "using plain copy for class", typeid( T ).name() );
				if ( auto err = cudaMemcpy( host_value, value, alloc_size, cudaMemcpyDeviceToHost ) )
				{
					THROW( cudaMemcpy to host failed );
				}
			}
			value = host_value;
			LOG( "value", value );
			is_device_ptr = false;
		}
		else
		{
			LOG( "copy from host to device" );
			if ( &other != this )
			{
				destroy();
				value = other.value;
				curr = other.curr;
			}
			pointer device_value;
			if ( auto err = cudaMalloc( &device_value, alloc_size ) )
			{
				std::ostringstream os;
				os << err;
				throw std::logic_error( os.str() );
				THROW( cudaMalloc on device failed );
			}
			LOG( "allocated device ptr", device_value );
			if ( !std::is_pod<T>::value )
			{
#if __GNUC__ == 4 && __GNUC_MINOR__ < 9
				if ( !std::is_standard_layout<T>::value )
#else
				if ( !std::is_trivially_copyable<T>::value )
#endif
				{
					LOG( "using non-trival copy for class", typeid( T ).name() );
					pointer buf;
					if ( auto err = cudaMallocManaged( &buf, alloc_size ) )
					{
						THROW( cudaMallocManaged failed );
					}
					LOG( "allocated union ptr", buf );
					for ( auto p = value, q = buf; p != value + curr; ++p, ++q )
					{
						__impl::copyConstructor<T>::apply( q, p );
					}
					LOG( "move construct", typeid( T ).name(), device_value, buf );
					__impl::move_construct<<<1, curr>>>( device_value, buf );
					cudaDeviceSynchronize();
					cudaFree( buf );
				}
				else
				{
					LOG( "using trival copy for class", typeid( T ).name() );
					auto buf = (pointer)std::malloc( alloc_size );
					for ( auto p = value, q = buf; p != value + curr; ++p, ++q )
					{
						__impl::copyConstructor<T>::apply( q, p );
					}
					if ( auto err = cudaMemcpy( device_value, buf, alloc_size, cudaMemcpyHostToDevice ) )
					{
						THROW( cudaMemcpy to device failed );
						//						throw err;  // buf objects are constructed on host, but used on device
					}
					std::free( buf );  // don't call dtor because they exist on device
				}
			}
			else
			{
				LOG( "using plain copy for class", typeid( T ).name() );
				if ( auto err = cudaMemcpy( device_value, value, alloc_size, cudaMemcpyHostToDevice ) )
				{
					THROW( cudaMemcpy to device failed );
					//					throw err;  // buf objects are constructed on host, but used on device
				}
			}
			value = device_value;
			LOG( "value", value );
			is_device_ptr = true;  // this vector now points to device
		}
	}

#endif
	void destroy()
	{
		if ( !is_forwarded && value != nullptr )
		{
			LOG( "destroy()", typeid( T ).name(), this );
			if ( is_device_ptr )
			{
				LOG( "is device ptr", value );
#ifdef KOISHI_USE_CUDA
				auto alloc_size = sizeof( T ) * curr;
				if ( !std::is_pod<T>::value )
				{
					LOG( "not pod" );
					auto buf = (pointer)std::malloc( alloc_size );
					if ( auto err = cudaMemcpy( buf, value, alloc_size, cudaMemcpyDeviceToHost ) )
					{
						THROW( cudaMemcpy to host failed );
						//						throw err;  // buf objects are constructed on host, but used on device
					}
					for ( auto p = buf; p != buf + curr; ++p )
					{
						p->~T();
					}
					std::free( buf );
				}
				cudaFree( value );
#else
				throw ::std::bad_alloc();
#endif
			}
			else
			{
				LOG( "not device ptr", value );
				if ( !std::is_pod<T>::value )
				{
					for ( auto p = value; p != value + curr; ++p )
					{
						p->~T();
					}
				}
				std::free( value );
			}
		}
	}

private:
	struct PolyVectorViewTag
	{
		T *value;
		size_type curr;
		bool is_device_ptr;
	};

public:
	PolyVectorView emit() const
	{
		if ( is_device_ptr )
		{
			throw std::bad_alloc();
		}
		PolyVectorView dev( *this );
		return std::move( dev );
	}
	void emitAndReplace()
	{
		if ( is_device_ptr )
		{
			throw std::bad_alloc();
		}
		copyBetweenDevice( *this );
	}
	PolyVectorView fetch() const
	{
		if ( !is_device_ptr )
		{
			throw std::bad_alloc();
		}
		PolyVectorView dev( *this );
		return std::move( dev );
	}
	void fetchAndReplace()
	{
		if ( !is_device_ptr )
		{
			throw std::bad_alloc();
		}
		copyBetweenDevice( *this );
	}
	PolyVectorViewTag forward() const
	{
		return PolyVectorViewTag{ value, curr, is_device_ptr };
	}
	PolyVectorView( const PolyVectorViewTag &other ) :
	  value( other.value ),
	  curr( other.curr ),
	  is_device_ptr( other.is_device_ptr ),
	  is_forwarded( true )
	{
	}

public:
	PolyVectorView( const buffer_type &other ) = delete;
	PolyVectorView( buffer_type &&other ) :
	  value( other.value ),
	  curr( other.curr ),
	  is_device_ptr( false )
	{
		other.value = nullptr;
	}
	PolyVectorView &operator=( const buffer_type &other ) = delete;
	PolyVectorView &operator=( buffer_type &&other )
	{
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
	bool is_forwarded = false;
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