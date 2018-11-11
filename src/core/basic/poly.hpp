#pragma once

#include <new>
#include <memory>
#include <cstdlib>
#include <utility>
#include <type_traits>
#include <vec/trait.hpp>
#include <vec/vmath.hpp>

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
struct PolyBase;

struct ProtectedCopyable
{
	template <typename T>
	struct PolyBase;

	KOISHI_HOST_DEVICE ProtectedCopyable() = default;
	KOISHI_HOST_DEVICE ProtectedCopyable( ProtectedCopyable && ) = default;
	KOISHI_HOST_DEVICE ProtectedCopyable &operator=( ProtectedCopyable && ) = default;

protected:
	KOISHI_HOST_DEVICE ProtectedCopyable( const ProtectedCopyable & ) = default;
	KOISHI_HOST_DEVICE ProtectedCopyable &operator=( const ProtectedCopyable & ) = default;
};

}  // namespace __impl

#define PolyStruct( type )                                       \
	__##type##_tag;                                              \
	using type = koishi::core::__impl::PolyBase<__##type##_tag>; \
	template <>                                                  \
	struct koishi::core::__impl::PolyBase<__##type##_tag> : __impl::ProtectedCopyable

// PolyVectorView holds a read-only data vector for either cpu or gpu
// use std::move to make
template <typename T>
struct PolyVectorView final
{
	template <typename U>
	friend class __impl::PolyBase;

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
	KOISHI_HOST_DEVICE PolyVectorView() = default;
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
	KOISHI_HOST_DEVICE ~PolyVectorView()
	{
		destroy();
	}

private:
	KOISHI_HOST_DEVICE PolyVectorView( const PolyVectorView &other )
#ifdef KOISHI_USE_CUDA
	{
		copyToDevice( other );
	}
#else
	  = delete;
#endif
	KOISHI_HOST_DEVICE PolyVectorView &operator=( const PolyVectorView &other )
#ifdef KOISHI_USE_CUDA
	{
		copyToDevice( other );
		return *this;
	}
#else
	  = delete;
#endif
#ifdef KOISHI_USE_CUDA
	void copyToDevice( const PolyVectorView &other )
	{
		if ( other.is_device_ptr )
		{
			throw std::bad_alloc();  // only host vector can be copied to device
		}
		else
		{
			destroy();
			value = other.value;
			curr = other.curr;
			auto buf = (pointer)std::malloc( sizeof( T ) * curr );
			for ( auto p = value, q = buf; p != value + curr; ++p, ++q )
			{
				*q = *p;
			}
			pointer device_value;
			if ( auto err = cudaMalloc( &device_value, curr ) )
			{
				throw err;
			}
			if ( auto err = cudaMemcpy( device_value, buf, sizeof( T ) * curr, cudaHostToDevice ) )
			{
				throw err;  // buf objects are constructed on host, but used on device
			}
			value = device_value;
			std::free( buf );	  // don't call dtor because they exist on device
			is_device_ptr = true;  // this vector now points to device
		}
	}
#endif
	void destroy()
	{
		if ( value != nullptr )
		{
			for ( auto p = value; p != value + curr; ++p )
			{
				p->~T();
			}
			if ( is_device_ptr )
			{
#ifdef KOISHI_USE_CUDA
				cudaFree( value );
#else
				throw std::bad_alloc();  //  incorrect internal state
#endif
			}
			else
			{
				std::free( value );
			}
		}
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

private:
	T *value = nullptr;
	size_type curr = 0;
	bool is_device_ptr = false;
};

}  // namespace core

}  // namespace koishi
