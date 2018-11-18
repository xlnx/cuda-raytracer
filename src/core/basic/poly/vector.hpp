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

}  // namespace core

}  // namespace koishi