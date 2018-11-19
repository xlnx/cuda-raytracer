#pragma once

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>

namespace koishi
{
namespace core
{
//#define KOISHI_DEBUG

#ifdef KOISHI_DEBUG
	#define LOG( ... ) koishi::core::println( __VA_ARGS__ )
#else
	#define LOG( ... )
#endif

#if KOISHI_DEBUG >= 1
	#define LOG1(...) LOG(__VA_ARGS__)
#else 
	#define LOG1(...)
#endif

#if KOISHI_DEBUG >= 2
	#define LOG2(...) LOG(__VA_ARGS__)
#else 
	#define LOG2(...)
#endif

#if KOISHI_DEBUG >= 3
	#define LOG3(...) LOG(__VA_ARGS__)
#else 
	#define LOG3(...)
#endif

#define STR( x ) _STR( x )
#define _STR( x ) #x
#define THROW( ... ) _THROW( __VA_ARGS__ )
#define _THROW( ... )                                                                    \
	{                                                                                    \
		throw std::logic_error( STR( __FILE__ ) ":" STR( __LINE__ ) ": " #__VA_ARGS__ ); \
	}                                                                                    \
	while ( 0 )
#ifdef KOISHI_DEBUG
#define ASSERT( ... )                                    \
	{                                                    \
		if ( !( __VA_ARGS__ ) )                          \
		{                                                \
			THROW( "assertion failed: ", #__VA_ARGS__ ); \
		}                                                \
	}                                                    \
	while ( 0 )
#else
#define ASSERT( ... )
#endif

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

}  // namespace core

}  // namespace koishi
