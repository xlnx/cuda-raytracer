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
#define STR( x ) _STR( x )
#define _STR( x ) #x
#define THROW( ... )                                                                     \
	{                                                                                    \
		throw std::logic_error( STR( __FILE__ ) ":" STR( __LINE__ ) ": " #__VA_ARGS__ ); \
	}                                                                                    \
	while ( 0 )

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