#pragma once

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <cassert>

namespace koishi
{
namespace core
{
//#define KOISHI_DEBUG

// #ifdef KOISHI_DEBUG
#define KLOG( ... ) koishi::core::println( __VA_ARGS__ )
// #else
// 	#define LOG( ... )
// #endif

#if KOISHI_DEBUG >= 1
#define KLOG1( ... ) KLOG( __VA_ARGS__ )
#else
#define KLOG1( ... )
#endif

#if KOISHI_DEBUG >= 2
#define KLOG2( ... ) KLOG( __VA_ARGS__ )
#else
#define KLOG2( ... )
#endif

#if KOISHI_DEBUG >= 3
#define KLOG3( ... ) KLOG( __VA_ARGS__ )
#else
#define KLOG3( ... )
#endif

#define KSTR( x ) K_STR( x )
#define K_STR( x ) #x
#define KTHROW( ... ) K_THROW( __VA_ARGS__ )
#define K_THROW( ... )                                                             \
	{                                                                              \
		throw std::logic_error( __FILE__ ":" KSTR( __LINE__ ) ": " #__VA_ARGS__ ); \
	}                                                                              \
	while ( 0 )
#ifdef KOISHI_DEBUG
#ifndef __CUDA_ARCH__
#define KASSERT( ... )                                    \
	do                                                    \
	{                                                     \
		if ( !( __VA_ARGS__ ) )                           \
		{                                                 \
			KTHROW( "assertion failed: ", #__VA_ARGS__ ); \
		}                                                 \
	} while ( 0 )
#else
#define KASSERT assert
#endif
#else
#define KASSERT( ... )
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
