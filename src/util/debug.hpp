#pragma once

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <cassert>
#include <chrono>

namespace koishi
{
namespace util
{

inline double tick()
{
	using namespace std::chrono;
	static bool is = false;
	static decltype(system_clock::now()) prev;
	if ( is = !is ) 
	{
		prev = system_clock::now();
		return 0;
	}
	else
	{
		auto now = system_clock::now();
		auto duration = duration_cast<microseconds>( now - prev );
		return double( duration.count() ) * microseconds::period::num / microseconds::period::den;
	}
}

//#define KOISHI_DEBUG

// #ifdef KOISHI_DEBUG
#define KLOG( ... ) koishi::util::println( __VA_ARGS__ )
// #else
// 	#define LOG( ... )
// #endif
//

#define KINFO(Session, ...) KLOG( "[", #Session, "] ", __VA_ARGS__ )

//#define KTICK() koishi::core::__impl::tick()

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

#define KWRAP( ... ) \
	__VA_ARGS__

#define KREP0( X )
#define KREP1( X ) X
#define KREP2( X ) \
	KREP1( X ),    \
	  X
#define KREP3( X ) \
	KREP2( X ),    \
	  X
#define KREP4( X ) \
	KREP3( X ),    \
	  X
#define KREP5( X ) \
	KREP4( X ),    \
	  X
#define KREP6( X ) \
	KREP5( X ),    \
	  X
#define KREP7( X ) \
	KREP6( X ),    \
	  X
#define KREP8( X ) \
	KREP7( X ),    \
	  X
#define KREP9( X ) \
	KREP8( X ),    \
	  X

#define K_REP( TIMES, X ) \
	KREP##TIMES( X )

#define KREP( TIMES, X ) \
	K_REP( TIMES, KWRAP X )

#define KREPID0( X, Y )
#define KREPID1( X, Y ) X##1 Y
#define KREPID2( X, Y ) \
	KREPID1( X, Y ),    \
	  X##2 Y
#define KREPID3( X, Y ) \
	KREPID2( X, Y ),    \
	  X##3 Y
#define KREPID4( X, Y ) \
	KREPID3( X, Y ),    \
	  X##4 Y
#define KREPID5( X, Y ) \
	KREPID4( X, Y ),    \
	  X##5 Y
#define KREPID6( X, Y ) \
	KREPID5( X, Y ),    \
	  X##6 Y
#define KREPID7( X, Y ) \
	KREPID6( X, Y ),    \
	  X##7 Y
#define KREPID8( X, Y ) \
	KREPID7( X, Y ),    \
	  X##8 Y
#define KREPID9( X, Y ) \
	KREPID8( X, Y ),    \
	  X##9 Y

#define K_REPID( TIMES, X, Y ) \
	KREPID##TIMES( X, Y )

#define KREPID( TIMES, X, Y ) \
	K_REPID( TIMES, KWRAP X, KWRAP Y )

#define KREPIDID0( X, Y, Z )
#define KREPIDID1( X, Y, Z ) X##1, Y##1 Z
#define KREPIDID2( X, Y, Z ) \
	KREPIDID1( X, Y, Z ),    \
	  X##2, Y##2 Z
#define KREPIDID3( X, Y, Z ) \
	KREPIDID2( X, Y, Z ),    \
	  X##3, Y##3 Z
#define KREPIDID4( X, Y, Z ) \
	KREPIDID3( X, Y, Z ),    \
	  X##4, Y##4 Z
#define KREPIDID5( X, Y, Z ) \
	KREPIDID4( X, Y, Z ),    \
	  X##5, Y##5 Z
#define KREPIDID6( X, Y, Z ) \
	KREPIDID5( X, Y, Z ),    \
	  X##6, Y##6 Z
#define KREPIDID7( X, Y, Z ) \
	KREPIDID6( X, Y, Z ),    \
	  X##7, Y##7 Z
#define KREPIDID8( X, Y, Z ) \
	KREPIDID7( X, Y, Z ),    \
	  X##8, Y##8 Z
#define KREPIDID9( X, Y, Z ) \
	KREPIDID8( X, Y, Z ),    \
	  X##9, Y##9 Z

#define K_REPIDID( TIMES, X, Y, Z ) \
	KREPIDID##TIMES( X, Y, Z )

#define KREPIDID( TIMES, X, Y, Z ) \
	K_REPIDID( TIMES, KWRAP X, KWRAP Y, KWRAP Z )

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

}  // namespace util

}  // namespace koishi
