#pragma once

#include <cstdlib>
#include <ctime>
#include <core/basic/poly.hpp>

namespace koishi
{
namespace core
{
PolyFunction( DRand48, Require<Host> )(
  ()->double {
	  static unsigned long long seed = ( ( (long long int)time( nullptr ) ) << 16 ) | ::rand();

	  constexpr auto m = 0x100000000LL;
	  constexpr auto c = 0xB16;
	  constexpr auto a = 0x5DEECE66DLL;
	  seed = ( a * seed + c ) & 0xFFFFFFFFFFFFLL;
	  unsigned int x = seed >> 16;
	  return ( (double)x / (double)m );
  } );

PolyFunction( FakeRand, Require<Device>, Require<Host> )(
  ()->double {
	  return 0.5;
  } );

}  // namespace core

}  // namespace koishi
