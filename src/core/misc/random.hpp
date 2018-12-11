#pragma once

#include <cstdlib>
#include <ctime>
#include <core/basic/poly.hpp>

namespace koishi
{
namespace core
{
PolyFunction( DRand48, Require<Host> )(
  ()
	->float {
		static unsigned long long seed = ( ( (long long int)time( nullptr ) ) << 16 ) | ::rand();

		constexpr auto m = 0x100000000LL;
		constexpr auto c = 0xB16;
		constexpr auto a = 0x5DEECE66DLL;
		seed = ( a * seed + c ) & 0xFFFFFFFFFFFFLL;
		unsigned int x = seed >> 16;
		return ( (float)x / (float)m );
	} );

PolyFunction( FakeRand, Require<Device>, Require<Host> )(
  ()
	->float {
		return 0.5;
	} );

template <typename Function>
PolyFunction( Float2, Require<Function> )(
  ()
	->float2 {
		return float2{ call<Function>(),
					   call<Function>() };
	} );

template <typename Function>
PolyFunction( Float3, Require<Function> )(
  ()
	->float3 {
		return float3{ call<Function>(),
					   call<Function>(),
					   call<Function>() };
	} );

template <typename Function>
PolyFunction( Float4, Require<Function> )(
  ()
	->float4 {
		return float4{ call<Function>(),
					   call<Function>(),
					   call<Function>(),
					   call<Function>() };
	} );

}  // namespace core

}  // namespace koishi
