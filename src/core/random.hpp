#pragma once

#include <cstdlib>
#include <ctime>

namespace koishi
{
namespace core
{
struct DRand48
{
	static double rand()
	{
		static unsigned long long seed = ( ( (long long int)time( nullptr ) ) << 16 ) | ::rand();

		constexpr auto m = 0x100000000LL;
		constexpr auto c = 0xB16;
		constexpr auto a = 0x5DEECE66DLL;
		seed = ( a * seed + c ) & 0xFFFFFFFFFFFFLL;
		unsigned int x = seed >> 16;
		return ( (double)x / (double)m );
	}
};

}  // namespace core

}  // namespace koishi