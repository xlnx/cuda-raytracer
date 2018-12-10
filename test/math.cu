#include <gtest/gtest.h>
#include <vec/vmath.hpp>

using namespace koishi;

TEST( test_math, reflect_refract )
{
	float3 wi{ 1, 1, 1 };
	normalized_float3 n( float3{ 0, 0, 1 } );
	auto wo = reflect( wi, n ), expwo = float3{ -1, -1, 1 };
	EXPECT_EQ( wo, expwo );
	// wo = refract( wi, n, 0.5 ), expwo = float3{ -0.5, -0.5, -1.1242 };
	// EXPECT_EQ( wo, expwo );
	std::cout << refract( wi, n, 1.2246 ) << std::endl;
}
