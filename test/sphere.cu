#include <core/primitive/sphere.hpp>
#include <gtest/gtest.h>

using namespace koishi;
using namespace core;

TEST( test_sphere, intersect )
{
	Hit hit;
	char s[ 10 ];
	HybridAllocator al( s, 10 );
	Sphere c( float3{ 0, 0, 0 }, 1, 0 );
	ASSERT_EQ( c.intersect( Ray{ { 2, 0, 0 }, normalize( float3{ -1, 0, 0 } ) },
							hit, al ),
			   true );
	ASSERT_EQ( c.intersect( Ray{ { 2, 0, 0 }, normalize( float3{ -1, 0, 1 } ) },
							hit, al ),
			   false );
}