#include <gtest/gtest.h>
#include <ext/scala/fresnel.hpp>

using namespace koishi;

TEST( test_math, reflect_refract )
{
	solid n( float3{ 0, 0, 1 } );
	solid wo = normalize( float3{ 1, 0, 1 } );
	solid wi, wi2;
	refract( wo, wi, n, 1 / 1.33 );
	KLOG( wo, wi, n );
	EXPECT_NEAR( wi.x, -0.531659, 1e-4 );
	EXPECT_NEAR( wi.y, -0, 1e-4 );
	EXPECT_NEAR( wi.z, -0.846958, 1e-4 );
	KLOG( acos( H::cosTheta( wo ) ) * 180 / PI, acos( H::cosTheta( -wi ) ) * 180 / PI );
	refract( wi, wi2, -n, 1.33 );
	KLOG( wi, wi2, -n );
	EXPECT_NEAR( wi2.x, 0.707107, 1e-4 );
	EXPECT_NEAR( wi2.y, 0, 1e-4 );
	EXPECT_NEAR( wi2.z, 0.707107, 1e-4 );
	KLOG( acos( H::cosTheta( -wi ) ) * 180 / PI, acos( H::cosTheta( wi2 ) ) * 180 / PI );
	char src[ 32 ];
	core::HybridAllocator al( src, 32 );
	core::Varyings varyings;
	varyings.wo = wo;
	ext::Fresnel fresnel( Properties{ { "ior", 1.4 } } );
	EXPECT_NEAR( fresnel.compute( varyings, al ), 0.0365785, 1e-4 );
}
