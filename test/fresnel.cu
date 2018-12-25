#include <gtest/gtest.h>
#include <ext/scala/fresnel.hpp>

using namespace koishi;
using namespace core;
using namespace ext;

TEST( test_fresnel, )
{
	Config conf;
	conf.name = "Fresnel";
	conf.props[ "ior" ] = nlohmann::json( 1.f / 1.435 );
	char buf[ 1024 ];
	HybridAllocator al( buf, 1024 );
	Varyings varyings;
	varyings.n = normalized_float3( float3{ 0, 0, 1 } );
	varyings.u = normalized_float3( float3{ 1, 0, 0 } );
	varyings.v = normalized_float3( float3{ 0, 1, 0 } );
	varyings.wo = normalize( float3{ 0, 0, 1 } );
	auto fresnel = Factory<Scala<float>>::create( conf );
	auto fr = fresnel->compute( varyings, al );
	EXPECT_EQ( fr, 0 );
}