#include <core/basic/poly.hpp>
#include <core/meta/scene.hpp>
#include <gtest/gtest.h>

using namespace koishi;
using namespace core;

TEST( test_scene, x6_json )
{
	core::Scene scene( "./x6.json" );
	ASSERT_EQ( scene, true );
	PolyVector<BVHTree> mesh;
	for ( core::Mesh &e : scene.mesh )
	{
		mesh.emplace_back( std::move( e.bvh ) );
	}
}
