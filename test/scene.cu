#include <core/basic/poly.hpp>
#include <core/meta/scene.hpp>
#include <gtest/gtest.h>

using namespace koishi;
using namespace core;

struct Mesh
{
	BVHTree bvh;
};

TEST( test_scene, x6_json )
{
	core::Scene scene( "./x6.json" );
	ASSERT_EQ( scene, true );
	PolyVector<::Mesh> mesh;
	for ( core::Mesh &e : scene.mesh )
	{
		::Mesh m;
		m.bvh = std::move( e.bvh );
		mesh.emplace_back( std::move( m ) );
	}
}
