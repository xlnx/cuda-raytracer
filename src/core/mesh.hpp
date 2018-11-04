#pragma once

#include <string>
#include <vector>
#include <utility>
#include <vec/vec.hpp>
#include <util/config.hpp>

namespace koishi
{
namespace core
{
struct BVHNode
{
	double3 vmax, vmin;
	uint begin, end, isleaf;
};

using BVHTree = std::vector<BVHNode>;

struct SubMeshCore
{
	double3 emissive;
	double3 color;
};

struct SubMesh : SubMeshCore
{
	std::vector<double3> vertices;
	std::vector<double3> normals;
	std::vector<uint> indices;
	BVHTree bvh;
};

struct PolyMesh
{
	PolyMesh( const jsel::Mesh &config );

	std::vector<SubMesh> mesh;
};

namespace dev
{
struct Mesh : SubMeshCore
{
	const double3 *vertices;
	const double3 *normals;
	const uint *indices;
	const BVHNode *bvh;

	Mesh() = delete;
	Mesh( const SubMesh &other ) :
	  SubMeshCore( other ),
	  vertices( &other.vertices[ 0 ] ),
	  normals( &other.normals[ 0 ] ),
	  indices( &other.indices[ 0 ] ),
	  bvh( &other.bvh[ 0 ] )
	{
	}
};

}  // namespace dev

#if defined( KOISHI_USE_CUDA )

#endif

}  // namespace core

}  // namespace koishi
