#pragma once

#include <string>
#include <vector>
#include <vec/vec.hpp>
#include <util/config.hpp>

namespace koishi
{
namespace util
{
struct BVHNode
{
	float3 vmax, vmin;
	uint begin, end, isleaf;
};

using BVHTree = std::vector<BVHNode>;

struct SubMesh
{
	std::vector<float3> vertices;
	std::vector<uint> indices;
	BVHTree bvh;
};

struct PolyMesh
{
	PolyMesh( const jsel::Mesh &config );

	std::vector<SubMesh> mesh;
};

}  // namespace util

}  // namespace koishi
