#pragma once

#include <string>
#include <vector>
#include <vec/vec.hpp>

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
	std::vector<float> vertices;
	std::vector<uint> indices;
	BVHTree bvh;
};

struct PolyMesh
{
	PolyMesh( const std::string &path );

	std::vector<SubMesh> mesh;
};

}  // namespace util

}  // namespace koishi
