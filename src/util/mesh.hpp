#pragma once

#include <string>
#include <vector>
#include <vec/vec.hpp>

namespace koishi
{
namespace util
{
struct SubMesh
{
	std::vector<float> vertices;
	std::vector<uint> indices;
};

struct PolyMesh
{
	PolyMesh( const std::string &path );

	std::vector<SubMesh> mesh;
};

struct BVHNode
{
	float3 vmax, vmin;
	uint begin, end;
};

using BVHTree = std::vector<BVHNode>;

}  // namespace util

}  // namespace koishi
