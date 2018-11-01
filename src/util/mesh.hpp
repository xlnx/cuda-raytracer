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
	double3 vmax, vmin;
	uint begin, end, isleaf;
};

using BVHTree = std::vector<BVHNode>;

struct SubMesh
{
	std::vector<double3> vertices;
	std::vector<double3> normals;
	std::vector<uint> indices;
	BVHTree bvh;
	double3 emissive;
	double3 color;
};

struct PolyMesh
{
	PolyMesh( const jsel::Mesh &config );

	std::vector<SubMesh> mesh;
};

}  // namespace util

}  // namespace koishi
