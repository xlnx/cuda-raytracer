#pragma once

#include <new>
#include <cstdlib>
#include <string>
#include <vector>
#include <utility>
#include <vec/vec.hpp>
#include <util/config.hpp>
#include <assimp/scene.h>
#include <core/ray.hpp>

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

struct Mesh
{
	std::vector<double3> vertices;
	std::vector<double3> normals;
	std::vector<uint> indices;
	BVHTree bvh;
	uint matid;
};

struct PolyMesh
{
	PolyMesh( const std::vector<double3> &vertices,
			  const std::vector<double3> &normals,
			  const std::vector<uint3> &indices,
			  const jsel::Mesh &default_config );
	PolyMesh( const aiScene *scene, const jsel::Mesh &default_config );

	std::vector<Mesh> mesh;
};

}  // namespace core

}  // namespace koishi
