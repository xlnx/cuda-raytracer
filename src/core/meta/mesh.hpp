#pragma once

#include <new>
#include <cstdlib>
#include <string>
#include <vector>
#include <utility>
#include <assimp/scene.h>
#include <vec/vec.hpp>
#include <util/config.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/poly.hpp>

namespace koishi
{
namespace core
{
struct BVHNode
{
	double3 vmax, vmin;
	uint begin, end, isleaf;
};

using BVHTree = PolyVectorView<BVHNode>;

struct Mesh : Poly<Mesh>
{
	PolyVectorView<double3> vertices;
	PolyVectorView<double3> normals;
	PolyVectorView<uint> indices;
	BVHTree bvh;
	uint matid;

	KOISHI_HOST_DEVICE bool intersect( const Ray &ray, uint root, Hit &hit ) const;
};

struct PolyMesh
{
	PolyMesh( PolyVector<double3> &&vertices,
			  PolyVector<double3> &&normals,
			  const std::vector<uint3> &indices );
	PolyMesh( const aiScene *scene );

private:
	void collectObjects( const aiScene *scene, const aiNode *node, const aiMatrix4x4 &tr );

public:
	PolyVector<Mesh> mesh;
	std::vector<std::string> material;
};

}  // namespace core

}  // namespace koishi
