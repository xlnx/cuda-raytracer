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
#include <core/basic/allocator.hpp>
#include <core/basic/queue.hpp>
#include "primitive.hpp"

namespace koishi
{
namespace core
{
struct BVHNode
{
	float3 vmax, vmin;
	uint begin, end, offset;
};

using BVHTree = poly::vector<BVHNode>;

struct CompactMesh
{
	poly::vector<float3> vertices;
	poly::vector<float3> normals;
	poly::vector<uint> indices;
	BVHTree bvh;
	uint matid;
};

namespace attr
{
struct Face
{
	float3 o, d1, d2;  // o is vertex0, o + d1 is vertex1, o + d2 is vertex2
};

struct Normal
{
	float3 n0, n1, n2;
};

}  // namespace attr

struct Mesh : Primitive
{
	Mesh( const CompactMesh &other );
	Mesh( CompactMesh &&other );

	// SOA of vertex attributes
	poly::vector<attr::Face> faces;
	poly::vector<attr::Normal> normals;
	BVHTree bvh;

	KOISHI_HOST_DEVICE float3 normal( const Hit &hit ) const override
	{
		return interplot( normals[ hit.id ].n0,
						  normals[ hit.id ].n1,
						  normals[ hit.id ].n2,
						  hit.uv );
	}
	KOISHI_HOST_DEVICE bool intersect( const Ray &ray, Hit &hit, Allocator &pool ) const override;
	KOISHI_HOST_DEVICE bool intersect( const Seg &seg, Allocator &pool ) const override;
};

struct PolyMesh
{
	PolyMesh( poly::vector<float3> &&vertices,
			  poly::vector<float3> &&normals,
			  const std::vector<uint3> &indices );
	PolyMesh( const aiScene *scene );

private:
	void collectObjects( const aiScene *scene, const aiNode *node, const aiMatrix4x4 &tr );

public:
	poly::vector<Mesh> mesh;
	std::vector<std::string> material;
};

}  // namespace core

}  // namespace koishi
