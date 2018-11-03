#pragma once

#include <string>
#include <vector>
#include <utility>
#include <vec/vec.hpp>
#include <util/config.hpp>

#if defined( KOISHI_USE_CUDA )
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

namespace koishi
{
namespace core
{
#if defined( KOISHI_USE_CUDA )

namespace dev
{
template <typename T>
using vector = thrust::device_vector<T>;

}

namespace host
{
template <typename T>
using vector = thrust::host_vector<T>;

}
#endif

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

#if defined( KOISHI_USE_CUDA )

namespace dev
{
using BVHTree = dev::vector<BVHNode>;

struct SubMesh : core::SubMeshCore
{
	dev::vector<double3> vertices;
	dev::vector<double3> normals;
	dev::vector<uint> indices;
	dev::BVHTree bvh;

	SubMesh() = default;
	SubMesh( const SubMesh & ) = default;
	SubMesh( SubMesh && ) = default;
	SubMesh &operator=( const SubMesh & ) = default;
	SubMesh &operator=( SubMesh && ) = default;
	~SubMesh() = default;

	SubMesh( const core::SubMesh &other ) :
	  SubMeshCore( static_cast<const SubMeshCore &>( other ) ),
	  vertices( other.vertices ),
	  normals( other.normals ),
	  indices( other.indices ),
	  bvh( other.bvh )
	{
	}
	SubMesh( core::SubMesh &&other ) :
	  SubMeshCore( std::move( static_cast<SubMeshCore &&>( other ) ) ),
	  vertices( std::move( other.vertices ) ),
	  normals( std::move( other.normals ) ),
	  indices( std::move( other.indices ) ),
	  bvh( std::move( other.bvh ) )
	{
	}
	SubMesh &operator=( const core::SubMesh &other )
	{
		static_cast<SubMeshCore &>( *this ) = static_cast<const SubMeshCore &>( other );
		vertices = other.vertices;
		normals = other.normals;
		indices = other.indices;
		bvh = other.bvh;
		return *this;
	}
	SubMesh &operator=( core::SubMesh &&other )
	{
		static_cast<SubMeshCore &>( *this ) = std::move( static_cast<SubMeshCore &&>( other ) );
		vertices = std::move( other.vertices );
		normals = std::move( other.normals );
		indices = std::move( other.indices );
		bvh = std::move( other.bvh );
		return *this;
	}
};

}  // namespace dev

#else

namespace dev
{
using SubMesh = void;

}  // namespace dev

#endif

}  // namespace core

}  // namespace koishi
