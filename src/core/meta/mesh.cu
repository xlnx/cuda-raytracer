#include <algorithm>
#include <cstdlib>
#include <vector>
#include <utility>
#include <iostream>
#include <util/exception.hpp>
#include <vec/vmath.hpp>
#include <util/debug.hpp>
#include "mesh.hpp"

#define KOISHI_TRIANGLE_STRIPE 16
#define KOISHI_TRIANGLE_WARP 4

namespace koishi
{
namespace core
{
struct TriangleInfo
{
	uint3 index;
	float3 vmax, vmin;
};

static inline float area( const float3 &vmin, const float3 &vmax )
{
	auto e = vmax - vmin;
	return e.x * e.y * e.z;
}

struct BVHTreeNode
{
	BVHNode node;
	BVHTreeNode *left, *right;
};

static BVHTreeNode *doCreateBVH(
  std::vector<TriangleInfo> &info,
  std::vector<TriangleInfo>::iterator begin,
  std::vector<TriangleInfo>::iterator end )
{
	BVHNode node;

	node.vmax = begin->vmax;
	node.vmin = begin->vmin;
	for ( auto iter = begin; iter != end; ++iter )
	{
		node.vmax = max( node.vmax, iter->vmax );
		node.vmin = min( node.vmin, iter->vmin );
	}

	// len = k * KOISHI_TRIANGLE_WARP
	auto len = end - begin;
	node.begin = ( begin - info.begin() ) * 3;
	node.end = ( end - info.begin() ) * 3;
	node.offset = len > KOISHI_TRIANGLE_STRIPE;

	auto res = new BVHTreeNode;
	res->node = node;

	if ( node.offset )
	{
		auto w = node.vmax - node.vmin;
		if ( w.x >= w.y && w.x >= w.z )
		{
			std::sort( begin, end, []( const TriangleInfo &a, const TriangleInfo &b ) { return a.vmin.x < b.vmin.x; } );
		}
		else if ( w.y >= w.x && w.y >= w.z )
		{
			std::sort( begin, end, []( const TriangleInfo &a, const TriangleInfo &b ) { return a.vmin.y < b.vmin.y; } );
		}
		else
		{
			std::sort( begin, end, []( const TriangleInfo &a, const TriangleInfo &b ) { return a.vmin.z < b.vmin.z; } );
		}

		std::vector<float> low( len ), high( len );
		auto vmin = float3{ INFINITY, INFINITY, INFINITY }, vmax = float3{ -INFINITY, -INFINITY, -INFINITY };
		auto low_iter = low.begin();
		for ( auto iter = begin; iter != end; ++iter )
		{
			vmin = min( iter->vmin, vmin );
			vmax = max( iter->vmax, vmax );
			*low_iter++ = area( vmin, vmax );
		}

		vmin = float3{ INFINITY, INFINITY, INFINITY }, vmax = float3{ -INFINITY, -INFINITY, -INFINITY };
		using riter_t = std::reverse_iterator<decltype( end )>;
		auto high_iter = high.rbegin();
		for ( auto iter = riter_t( end ); iter != riter_t( begin ); ++iter )
		{
			vmin = min( iter->vmin, vmin );
			vmax = max( iter->vmax, vmax );
			*high_iter++ = area( vmin, vmax );
		}

		for ( auto i = KOISHI_TRIANGLE_WARP; i < len; i += KOISHI_TRIANGLE_WARP )
		{
			if ( low[ i - 1 ] - high[ i ] >= 0.f )
			{  // i = k * KOISHI_TRIANGLE_WARP
				res->left = doCreateBVH( info, begin, begin + i );
				res->right = doCreateBVH( info, begin + i, end );

				return res;
			}
		}

		res->left = doCreateBVH( info, begin, begin + len / 2 );
		res->right = doCreateBVH( info, begin + len / 2, end );
	}

	return res;
}

static void traversal( BVHTreeNode *root, BVHTree &tree )
{
	auto i = tree.size();
	tree.emplace_back( root->node );
	if ( root->node.offset )
	{
		traversal( root->left, tree );  // left root = i + 1
		delete root->left;
		tree[ i ].offset = tree.size() - i;
		traversal( root->right, tree );
		delete root->right;
	}
}

static BVHTree plainlizeBVH( BVHTreeNode *root )
{
	BVHTree tree;
	traversal( root, tree );
	delete root;
	return std::move( tree );
}

static BVHTree createBVH( std::vector<TriangleInfo> &info )
{
	auto e = info.back();
	while ( info.size() % KOISHI_TRIANGLE_WARP != 0 )
	{
		info.emplace_back( e );
	}
	return plainlizeBVH( doCreateBVH( info, info.begin(), info.end() ) );
}

Mesh::Mesh( const CompactMesh &other ) :
  faces( other.indices.size() / 3 ),
  bvh( other.bvh.size() ),
  matid( other.matid )
{
	for ( uint i = 0, j = 0; i != other.indices.size(); i += 3, ++j )
	{
		auto i0 = other.indices[ i ];
		auto i1 = other.indices[ i + 1 ];
		auto i2 = other.indices[ i + 2 ];

		auto o = other.vertices[ i0 ];
		faces[ j ].o = o;
		faces[ j ].d1 = other.vertices[ i1 ] - o;
		faces[ j ].d2 = other.vertices[ i2 ] - o;

		faces[ j ].n0 = other.normals[ i0 ];
		faces[ j ].n1 = other.normals[ i1 ];
		faces[ j ].n2 = other.normals[ i2 ];
	}
	memcpy( bvh.data(), other.bvh.data(), sizeof( BVHNode ) * other.bvh.size() );
	for ( auto &node : bvh )
	{
		node.begin /= 3, node.end /= 3;
	}
}

Mesh::Mesh( CompactMesh &&other ) :
  faces( other.indices.size() / 3 ),
  bvh( std::move( other.bvh ) ),
  matid( other.matid )
{
	for ( uint i = 0, j = 0; i != other.indices.size(); i += 3, ++j )
	{
		auto i0 = other.indices[ i ];
		auto i1 = other.indices[ i + 1 ];
		auto i2 = other.indices[ i + 2 ];

		auto o = other.vertices[ i0 ];
		faces[ j ].o = o;
		faces[ j ].d1 = other.vertices[ i1 ] - o;
		faces[ j ].d2 = other.vertices[ i2 ] - o;

		faces[ j ].n0 = other.normals[ i0 ];
		faces[ j ].n1 = other.normals[ i1 ];
		faces[ j ].n2 = other.normals[ i2 ];
	}
	for ( auto &node : bvh )
	{
		node.begin /= 3, node.end /= 3;
	}
}

KOISHI_HOST_DEVICE bool Mesh::intersect( const Ray &ray, Hit &hit, Allocator &pool ) const
{
	hit.t = INFINITY;

	Stack<uint> Q( pool );

	Q.emplace( 0 );

	while ( !Q.empty() )
	{
		auto i = Q.top();
		Q.pop();

		while ( auto offset = bvh[ i ].offset )
		{  // using depth frist search will cost less space than bfs.
			int left = ray.intersect_bbox( bvh[ i + 1 ].vmin, bvh[ i + 1 ].vmax );
			int right = ray.intersect_bbox( bvh[ i + offset ].vmin, bvh[ i + offset ].vmax );
			if ( !left && !right )  // no intersection on this branch
			{
				goto NEXT_BRANCH;
			}
			if ( left && right )  // both intersects, trace second and push first
			{
				Q.emplace( i + offset );
			}
			if ( left )
			{
				i++;
			}
			else
			{
				i += offset;
			}
		}
		// now this node is leaf
		uint begin, end;
		begin = bvh[ i ].begin, end = bvh[ i ].end;

#if ( KOISHI_TRIANGLE_WARP == 2 )
#pragma unroll( 2 )
#elif ( KOISHI_TRIANGLE_WARP == 4 )
#pragma unroll( 4 )
#elif ( KOISHI_TRIANGLE_WARP >= 8 )
#pragma unroll( 8 )
#endif
		for ( uint j = begin; j < end; ++j )
		{
			Hit hit1;
			auto &face = faces[ j ];
			if ( ray.intersect_triangle( 
			  face.o, face.d1, face.d2, 
			  hit1 ) && hit1.t < hit.t )
			{
				hit = hit1;
				hit.id = j;
			}
		}

	NEXT_BRANCH:;
	}

	return hit.t != INFINITY;
}

void PolyMesh::collectObjects( const aiScene *scene, const aiNode *node, const aiMatrix4x4 &tr )
{
	auto trans = tr * node->mTransformation;
	for ( uint i = 0; i != node->mNumMeshes; ++i )
	{
		auto aimesh = scene->mMeshes[ node->mMeshes[ i ] ];
		PolyVector<float3> vertices;
		if ( aimesh->HasPositions() )
		{
			vertices.resize( aimesh->mNumVertices );
			for ( uint j = 0; j != aimesh->mNumVertices; ++j )
			{
				auto v = trans * aimesh->mVertices[ j ];
				vertices[ j ] = float3{ v.x, v.y, v.z };
			}
		}
		PolyVector<float3> normals;
		if ( aimesh->HasNormals() )
		{
			normals.resize( aimesh->mNumVertices );
			for ( uint j = 0; j != aimesh->mNumVertices; ++j )
			{
				normals[ j ] = float3{ aimesh->mNormals[ j ].x,
									   aimesh->mNormals[ j ].y,
									   aimesh->mNormals[ j ].z };
			}
		}
		std::vector<TriangleInfo> indices;
		if ( aimesh->HasFaces() )
		{
			for ( uint j = 0; j != aimesh->mNumFaces; ++j )
			{
				if ( aimesh->mFaces[ j ].mNumIndices == 3 )
				{
					auto index = uint3{ aimesh->mFaces[ j ].mIndices[ 0 ],
										aimesh->mFaces[ j ].mIndices[ 1 ],
										aimesh->mFaces[ j ].mIndices[ 2 ] };
					float3 v[] = { vertices[ index.x ], vertices[ index.y ], vertices[ index.z ] };
					TriangleInfo info;
					info.index = index;
					info.vmax = max( v[ 0 ], max( v[ 1 ], v[ 2 ] ) );
					info.vmin = min( v[ 0 ], min( v[ 1 ], v[ 2 ] ) );
					indices.emplace_back( std::move( info ) );
				}
			}
		}
		if ( indices.size() <= 0 ) continue;

		CompactMesh m;
		// m.emissive = default_config.emissive;
		// m.color = default_config.color;
		m.bvh = std::move( createBVH( indices ) );
		KLOG( "- Built BVH of size", m.bvh.size(), ", depth", ceil( log2( m.bvh.size() ) ) );
		m.vertices = std::move( vertices );
		m.normals = std::move( normals );
		m.matid = aimesh->mMaterialIndex;
		PolyVector<uint> idxBuffer( indices.size() * 3 );
		for ( uint j = 0; j != indices.size(); ++j )
		{
			idxBuffer[ 3 * j ] = indices[ j ].index.x;
			idxBuffer[ 3 * j + 1 ] = indices[ j ].index.y;
			idxBuffer[ 3 * j + 2 ] = indices[ j ].index.z;
		}
		m.indices = std::move( idxBuffer );
		mesh.emplace_back( std::move( m ) );
	}
	for ( auto i = 0u; i != node->mNumChildren; ++i )
	{
		collectObjects( scene, node->mChildren[ i ], trans );
	}
}

PolyMesh::PolyMesh( PolyVector<float3> &&vertices,
					PolyVector<float3> &&normals,
					const std::vector<uint3> &idx )
{
	std::vector<TriangleInfo> indices;
	for ( auto &index : idx )
	{
		float3 v[] = { vertices[ index.x ], vertices[ index.y ], vertices[ index.z ] };
		TriangleInfo info;
		info.index = index;
		info.vmax = max( v[ 0 ], max( v[ 1 ], v[ 2 ] ) );
		info.vmin = min( v[ 0 ], min( v[ 1 ], v[ 2 ] ) );
		indices.emplace_back( std::move( info ) );
	}
	CompactMesh m;
	m.bvh = createBVH( indices );
	KLOG( "- Built BVH of size", m.bvh.size(), ", depth", ceil( log2( m.bvh.size() ) ) );
	m.vertices = std::move( vertices );
	m.normals = std::move( normals );
	PolyVector<uint> idxBuffer( indices.size() * 3 );
	for ( uint j = 0; j != indices.size(); ++j )
	{
		idxBuffer[ 3 * j ] = indices[ j ].index.x;
		idxBuffer[ 3 * j + 1 ] = indices[ j ].index.y;
		idxBuffer[ 3 * j + 2 ] = indices[ j ].index.z;
	}
	m.indices = std::move( idxBuffer );
	mesh.emplace_back( std::move( m ) );
}

PolyMesh::PolyMesh( const aiScene *scene )
{
	auto t = scene->mRootNode->mTransformation;
	collectObjects( scene, scene->mRootNode, t );
	for ( uint i = 0; i != scene->mNumMaterials; ++i )
	{
		auto mat = scene->mMaterials[ i ];
		aiString name;
		mat->Get( AI_MATKEY_NAME, name );
		material.emplace_back( name.C_Str() );
	}
}

}  // namespace core

}  // namespace koishi
