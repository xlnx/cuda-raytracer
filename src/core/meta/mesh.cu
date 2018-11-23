#include <algorithm>
#include <vector>
#include <utility>
#include <queue>
#include <iostream>
#include <util/exception.hpp>
#include <vec/vmath.hpp>
#include <util/debug.hpp>
#include "mesh.hpp"

#define KOISHI_TRIANGLE_STRIPE 32

namespace koishi
{
namespace core
{
struct TriangleInfo
{
	uint3 index;
	float3 vmax;
	float3 vmin;
	float area;
};

static BVHTree createBVH( std::vector<TriangleInfo> &info )
{
	struct QueueItem
	{
		uint index;
		std::vector<TriangleInfo>::iterator begin, end;
	};
	BVHTree res( 2 );
	std::queue<QueueItem> Q;
	Q.emplace( QueueItem{ 1, info.begin(), info.end() } );
	while ( !Q.empty() )
	{
		uint index = Q.front().index;
		auto begin = Q.front().begin;
		auto end = Q.front().end;
		Q.pop();

		BVHNode node;  // current bbox
		node.vmax = begin->vmax;
		node.vmin = begin->vmin;
		float s = 0;
		for ( auto iter = begin; iter != end; ++iter )
		{
			node.vmax = max( node.vmax, iter->vmax );
			node.vmin = min( node.vmin, iter->vmin );
			s += iter->area;
		}
		node.begin = ( begin - info.begin() ) * 3;
		node.end = ( end - info.begin() ) * 3;
		node.isleaf = end - begin <= KOISHI_TRIANGLE_STRIPE;
		if ( index >= res.size() )
		{
			res.resize( index + 1 );
		}
		res[ index ] = node;
		if ( !node.isleaf )
		{
			s /= 2;
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
			for ( auto iter = begin; iter != end; ++iter )
			{
				if ( ( s -= iter->area ) <= 0.f )
				{
					Q.emplace( QueueItem{ index << 1, begin, iter } );
					Q.emplace( QueueItem{ ( index << 1 ) + 1, iter, end } );
					break;
				}
			}
		}
	}
	return std::move( res );
}

static void printBVH( const BVHTree &tr, uint index = 1 )
{
	std::cout << index << " ";
	if ( !tr[ index ].isleaf )
	{
		printBVH( tr, index << 1 );
		printBVH( tr, ( index << 1 ) + 1 );
	}
}

KOISHI_HOST_DEVICE bool Mesh::intersect( const Ray &ray, uint root, Hit &hit, Allocator &pool ) const
{
	hit.t = INFINITY;

	CyclicQueue<uint> Q( pool );

	Q.emplace( root );

	while ( !Q.empty() )
	{
		KASSERT( !Q.overflow() );

		auto i = Q.front();
		Q.pop();

		while ( !bvh[ i ].isleaf )
		{  // using depth frist search will cost less space than bfs.
			auto left = ray.intersect_bbox( bvh[ i << 1 ].vmin, bvh[ i << 1 ].vmax );
			auto right = ray.intersect_bbox( bvh[ ( i << 1 ) + 1 ].vmin, bvh[ ( i << 1 ) + 1 ].vmax );
			if ( !left && !right )  // no intersection on this branch
			{
				goto NEXT_BRANCH;
			}
			if ( left && right )  // both intersects, trace second and push first
			{
				Q.emplace( i << 1 );
			}
			i <<= 1;
			if ( right ) i |= 1;
		}
		// now this node is leaf
		for ( uint j = bvh[ i ].begin; j < bvh[ i ].end; j += 3 )
		{
			Hit hit1;
			if ( ray.intersect_triangle( vertices[ indices[ j ] ],
										 vertices[ indices[ j + 1 ] ],
										 vertices[ indices[ j + 2 ] ], hit1 ) &&
				 hit1.t < hit.t )
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
					info.area = length( cross( v[ 2 ] - v[ 0 ], v[ 1 ] - v[ 0 ] ) );
					indices.emplace_back( std::move( info ) );
				}
			}
		}
		if ( indices.size() <= 0 ) continue;

		Mesh m;
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
		info.area = length( cross( v[ 2 ] - v[ 0 ], v[ 1 ] - v[ 0 ] ) );
		indices.emplace_back( std::move( info ) );
	}
	Mesh m;
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
