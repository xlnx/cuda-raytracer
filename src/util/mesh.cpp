#include <algorithm>
#include <vector>
#include <utility>
#include <queue>
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <util/exception.hpp>
#include <vec/vmath.hpp>
#include "mesh.hpp"

#define KOISHI_TRIANGLE_STRIPE 16

namespace koishi
{
namespace util
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
		node.begin = begin - info.begin();
		node.end = end - info.end();
		if ( index >= res.size() )
		{
			res.resize( index + 1 );
		}
		res[ index ] = node;
		if ( end - begin > KOISHI_TRIANGLE_STRIPE )
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
	return res;
}

PolyMesh::PolyMesh( const std::string &path )
{
	Assimp::Importer importer;
	auto scene = importer.ReadFile( path, aiProcess_Triangulate |
											aiProcess_GenSmoothNormals |
											aiProcess_FlipUVs |
											aiProcess_JoinIdenticalVertices |
											aiProcess_CalcTangentSpace );
	if ( !( !scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode ) )
	{
		mesh.resize( scene->mNumMeshes );
		for ( uint i = 0; i != scene->mNumMeshes; ++i )
		{
			auto &aimesh = scene->mMeshes[ i ];
			std::vector<float3> vertices;
			if ( aimesh->HasPositions() )
			{
				vertices.resize( aimesh->mNumVertices );
				for ( uint j = 0; j != aimesh->mNumVertices; ++j )
				{
					vertices[ j ] = float3{ aimesh->mVertices[ j ].x,
											aimesh->mVertices[ j ].y,
											aimesh->mVertices[ j ].z };
				}
			}
			std::vector<TriangleInfo> indices;
			if ( aimesh->HasFaces() )
			{
				indices.resize( aimesh->mNumFaces );
				for ( uint j = 0; j != aimesh->mNumFaces; ++j )
				{
					auto index = uint3{ aimesh->mFaces[ j ].mIndices[ 0 ],
										aimesh->mFaces[ j ].mIndices[ 1 ],
										aimesh->mFaces[ j ].mIndices[ 2 ] };
					float3 v[] = { vertices[ index.x ], vertices[ index.y ], vertices[ index.z ] };
					indices[ j ].index = index;
					indices[ j ].vmax = max( v[ 0 ], max( v[ 1 ], v[ 2 ] ) );
					indices[ j ].vmin = min( v[ 0 ], min( v[ 1 ], v[ 2 ] ) );
					indices[ j ].area = length( cross( v[ 2 ] - v[ 0 ], v[ 1 ] - v[ 0 ] ) );
				}
			}
			auto bvh = createBVH( indices );
			std::cout << "Successfully buit BVH: " << bvh.size() << std::endl;
			mesh[ i ].vertices.resize( vertices.size() * 3 );
			for ( uint i = 0; i != vertices.size(); ++i )
			{
				mesh[ i ].vertices[ 3 * i ] = vertices[ i ].x;
				mesh[ i ].vertices[ 3 * i + 1 ] = vertices[ i ].y;
				mesh[ i ].vertices[ 3 * i + 2 ] = vertices[ i ].z;
			}
			mesh[ i ].indices.resize( indices.size() * 3 );
			for ( uint i = 0; i != indices.size(); ++i )
			{
				mesh[ i ].indices[ 3 * i ] = indices[ i ].index.x;
				mesh[ i ].indices[ 3 * i + 1 ] = indices[ i ].index.y;
				mesh[ i ].indices[ 3 * i + 2 ] = indices[ i ].index.z;
			}
		}
	}
	importer.FreeScene();
}

}  // namespace util

}  // namespace koishi