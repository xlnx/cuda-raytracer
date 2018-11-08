#include <algorithm>
#include <vector>
#include <utility>
#include <queue>
#include <iostream>
#include <util/exception.hpp>
#include <vec/vmath.hpp>
#include <util/config.hpp>
#include "mesh.hpp"

#define KOISHI_TRIANGLE_STRIPE 32

namespace koishi
{
namespace core
{
struct TriangleInfo
{
	uint3 index;
	double3 vmax;
	double3 vmin;
	double area;
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
		double s = 0;
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
	return res;
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

#define GET( type, key, f )                                            \
	do                                                                 \
	{                                                                  \
		std::string name = #key;                                       \
		{                                                              \
			type value;                                                \
			if ( mat->Get( key, value ) == AI_SUCCESS )                \
			{                                                          \
				std::cout << "> " << name << ": " << f << std::endl;   \
			}                                                          \
			else                                                       \
			{                                                          \
				std::cout << "> " << name << ": unknown" << std::endl; \
			}                                                          \
		}                                                              \
	} while ( 0 )

static void collectObjects( const aiScene *scene, const aiNode *node, const jsel::Mesh &default_config,
							std::vector<core::Mesh> &mesh, const aiMatrix4x4 &tr )
{
	auto trans = tr * node->mTransformation;
	for ( uint i = 0; i != node->mNumMeshes; ++i )
	{
		auto aimesh = scene->mMeshes[ node->mMeshes[ i ] ];
		std::vector<double3> vertices;
		if ( aimesh->HasPositions() )
		{
			vertices.resize( aimesh->mNumVertices );
			for ( uint j = 0; j != aimesh->mNumVertices; ++j )
			{
				auto v = trans * aimesh->mVertices[ j ];
				vertices[ j ] = double3{ v.x, v.y, v.z } * default_config.scale;
				for ( auto &rot : default_config.rotate )
				{
					auto th = radians( rot.degree );
					auto c = cos( th ), s = sin( th );
					auto n = normalize( rot.axis );
					vertices[ j ] = vertices[ j ] * cos( th ) + cross( n, vertices[ j ] ) * s + dot( n, vertices[ j ] ) * n * ( 1 - c );
				}
				vertices[ j ] += default_config.translate;
			}
		}
		std::vector<double3> normals;
		if ( aimesh->HasNormals() )
		{
			normals.resize( aimesh->mNumVertices );
			for ( uint j = 0; j != aimesh->mNumVertices; ++j )
			{
				normals[ j ] = double3{ aimesh->mNormals[ j ].x,
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
					double3 v[] = { vertices[ index.x ], vertices[ index.y ], vertices[ index.z ] };
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

		std::cout << "mesh: " << node->mName.C_Str() << ", mateial: " << aimesh->mMaterialIndex << std::endl;
		auto mat = scene->mMaterials[ aimesh->mMaterialIndex ];

		GET( aiString, AI_MATKEY_NAME, value.C_Str() );
		GET( aiColor3D, AI_MATKEY_COLOR_DIFFUSE, value.r << "," << value.g << "," << value.b );
		GET( aiColor3D, AI_MATKEY_COLOR_AMBIENT, value.r << "," << value.g << "," << value.b );
		GET( aiColor3D, AI_MATKEY_COLOR_SPECULAR, value.r << "," << value.g << "," << value.b );
		GET( aiColor3D, AI_MATKEY_COLOR_EMISSIVE, value.r << "," << value.g << "," << value.b );
		GET( aiColor3D, AI_MATKEY_COLOR_TRANSPARENT, value.r << "," << value.g << "," << value.b );
		GET( aiColor3D, AI_MATKEY_COLOR_REFLECTIVE, value.r << "," << value.g << "," << value.b );
		GET( int, AI_MATKEY_TWOSIDED, value );
		GET( int, AI_MATKEY_SHADING_MODEL, value );
		GET( int, AI_MATKEY_ENABLE_WIREFRAME, value );
		GET( int, AI_MATKEY_BLEND_FUNC, value );
		GET( float, AI_MATKEY_OPACITY, value );
		GET( float, AI_MATKEY_BUMPSCALING, value );
		GET( float, AI_MATKEY_SHININESS, value );
		GET( float, AI_MATKEY_REFLECTIVITY, value );
		GET( float, AI_MATKEY_SHININESS_STRENGTH, value );
		GET( float, AI_MATKEY_REFRACTI, value );

		Mesh m;
		// m.emissive = default_config.emissive;
		// m.color = default_config.color;
		m.bvh = createBVH( indices );
		std::cout << "Successfully buit BVH: " << m.bvh.size() << std::endl;
		m.vertices = std::move( vertices );
		m.normals = std::move( normals );
		m.indices.resize( indices.size() * 3 );
		for ( uint j = 0; j != indices.size(); ++j )
		{
			m.indices[ 3 * j ] = indices[ j ].index.x;
			m.indices[ 3 * j + 1 ] = indices[ j ].index.y;
			m.indices[ 3 * j + 2 ] = indices[ j ].index.z;
		}
		mesh.emplace_back( std::move( m ) );
	}
	for ( auto i = 0u; i != node->mNumChildren; ++i )
	{
		collectObjects( scene, node->mChildren[ i ], default_config, mesh, trans );
	}
}

PolyMesh::PolyMesh( const std::vector<double3> &vertices,
					const std::vector<double3> &normals,
					const std::vector<uint3> &idx,
					const jsel::Mesh &default_config )
{
	std::vector<TriangleInfo> indices;
	for ( auto &index : idx )
	{
		double3 v[] = { vertices[ index.x ], vertices[ index.y ], vertices[ index.z ] };
		TriangleInfo info;
		info.index = index;
		info.vmax = max( v[ 0 ], max( v[ 1 ], v[ 2 ] ) );
		info.vmin = min( v[ 0 ], min( v[ 1 ], v[ 2 ] ) );
		info.area = length( cross( v[ 2 ] - v[ 0 ], v[ 1 ] - v[ 0 ] ) );
		indices.emplace_back( std::move( info ) );
	}
	Mesh m;
	// m.emissive = default_config.emissive;
	// m.color = default_config.color;
	m.bvh = createBVH( indices );
	std::cout << "Successfully buit BVH: " << m.bvh.size() << std::endl;
	m.vertices = vertices;
	m.normals = normals;
	m.indices.resize( indices.size() * 3 );
	for ( uint j = 0; j != indices.size(); ++j )
	{
		m.indices[ 3 * j ] = indices[ j ].index.x;
		m.indices[ 3 * j + 1 ] = indices[ j ].index.y;
		m.indices[ 3 * j + 2 ] = indices[ j ].index.z;
	}
	mesh.emplace_back( std::move( m ) );
}

PolyMesh::PolyMesh( const aiScene *scene, const jsel::Mesh &default_config )
{
	auto t = scene->mRootNode->mTransformation;
	collectObjects( scene, scene->mRootNode, default_config, mesh, t );
}

}  // namespace core

}  // namespace koishi