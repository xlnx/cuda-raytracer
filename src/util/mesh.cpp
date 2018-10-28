#include "mesh.hpp"
#include <util/exception.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace koishi
{
namespace util
{
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
			mesh[ i ].vertices.reserve( scene->mMeshes[ i ]->mNumVertices * 3 );
			mesh[ i ].vertices.insert( mesh[ i ].vertices.begin(),
									   reinterpret_cast<const float *>( scene->mMeshes[ i ]->mVertices ),
									   reinterpret_cast<const float *>( scene->mMeshes[ i ]->mVertices ) + scene->mMeshes[ i ]->mNumVertices * 3 );
		}
	}
	importer.FreeScene();
}

}  // namespace util

}  // namespace koishi