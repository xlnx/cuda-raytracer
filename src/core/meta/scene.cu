#include <fstream>
#include <iostream>
#include <util/debug.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <vec/vmath.hpp>
#include "scene.hpp"

namespace koishi
{
namespace core
{
Scene::Scene( const std::string &path )
{
	// auto pos = path.find_last_of( "." );
	// if ( pos != path.npos && path.substr( pos ) == ".json" )
	// {
	jsel::Scene config;
	std::ifstream( path ) >> config;
	std::vector<std::string> mats;

	{
		KINFO( scene, "Loading scene from config '" + path + "'" );
		util::tick();
		Assimp::Importer importer;
		auto scene = importer.ReadFile( config.path, aiProcess_Triangulate |
													   aiProcess_GenSmoothNormals |
													   aiProcess_FlipUVs |
													   aiProcess_JoinIdenticalVertices |
													   aiProcess_CalcTangentSpace );
		if ( !( !scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode ) )
		{
			auto poly = core::PolyMesh( scene );
			mesh = std::move( poly.mesh );
			mats = std::move( poly.material );
			if ( scene->HasCameras() )
			{
				for ( auto i = 0u; i != scene->mNumCameras; ++i )
				{
					auto &conf = scene->mCameras[ i ];
					jsel::Camera cc;
					cc.aspect = conf->mAspect;
					cc.fovx = 2 * degrees( conf->mHorizontalFOV );
					auto trans = scene->mRootNode->FindNode( conf->mName )->mTransformation;
					auto rot = trans;
					rot.a4 = rot.b4 = rot.c4 = 0.f;
					rot.d4 = 1.f;
					auto up = rot * conf->mUp;
					auto lookat = rot * conf->mLookAt;
					auto position = trans * conf->mPosition;
					cc.upaxis = normalize( float3{ up.x, up.y, up.z } );
					cc.target = normalize( float3{ lookat.x, lookat.y, lookat.z } );
					cc.position = { position.x, position.y, position.z };
					cc.zNear = conf->mClipPlaneNear;
					cc.zFar = conf->mClipPlaneFar;
					camera.emplace_back( cc );
				}
			}
			if ( scene->HasLights() )
			{
				for ( auto i = 0u; i != scene->mNumLights; ++i )
				{
					auto &conf = scene->mLights[ i ];
					auto trans = scene->mRootNode->FindNode( conf->mName )->mTransformation;
					auto rot = trans;
					rot.a4 = rot.b4 = rot.c4 = 0.f;
					rot.d4 = 1.f;
					if ( conf->mType == aiLightSource_AREA )
					{
						auto dir = conf->mDirection;
						auto p = conf->mPosition;
						auto u = dir ^ conf->mUp;
						u.Normalize();
						auto v = u ^ dir;
						v.Normalize();
						conf->mSize *= .5;

						std::vector<aiVector3D> avs = {
							trans * ( p + conf->mSize.x * u + conf->mSize.y * v ),
							trans * ( p + conf->mSize.x * u - conf->mSize.y * v ),
							trans * ( p - conf->mSize.x * u - conf->mSize.y * v ),
							trans * ( p - conf->mSize.x * u + conf->mSize.y * v )
						};
						PolyVector<float3> vertices = {
							{ avs[ 0 ].x, avs[ 0 ].y, avs[ 0 ].z },
							{ avs[ 1 ].x, avs[ 1 ].y, avs[ 1 ].z },
							{ avs[ 2 ].x, avs[ 2 ].y, avs[ 2 ].z },
							{ avs[ 3 ].x, avs[ 3 ].y, avs[ 3 ].z }
						};
						PolyVector<float3> normals = {
							{ dir.x, dir.y, dir.z },
							{ dir.x, dir.y, dir.z },
							{ dir.x, dir.y, dir.z },
							{ dir.x, dir.y, dir.z }
						};
						std::vector<uint3> indices = {
							{ 0, 1, 2 },
							{ 0, 2, 3 }
						};
						// jsel::Mesh def;
						// def.emissive = float3{ 100, 100, 100 };
						for ( auto &m : core::PolyMesh(
										  std::move( vertices ),
										  std::move( normals ),
										  indices )
										  .mesh )
						{
							mesh.emplace_back( std::move( m ) );
						}
					}
				}
			}
		}
		importer.FreeScene();
		KINFO( scene, "Imported", mesh.size(), "meshes in", util::tick(), "seconds" );
	}

	//material.resize( mats.size() );
	for ( uint i = 0; i != mats.size(); ++i )
	{
		//if ( config.material.find( mats[ i ] ) != config.material.end() )
		//{
		//	material[ i ] = std::move( config.material[ mats[ i ] ] );
		//}
		//else
		//{
		//	std::cout << "configuration for material <" << mats[ i ] << "> not found" << std::endl;
		//	valid = false;
		//}
	}
	valid = true;
}

}  // namespace core

}  // namespace koishi
