#include <fstream>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <vec/vmath.hpp>
#include "scene.hpp"
#include "mesh.hpp"

namespace koishi
{
namespace core
{
Scene::Scene( const std::string &path )
{
	auto pos = path.find_last_of( "." );
	if ( pos != path.npos && path.substr( pos ) == ".json" )
	{
		jsel::Scene scene;
		std::ifstream( path ) >> scene;
		for ( auto &m : scene.mesh )
		{
			Assimp::Importer importer;
			auto scene = importer.ReadFile( m.path, aiProcess_Triangulate |
													  aiProcess_GenSmoothNormals |
													  aiProcess_FlipUVs |
													  aiProcess_JoinIdenticalVertices |
													  aiProcess_CalcTangentSpace );
			if ( !( !scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode ) )
			{
				for ( auto &e : core::PolyMesh( scene, m ).mesh )
				{
					mesh.emplace_back( std::move( e ) );
				}
			}
			importer.FreeScene();
		}
		camera = std::move( scene.camera );
	}
	else
	{
		Assimp::Importer importer;
		auto scene = importer.ReadFile( path, aiProcess_Triangulate |
												aiProcess_GenSmoothNormals |
												aiProcess_FlipUVs |
												aiProcess_JoinIdenticalVertices |
												aiProcess_CalcTangentSpace );
		if ( !( !scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode ) )
		{
			jsel::Mesh m;
			m.color = double3{ 0.8392156862745098, 0.47058823529411764, 0.16470588235294117 };
			for ( auto &e : core::PolyMesh( scene, m ).mesh )
			{
				mesh.emplace_back( std::move( e ) );
			}
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
					cc.upaxis = normalize( double3{ up.x, up.y, up.z } );
					cc.target = normalize( double3{ lookat.x, lookat.y, lookat.z } );
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
						std::vector<double3> vertices = {
							{ avs[ 0 ].x, avs[ 0 ].y, avs[ 0 ].z },
							{ avs[ 1 ].x, avs[ 1 ].y, avs[ 1 ].z },
							{ avs[ 2 ].x, avs[ 2 ].y, avs[ 2 ].z },
							{ avs[ 3 ].x, avs[ 3 ].y, avs[ 3 ].z }
						};
						std::vector<double3> normals = {
							{ dir.x, dir.y, dir.z },
							{ dir.x, dir.y, dir.z },
							{ dir.x, dir.y, dir.z },
							{ dir.x, dir.y, dir.z }
						};
						std::vector<uint3> indices = {
							{ 0, 1, 2 },
							{ 0, 2, 3 }
						};
						jsel::Mesh def;
						def.emissive = double3{ 100, 100, 100 };
						for ( auto &m : core::PolyMesh( vertices, normals, indices, def ).mesh )
						{
							mesh.emplace_back( std::move( m ) );
						}
					}
				}
			}
		}
		importer.FreeScene();
	}
}

}  // namespace core

}  // namespace koishi