#include <fstream>
#include <iostream>
#include <util/debug.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <vec/vmath.hpp>
#include <core/light/pointLight.hpp>
#include <core/light/areaLight.hpp>
#include "scene.hpp"
#include "mesh.hpp"
#include "sphere.hpp"

#include <ext/material/luz.hpp>

namespace koishi
{
namespace core
{
Scene::Scene( const std::string &path )
{
	// auto pos = path.find_last_of( "." );
	// if ( pos != path.npos && path.substr( pos ) == ".json" )
	// {
	SceneConfig config;
	std::ifstream( path ) >> config;
	std::vector<std::string> mats;

	{
		KINFO( scene, "Loading scene from config '" + path + "'" );
		bool imported = false;
		for ( auto &asset : config.assets )
		{
			if ( asset.name == "import" )
			{
				if ( imported )
				{
					KTHROW( "importing multiple scenes is not allowed" );
				}
				imported = true;
				util::tick();
				auto path = get<std::string>( asset.props, "path" );
				auto diff = primitives.size();
				Assimp::Importer importer;
				auto scene = importer.ReadFile( path, aiProcess_Triangulate |
														aiProcess_GenSmoothNormals |
														aiProcess_FlipUVs |
														aiProcess_JoinIdenticalVertices |
														aiProcess_CalcTangentSpace );
				if ( !( !scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode ) )
				{
					auto poly = core::PolyMesh( scene );
					for ( auto &m : poly.mesh )
					{
						primitives.emplace_back( poly::make_object<Mesh>( std::move( m ) ) );
					}
					mats = std::move( poly.material );
					if ( scene->HasCameras() )
					{
						for ( auto i = 0u; i != scene->mNumCameras; ++i )
						{
							auto &conf = scene->mCameras[ i ];
							CameraConfig cc;
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
							cc.lens = get<std::string>( asset.props, "camera.lens", "pinhole" );
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
								poly::vector<float3> vertices = {
									{ avs[ 0 ].x, avs[ 0 ].y, avs[ 0 ].z },
									{ avs[ 1 ].x, avs[ 1 ].y, avs[ 1 ].z },
									{ avs[ 2 ].x, avs[ 2 ].y, avs[ 2 ].z },
									{ avs[ 3 ].x, avs[ 3 ].y, avs[ 3 ].z }
								};
								poly::vector<float3> normals = {
									{ dir.x, dir.y, dir.z },
									{ dir.x, dir.y, dir.z },
									{ dir.x, dir.y, dir.z },
									{ dir.x, dir.y, dir.z }
								};
								std::vector<uint3> indices = {
									{ 0, 1, 2 },
									{ 0, 2, 3 }
								};
								// MeshConfig def;
								// def.emissive = float3{ 100, 100, 100 };
								for ( auto &m : core::PolyMesh(
												  std::move( vertices ),
												  std::move( normals ),
												  indices )
												  .mesh )
								{
									primitives.emplace_back( poly::make_object<Mesh>( std::move( m ) ) );
								}
							}
						}
					}
				}
				importer.FreeScene();
				diff = primitives.size() - diff;
				KINFO( scene, "Imported", diff, "meshes in", util::tick(), "seconds" );
			}
			else if ( asset.name == "sphere" )
			{
				auto o = get<float3>( asset.props, "o" );
				auto r = get<float>( asset.props, "r" );
				auto mat = get<std::string>( asset.props, "material" );
				uint matid = -1u;
				for ( uint i = 0; i != mats.size(); ++i )
				{
					if ( mats[ i ] == mat )
					{
						matid = i;
						break;
					}
				}
				if ( !~matid )
				{
					matid = mats.size();
					mats.emplace_back( mat );
				}
				primitives.emplace_back( poly::make_object<Sphere>( o, r, matid ) );
			}
			else if ( asset.name.find( "light" ) != asset.name.npos ||
					  asset.name.find( "Light" ) != asset.name.npos )
			{
				if ( asset.name == "pointLight" )
				{
					lights.emplace_back( poly::make_object<PointLight>( asset ) );
				}
				else
				{
					KTHROW( "unknown light type: " + asset.name );
				}
			}
			else
			{
				KTHROW( "unknown asset type: " + asset.name );
			}
		}
	}

	for ( auto &cc : config.camera )
	{
		camera.emplace_back( std::move( cc ) );
	}

	material.resize( mats.size() );
	for ( uint i = 0; i != mats.size(); ++i )
	{
		KLOG( "Looking for material: ", mats[ i ] );
		if ( config.material.find( mats[ i ] ) != config.material.end() )
		{
			auto &mat = config.material[ mats[ i ] ];
			material[ i ] = std::move( Factory<Material>::create( mat ) );
			material[ i ]->print( std::cout );
			std::cout << std::endl;
		}
		else
		{
			KLOG( "Configuration for material {", mats[ i ], "} not found." );
			valid = false;
		}
	}

	if ( valid )
	{
		for ( auto &m : primitives )
		{
			if ( material[ m->matid ].template is<ext::LuzMaterial>() )
			{
				lights.emplace_back( poly::make_object<AreaLight>( material[ m->matid ], m ) );
				if ( auto *p = dynamic_cast<Mesh *>( &*m ) )
				{
					p->generateSamples();
				}
				KLOG( mats[ m->matid ], "is area light" );
			}
		}
		KLOG( lights.size() );
	}

	valid = true;
}

}  // namespace core

}  // namespace koishi
