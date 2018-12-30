#include <iostream>
#include <util/debug.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <vec/vmath.hpp>
#include <core/light/pointLight.hpp>
#include <core/light/areaLight.hpp>
#include <core/primitive/mesh.hpp>
#include <core/primitive/sphere.hpp>
#include "scene.hpp"

#include <ext/shader/emission.hpp>

namespace koishi
{
namespace core
{
Scene::Scene( SceneConfig &config )
{
	// auto pos = path.find_last_of( "." );
	// if ( pos != path.npos && path.substr( pos ) == ".json" )
	// {
	std::vector<std::string> shaderNames;

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
			auto cconf = get<CameraConfig>( asset.props, "camera" );
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
				shaderNames = std::move( poly.shaders );
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
						cc.lens = cconf.lens;
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
							// def.emission = float3{ 100, 100, 100 };
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
			auto shaderName = get<std::string>( asset.props, "shader" );
			uint shaderId = -1u;
			for ( uint i = 0; i != shaderNames.size(); ++i )
			{
				if ( shaderNames[ i ] == shaderName )
				{
					shaderId = i;
					break;
				}
			}
			if ( !~shaderId )
			{
				shaderId = shaderNames.size();
				shaderNames.emplace_back( shaderName );
			}
			primitives.emplace_back( poly::make_object<Sphere>( o, r, shaderId ) );
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

	for ( auto &cc : config.camera )
	{
		camera.emplace_back( std::move( cc ) );
	}

	shaders.resize( shaderNames.size() );
	for ( uint i = 0; i != shaderNames.size(); ++i )
	{
		KLOG( "Looking for shader:", shaderNames[ i ] );
		if ( config.shaders.find( shaderNames[ i ] ) != config.shaders.end() )
		{
			auto &shaderName = config.shaders[ shaderNames[ i ] ];
			shaders[ i ] = std::move( Factory<Shader>::create( shaderName ) );
			json j;
			shaders[ i ]->writeNode( j );
			j.dump( std::cout, true, 2 );
			KLOG( " =", i );
		}
		else
		{
			KLOG( "Configuration for shader {", shaderNames[ i ], "} not found." );
			valid = false;
		}
	}

	if ( valid )
	{
		for ( auto &m : primitives )
		{
			if ( shaders[ m->shaderId ].template is<ext::Emission>() )
			{
				auto &em = static_cast<const ext::Emission &>( *shaders[ m->shaderId ] );
				lights.emplace_back( poly::make_object<AreaLight>( em.emission, m ) );
				if ( auto *p = dynamic_cast<Mesh *>( &*m ) )
				{
					p->generateSamples();
				}
				KLOG( shaderNames[ m->shaderId ], "is area light" );
			}
		}
		KLOG( lights.size(), "lights in scene" );
	}

	valid = true;
}

}  // namespace core

}  // namespace koishi
