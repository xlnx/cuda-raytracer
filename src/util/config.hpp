#pragma once

#include <map>
#include <vector>
#include <string>
#include <vec/vec.hpp>
#include <poly/kernel.hpp>
#include <util/jsel.hpp>

namespace koishi
{
struct CameraConfig : serializable<CameraConfig>, emittable<CameraConfig>
{
	Property( float, fovx, 90 );
	Property( float, aspect, 0 );
	Property( float3, position, { 1, 0, 0 } );
	Property( float3, target, { -1, 0, 0 } );
	Property( float3, upaxis, { 0, 0, 1 } );
	Property( float, zNear, 1e-3 );
	Property( float, zFar, 1e5 );
};

struct MaterialConfig : serializable<MaterialConfig>, emittable<MaterialConfig>
{
};

struct SceneConfig : serializable<SceneConfig>
{
	using TMaterials = std::map<std::string, MaterialConfig>;
	Property( std::vector<CameraConfig>, camera, {} );
	Property( std::string, path );
	//Property( TMaterials, material );
};

}  // namespace koishi
