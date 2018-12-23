#pragma once

#include <map>
#include <vector>
#include <string>
#include <vec/vec.hpp>
#include <poly/kernel.hpp>
#include <util/jsel.hpp>

namespace koishi
{
struct CameraConfig : serializable<CameraConfig>, emittable
{
	Property( std::string, lens, "pinhole" );
	Property( float, fovx, 90 );
	Property( float, aspect, 0 );
	Property( float3, position, { 1, 0, 0 } );
	Property( float3, target, { -1, 0, 0 } );
	Property( float3, upaxis, { 0, 0, 1 } );
	Property( float, zNear, 1e-3 );
	Property( float, zFar, 1e5 );
};

using Properties = std::map<std::string, nlohmann::json>;

struct Config : serializable<Config, as_array>, emittable
{
	Property( std::string, name );
	Property( Properties, props, {} );
};

struct SceneConfig : serializable<SceneConfig>
{
	using TConfig = std::map<std::string, Config>;
	Property( std::vector<CameraConfig>, camera, {} );
	Property( std::vector<Config>, assets );
	Property( TConfig, material );
};

}  // namespace koishi
