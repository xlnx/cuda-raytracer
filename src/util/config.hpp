#pragma once

#include <map>
#include <vector>
#include <string>
#include <vec/vec.hpp>
#include <core/basic/poly.hpp>
#include <util/jsel.hpp>

namespace koishi
{
namespace jsel
{
struct Camera : Serializable<Camera>, core::Emittable<Camera>
{
	Property( float, fovx, 90 );
	Property( float, aspect, 0 );
	Property( float3, position, { 1, 0, 0 } );
	Property( float3, target, { -1, 0, 0 } );
	Property( float3, upaxis, { 0, 0, 1 } );
	Property( float, zNear, 1e-3 );
	Property( float, zFar, 1e5 );
};

struct Material : Serializable<Material>, core::Emittable<Material>
{
};

struct Scene : Serializable<Scene>
{
	using TMaterials = std::map<std::string, jsel::Material>;
	Property( std::vector<Camera>, camera, {} );
	Property( std::string, path );
	//Property( TMaterials, material );
};

}  // namespace jsel

}  // namespace koishi
