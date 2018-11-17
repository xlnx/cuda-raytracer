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
	Property( double, fovx, 90 );
	Property( double, aspect, 0 );
	Property( double3, position, { 1, 0, 0 } );
	Property( double3, target, { -1, 0, 0 } );
	Property( double3, upaxis, { 0, 0, 1 } );
	Property( double, zNear, 1e-3 );
	Property( double, zFar, 1e5 );
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
