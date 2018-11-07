#pragma once

#include <vector>
#include <string>
#include <vec/vec.hpp>
#include <util/jsel.hpp>

namespace koishi
{
namespace jsel
{
struct Serializable( Camera )
{
	Property( double, fovx, 90 );
	Property( double, aspect, 0 );
	Property( double3, position, { 1, 0, 0 } );
	Property( double3, target, { -1, 0, 0 } );
	Property( double3, upaxis, { 0, 0, 1 } );
	Property( double, zNear, 1e-3 );
	Property( double, zFar, 1e5 );
};

struct Serializable( Rotate )
{
	Property( double3, axis );
	Property( double, degree );
};

struct Serializable( Mesh )
{
	Property( std::string, path );
	Property( double3, translate, { 0, 0, 0 } );
	Property( std::vector<Rotate>, rotate, {} );
	Property( double, scale, 1 );
	Property( double3, emissive, { 0, 0, 0 } );
	Property( double3, color, { 0, 0, 0 } );
};

struct Serializable( Scene )
{
	Property( std::vector<Camera>, camera, {} );
	Property( std::vector<Mesh>, mesh, {} );
};

}  // namespace jsel

}  // namespace koishi
