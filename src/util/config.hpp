#pragma once

#include <vector>
#include <vec/vec.hpp>
#include <util/jsel.hpp>

namespace koishi
{
namespace jsel
{
struct Serializable( Camera )
{
	Property( double, fovy, 60 );
	Property( double3, position, { 1, 0, 0 } );
	Property( double3, target, { -1, 0, 0 } );
};

struct Serializable( Mesh )
{
	Property( std::string, path );
	Property( double3, translate, { 0, 0, 0 } );
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
