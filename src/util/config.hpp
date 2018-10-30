#pragma once

#include <vector>
#include <vec/vec.hpp>
#include <vec/vsel.hpp>
#include <util/jsel.hpp>

namespace koishi
{
namespace jsel
{
struct Serializable( Camera )
{
	Property( float, fovy, 60 );
	Property( float3, position, { 1, 0, 0 } );
	Property( float3, target, { -1, 0, 0 } );
};

struct Serializable( Mesh )
{
	Property( std::string, path );
	Property( float3, transform, { 0, 0, 0 } );
	Property( float, scale, 1 );
};

struct Serializable( Scene )
{
	Property( std::vector<Camera>, camera, {} );
	Property( std::vector<Mesh>, mesh, {} );
};

}  // namespace jsel

}  // namespace koishi
