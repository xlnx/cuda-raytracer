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
	Property( float, fovy );
	Property( float3, position );
	Property( float3, target );
};

struct Serializable( Object )
{
	Property( std::string, type );
	Property( std::string, path );
};

struct Serializable( Scene )
{
	Property( std::vector<Camera>, camera );
	Property( std::vector<Object>, object );
};

}  // namespace jsel

}  // namespace koishi
