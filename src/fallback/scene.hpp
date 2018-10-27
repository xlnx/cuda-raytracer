#pragma once

#include <vector>
#include <vec/vec.hpp>
#include <vec/vsel.hpp>
#include <util/jsel.hpp>

namespace koishi
{
struct Serializable( Camera )
{
	Property( float, fovy );
	Property( float3, position );
};

struct Serializable( Scene )
{
	Property( std::vector<Camera>, camera );
};

}  // namespace koishi
