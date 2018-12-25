#pragma once

#include <core/basic/basic.hpp>
#include <core/meta/varyings.hpp>

namespace koishi
{
namespace core
{
struct Scene;

struct Light : emittable
{
	KOISHI_HOST_DEVICE virtual solid sample( const Scene &scene,
											 const LocalVaryings &res,
											 const float2 &u, float3 &li,
											 Allocator &pool ) const = 0;
};

}  // namespace core

}  // namespace koishi