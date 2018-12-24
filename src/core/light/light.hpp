#pragma once

#include <core/basic/poly.hpp>
#include <core/meta/input.hpp>

namespace koishi
{
namespace core
{
struct Scene;

struct Light : emittable
{
	KOISHI_HOST_DEVICE virtual solid sample( const Scene &scene,
											 const LocalInput &res,
											 const float2 &u, float3 &li,
											 Allocator &pool ) const = 0;
};

}  // namespace core

}  // namespace koishi