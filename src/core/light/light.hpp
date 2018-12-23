#pragma once

#include <core/basic/poly.hpp>
#include <core/meta/interreact.hpp>

namespace koishi
{
namespace core
{
struct Scene;

struct Light : emittable
{
	KOISHI_HOST_DEVICE virtual solid sample( const Scene &scene,
											 const Interreact &res,
											 const float2 &u, float3 &li,
											 Allocator &pool ) const = 0;
};

}  // namespace core

}  // namespace koishi