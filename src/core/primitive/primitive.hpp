#pragma once

#include <core/basic/basic.hpp>
#include <core/meta/varyings.hpp>

namespace koishi
{
namespace core
{
struct Primitive : emittable
{
	KOISHI_HOST_DEVICE virtual normalized_float3 normal(
	  const Hit &hit ) const = 0;
	KOISHI_HOST_DEVICE virtual bool intersect(
	  const Ray &ray, Hit &hit, Allocator &pool ) const = 0;
	KOISHI_HOST_DEVICE virtual bool intersect(
	  const Seg &seg, Allocator &pool ) const
	{
		Hit hit;
		return intersect( seg, hit, pool ) && hit.t <= seg.t;
	}

	KOISHI_HOST_DEVICE virtual LocalVaryings sample(
	  const float2 &u, float &pdf ) const = 0;
	KOISHI_HOST_DEVICE virtual LocalVaryings sample(
	  const float3 &p, const float2 &u, float &pdf ) const
	{
		LocalVaryings varyings = sample( u, pdf );
		auto wi = p - varyings.p;
		auto d2 = squaredLength( wi );
		pdf *= d2 / fabs( dot( varyings.n, normalize( wi ) ) );
		return varyings;
	}

public:
	uint shaderId;
};

}  // namespace core

}  // namespace koishi