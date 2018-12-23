#pragma once

#include <core/basic/poly.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/allocator.hpp>
#include <core/meta/interreact.hpp>
#include <util/hemisphere.hpp>

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

	KOISHI_HOST_DEVICE virtual Interreact sample(
	  const float2 &u, float &pdf ) const = 0;
	KOISHI_HOST_DEVICE virtual Interreact sample(
	  const float3 &p, const float2 &u, float &pdf ) const
	{
		Interreact isect = sample( u, pdf );
		auto wi = p - isect.p;
		auto d2 = squaredLength( wi );
		pdf *= d2 / abs( dot( isect.n, normalize( wi ) ) );
		return isect;
	}

public:
	uint matid;
};

}  // namespace core

}  // namespace koishi