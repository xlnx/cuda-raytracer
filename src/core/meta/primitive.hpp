#pragma once

#include <core/basic/poly.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/allocator.hpp>
#include <core/meta/input.hpp>
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

	KOISHI_HOST_DEVICE virtual LocalInput sample(
	  const float2 &u, float &pdf ) const = 0;
	KOISHI_HOST_DEVICE virtual LocalInput sample(
	  const float3 &p, const float2 &u, float &pdf ) const
	{
		LocalInput input = sample( u, pdf );
		auto wi = p - input.p;
		auto d2 = squaredLength( wi );
		pdf *= d2 / abs( dot( input.n, normalize( wi ) ) );
		return input;
	}

public:
	uint matid;
};

}  // namespace core

}  // namespace koishi