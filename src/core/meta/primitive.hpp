#pragma once

#include <core/basic/poly.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/allocator.hpp>

namespace koishi
{
namespace core
{
struct Primitive : emittable
{
	KOISHI_HOST_DEVICE virtual float3 normal(
	  const Hit &hit ) const = 0;
	KOISHI_HOST_DEVICE virtual bool intersect(
	  const Ray &ray, Hit &hit, Allocator &pool ) const = 0;
	KOISHI_HOST_DEVICE virtual bool intersect(
	  const Seg &seg, Allocator &pool ) const = 0;

public:
	uint matid;
};

}  // namespace core

}  // namespace koishi