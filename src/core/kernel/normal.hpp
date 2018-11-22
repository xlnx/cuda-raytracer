#pragma once

#include <vec/vec.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/poly.hpp>
#include <core/basic/allocator.hpp>
#include <core/meta/scene.hpp>

namespace koishi
{
namespace core
{
template <typename Random>
PolyFunction( Normal, Require<Random> )(
  ( const Ray &r, const Scene &scene, Allocator &pool )->double3 {
	  double3 L = { 0, 0, 0 };

	  if ( auto isect = scene.intersect( r, pool ) )
	  {
		  auto wo = isect.local( -r.d );
		  L = wo;
	  }

	  return L;
  } );

}  // namespace core

}  // namespace koishi