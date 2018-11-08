#pragma once

#include <vec/vec.hpp>
#include <core/device/scene.hpp>
#include "mesh.hpp"
#include "ray.hpp"
#include "poly.hpp"
#include "allocator.hpp"

namespace koishi
{
namespace core
{
template <typename Random>
PolyFunction( CosSampleHemisphere, Require<Random> )(
  ()->double3 {
	  double3 w;
	  auto r1 = 2 * M_PI * call<Random>(), r2 = call<Random>(), r2s = sqrt( r2 );

	  return w;
  } );

template <typename Random>
PolyFunction( SampleBsdf, Require<Random> )(
  ( const double3 &wo )->double3 {
	  double3 w;
	  auto r1 = 2 * M_PI * call<Random>(), r2 = call<Random>(), r2s = sqrt( r2 );
	  auto u = normalize( cross( abs( wo.x ) > abs( wo.y ) ? double3{ 0, 1, 0 } : double3{ 1, 0, 0 }, wo ) );
	  auto v = cross( wo, u );
	  return normalize( ( u * cos( r1 ) + v * sin( r1 ) ) * r2s + wo * sqrt( 1 - r2s ) );
  } );

template <typename Random>
PolyFunction( Radiance, Require<SampleBsdf<Random>> )(
  ( const core::Ray &r, dev::Scene *scene, Allocator &pool )->double3 {
	  auto ray = r;
	  dev::Hit hit;
	  double3 L = { 0, 0, 0 };
	  constexpr auto maxBounce = 10;

	  for ( auto bounce = 0; ( hit = scene->intersect( ray, pool ) ) && bounce != maxBounce; ++bounce )
	  {
		  auto spec = reflect( ray.d, hit.n );
		  auto diff = call<SampleBsdf<Random>>( spec );

		  // L += hit.mesh->emissive;

		  ray = hit.emitRay( call<Random>() < .9 ? diff : spec );
		  // clear( pool );
	  }

	  return L;
  } );

}  // namespace core

}  // namespace koishi
