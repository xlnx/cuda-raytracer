#pragma once

#include <vec/vec.hpp>
#include <core/basic/ray.hpp>
#include <core/basic/poly.hpp>
#include <core/basic/allocator.hpp>
#include <core/meta/mesh.hpp>
#include <core/meta/scene.hpp>

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

// template <typename Random>
// PolyFunction( SampleBsdf, Require<Random> )(
//   ( const double3 &wo )->double3 {
// 	  double3 w;
// 	  auto r1 = 2 * M_PI * call<Random>(), r2 = call<Random>(), r2s = sqrt( r2 );
// 	  auto u = normalize( cross( abs( wo.x ) > abs( wo.y ) ? double3{ 0, 1, 0 } : double3{ 1, 0, 0 }, wo ) );
// 	  auto v = cross( wo, u );
// 	  return normalize( ( u * cos( r1 ) + v * sin( r1 ) ) * r2s + wo * sqrt( 1 - r2s ) );
//   } );

template <typename Random>
PolyFunction( Radiance, Require<Random> )(
  ( const Ray &r, const Scene &scene, Allocator &pool )->double3 {
	  auto ray = r;
	  Interreact isect;
	  double3 L = { 0, 0, 0 };
	  constexpr auto maxBounce = 10;

	  for ( auto bounce = 0; ( isect = scene.intersect( ray, pool ) ) && bounce != maxBounce; ++bounce )
	  {
		  // auto spec = reflect( ray.d, isect.n );
		  // auto diff = call<SampleBsdf<Random>>( spec );

		  // L += isect.mesh->emissive;

		  // ray = isect.emitRay( call<Random>() < .9 ? diff : spec );
		  // clear( pool );

		  auto wo = isect.local( -ray.d );
		  L = wo;
		  break;
		  pool.clear();
	  }

	  return L;
  } );

}  // namespace core

}  // namespace koishi