#pragma once

#include <vec/vec.hpp>
#include "mesh.hpp"
#include "ray.hpp"
#include "dev.hpp"

namespace koishi
{
namespace core
{
template <typename Random>
PolyFunction( Radiance, Require<Random> )(
  ( const core::Ray &r, const poly::vector<poly::SubMesh> &mesh, uint depth = 0 ) {
	  const core::SubMesh *pm;
	  core::Hit hit;
	  for ( auto &m : mesh )
	  {
		  core::Hit hit1;
		  if ( r.intersect( m, 1, hit1 ) && hit1.t < hit.t )
		  {
			  hit = hit1;
			  pm = &m;
		  }
	  }
	  if ( hit )
	  {
		  auto n = core::interplot( pm->normals[ pm->indices[ hit.id ] ],
									pm->normals[ pm->indices[ hit.id + 1 ] ],
									pm->normals[ pm->indices[ hit.id + 2 ] ],
									hit.uv );
		  // if ( dot( n, r.v ) > 0 ) n = -n;
		  auto nr = r.reflect( hit.t, n );
		  // return normalize( nr.v );
		  auto r1 = 2 * M_PI * Random::rand(), r2 = Random::rand(), r2s = sqrt( r2 );
		  auto u = normalize( cross( abs( nr.v.x ) > abs( nr.v.y ) ? double3{ 0, 1, 0 } : double3{ 1, 0, 0 }, nr.v ) );
		  auto v = cross( nr.v, u );
		  auto dr = nr;
		  dr.v = normalize( ( u * cos( r1 ) + v * sin( r1 ) ) * r2s + nr.v * sqrt( 1 - r2s ) );
		  // return nr.v;  //( r.o + r.v * hit.t ) / 100.;
		  if ( depth < 100 )
			  if ( Random::rand() < .9 )
				  return pm->emissive +
						 pm->color *
						   radiance( dr, mesh, depth + 1 );
			  else
				  return pm->emissive +
						 pm->color *
						   radiance( nr, mesh, depth + 1 );
		  // return  //pm->emissive +
		  // 		//pm->color *
		  //   radiance( nr, mesh, depth + 1 );
		  else
			  // return ( r.o + r.v * hit.t ) / 100.;
			  return pm->emissive;
		  // radiance( nr, mesh );
	  }
	  else
	  {
		  return double3{ 0.f, 0.f, 0.f };
	  }
  } );

}  // namespace core

}  // namespace koishi
