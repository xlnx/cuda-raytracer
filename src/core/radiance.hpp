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
  ( const core::Ray &r, const dev::Mesh *mesh, uint N, uint depth = 0 )->double3 {
	  const dev::Mesh *pm;
	  core::Hit hit;
	  for ( uint i = 0; i != N; ++i )
	  {
		  core::Hit hit1;
		  if ( r.intersect( mesh[ i ], 1, hit1 ) && hit1.t < hit.t )
		  {
			  hit = hit1;
			  pm = mesh + i;
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
		  auto r1 = 2 * M_PI * call<Random>(), r2 = call<Random>(), r2s = sqrt( r2 );
		  auto u = normalize( cross( abs( nr.v.x ) > abs( nr.v.y ) ? double3{ 0, 1, 0 } : double3{ 1, 0, 0 }, nr.v ) );
		  auto v = cross( nr.v, u );
		  auto dr = nr;
		  dr.v = normalize( ( u * cos( r1 ) + v * sin( r1 ) ) * r2s + nr.v * sqrt( 1 - r2s ) );
		  // return nr.v;  //( r.o + r.v * hit.t ) / 100.;
		  if ( depth < 100 )
			  if ( call<Random>() < .9 )
				  return pm->emissive +
						 pm->color *
						   call<Radiance>( dr, mesh, depth + 1 );
			  else
				  return pm->emissive +
						 pm->color *
						   call<Radiance>( nr, mesh, depth + 1 );
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
