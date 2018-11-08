#pragma once

#include <vec/vmath.hpp>
#include <core/mesh.hpp>
#include <core/allocator.hpp>
#include "material.hpp"

namespace koishi
{
namespace core
{
namespace dev
{
struct Mesh
{
	const double3 *vertices;
	const double3 *normals;
	const uint *indices;
	const BVHNode *bvh;
	const Material *material;

public:
	Mesh() = delete;
	Mesh( const core::Mesh &other ) :
	  vertices( &other.vertices[ 0 ] ),
	  normals( &other.normals[ 0 ] ),
	  indices( &other.indices[ 0 ] ),
	  bvh( &other.bvh[ 0 ] ),
	  material( nullptr )
	{
	}

	KOISHI_HOST_DEVICE bool intersect( const core::Ray &ray, uint root, core::Hit &hit ) const
	{
		uint i = root;
		while ( !bvh[ i ].isleaf )
		{
			auto left = ray.intersect_bbox( bvh[ i << 1 ].vmin, bvh[ i << 1 ].vmax );
			auto right = ray.intersect_bbox( bvh[ ( i << 1 ) + 1 ].vmin, bvh[ ( i << 1 ) + 1 ].vmax );
			if ( !left && !right ) return false;
			if ( left && right )
			{
				core::Hit hit1;
				auto b0 = intersect( ray, root << 1, hit );
				auto b1 = intersect( ray, ( root << 1 ) | 1, hit1 );
				if ( !b0 && !b1 )
				{
					return false;
				}
				if ( !b0 || b1 && hit1.t < hit.t )
				{
					hit = hit1;
				}
				return true;
			}
			i <<= 1;
			if ( right ) i |= 1;
		}
		// return true;
		hit.t = INFINITY;
		for ( uint j = bvh[ i ].begin; j < bvh[ i ].end; j += 3 )
		{
			core::Hit hit1;
			if ( ray.intersect_triangle( vertices[ indices[ j ] ],
										 vertices[ indices[ j + 1 ] ],
										 vertices[ indices[ j + 2 ] ], hit1 ) &&
				 hit1.t < hit.t )
			{
				hit = hit1;
				hit.id = j;
			}
		}
		return hit.t != INFINITY;
	}
};

}  // namespace dev

}  // namespace core

}  // namespace koishi