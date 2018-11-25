#pragma once

#include <string>
#include <vector>
#include <util/config.hpp>
#include <core/basic/poly.hpp>
#include "mesh.hpp"
#include "interreact.hpp"

namespace koishi
{
namespace core
{
struct Scene : Emittable<Scene>
{
	Scene( const std::string &path );

	PolyVector<Mesh> mesh;
	PolyVector<jsel::Material> material;

	PolyVector<jsel::Camera> camera;

	operator bool() const { return valid; }

	KOISHI_HOST_DEVICE Interreact intersect( const Ray &r, Allocator &pool ) const
	{
		PolyVector<Mesh>::const_iterator pm = nullptr;
		Hit hit;
		for ( auto it = mesh.begin(); it != mesh.end(); ++it )
		{
			Hit hit1;
			if ( it->intersect( r, 1, hit1, pool ) && hit1.t < hit.t )
			{
				hit = hit1, pm = it;
			}
			pool.clear();
		}
		Interreact res;
		if ( hit )
		{
			res.mesh = pm;
			res.isNull = false;
			res.n = core::interplot( pm->faces[ hit.id ].n0,
									 pm->faces[ hit.id ].n1,
									 pm->faces[ hit.id ].n2,
									 hit.uv );
			res.p = r.o + r.d * hit.t;
			res.uv = hit.uv;
			res.u = cross( res.n, float3{ 0, 1, 0 } );
			res.v = cross( res.n, res.u );
			// pm->material->fetchTo( res, pool );
		}
		return res;
	}

private:
	bool valid = false;
};

}  // namespace core

}  // namespace koishi
