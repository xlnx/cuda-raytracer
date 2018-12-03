#pragma once

#include <string>
#include <vector>
#include <core/basic/poly.hpp>
#include "mesh.hpp"
#include "material.hpp"
#include "interreact.hpp"

namespace koishi
{
namespace core
{
struct Scene : emittable
{
	Scene( const std::string &path );

	poly::vector<Mesh> mesh;
	poly::vector<poly::object<Material>> material;

	poly::vector<CameraConfig> camera;

	operator bool() const { return valid; }

	KOISHI_HOST_DEVICE Interreact intersect( const Ray &r, Allocator &pool ) const
	{
		poly::vector<Mesh>::const_iterator pm = nullptr;
		Hit hit;
		for ( auto it = mesh.begin(); it != mesh.end(); ++it )
		{
			Hit hit1;
			if ( it->intersect( r, hit1, pool ) && hit1.t < hit.t )
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
			res.n = core::interplot( pm->normals[ hit.id ].n0,
									 pm->normals[ hit.id ].n1,
									 pm->normals[ hit.id ].n2,
									 hit.uv );
			res.p = r.o + r.d * hit.t;
			res.uv = hit.uv;
			res.u = cross( res.n, float3{ 0, 1, 0 } );
			res.v = cross( res.n, res.u );
			material[ pm->matid ]->apply( res, pool );
			// pm->material->apply( res, pool );
		}
		return res;
	}

private:
	bool valid = false;
};

}  // namespace core

}  // namespace koishi
