#pragma once

#include <string>
#include <vector>
#include <core/basic/poly.hpp>
#include "material.hpp"
#include "interreact.hpp"

namespace koishi
{
namespace core
{
struct Scene : emittable
{
	Scene( const std::string &path );

	poly::vector<poly::object<Primitive>> primitives;
	poly::vector<poly::object<Material>> material;

	poly::vector<CameraConfig> camera;

	operator bool() const { return valid; }

	KOISHI_HOST_DEVICE Interreact intersect( const Ray &r, Allocator &pool ) const
	{
		poly::object<Primitive> const *pm = nullptr;
		Hit hit;
		for ( auto it = primitives.begin(); it != primitives.end(); ++it )
		{
			Hit hit1;
			if ( ( *it )->intersect( r, hit1, pool ) && hit1.t < hit.t )
			{
				hit = hit1, pm = it;
			}
			pool.clear();
		}
		Interreact res;
		if ( hit )
		{
			res.isNull = false;
			res.n = ( *pm )->normal( hit );
			res.p = r.o + r.d * hit.t;
			res.uv = hit.uv;
			res.u = cross( res.n, float3{ 0, 1, 0 } );
			res.v = cross( res.n, res.u );
			material[ ( *pm )->matid ]->apply( res, pool );
			// pm->material->apply( res, pool );
		}
		return res;
	}

	KOISHI_HOST_DEVICE bool intersect( const Seg &s, Allocator &pool ) const
	{
		for ( auto &m : primitives )
		{
			if ( m->intersect( s, pool ) )
			{
				return true;
			}
			pool.clear();
		}
		return false;
	}

private:
	bool valid = false;
};

}  // namespace core

}  // namespace koishi
