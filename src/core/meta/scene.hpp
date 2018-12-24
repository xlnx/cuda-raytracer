#pragma once

#include <string>
#include <vector>
#include <core/basic/poly.hpp>
#include <core/light/light.hpp>
#include "material.hpp"
#include "input.hpp"

namespace koishi
{
namespace core
{
struct Scene : emittable
{
	Scene( const std::string &path );

	poly::vector<poly::object<Primitive>> primitives;
	poly::vector<poly::object<Light>> lights;
	poly::vector<poly::object<Material>> material;

	poly::vector<CameraConfig> camera;

	operator bool() const { return valid; }

	KOISHI_HOST_DEVICE bool intersect( const Ray &r, Input &input, Allocator &pool ) const
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
		}
		if ( !hit ) return false;
		input.n = ( *pm )->normal( hit );
		input.p = r.o + r.d * hit.t;
		input.u = normalize( cross( input.n, float3{ 0, 1, 0 } ) );
		input.v = normalize( cross( input.n, input.u ) );
		input.matid = ( *pm )->matid;
		input.wo = input.local( -r.d );
		input.color = input.emissive = float3{ 0, 0, 0 };
		material[ input.matid ]->apply( input, pool );
		return true;
	}

	KOISHI_HOST_DEVICE bool intersect( const Seg &s, Allocator &pool ) const
	{
		for ( auto &m : primitives )
		{
			if ( m->intersect( s, pool ) )
			{
				return true;
			}
		}
		return false;
	}

private:
	bool valid = true;
};

}  // namespace core

}  // namespace koishi
