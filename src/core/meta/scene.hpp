#pragma once

#include <string>
#include <vector>
#include <core/basic/basic.hpp>
#include <core/light/light.hpp>
#include <core/primitive/primitive.hpp>
#include <core/nodes/shader.hpp>
#include "varyings.hpp"

namespace koishi
{
namespace core
{
struct Scene : emittable
{
	struct Configuration : serializable<Configuration>
	{
		using TConfig = std::map<std::string, Config>;
		Property( std::vector<CameraConfig>, camera, {} );
		Property( std::vector<Config>, assets );
		Property( TConfig, shaders );
	};

	Scene( const Properties &props );

	poly::vector<poly::object<Primitive>> primitives;
	poly::vector<poly::object<Light>> lights;
	poly::vector<poly::object<Shader>> shaders;

	poly::vector<CameraConfig> camera;

	operator bool() const { return valid; }

	KOISHI_HOST_DEVICE bool intersect( const Ray &r, LocalVaryings &varyings, Allocator &pool ) const
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
		varyings.n = ( *pm )->normal( hit );
		varyings.u = normalize( cross( varyings.n, float3{ 0, 1, 0 } ) );
		varyings.v = normalize( cross( varyings.n, varyings.u ) );
		varyings.p = r.o + r.d * hit.t;
		varyings.wo = varyings.local( -r.d );
		varyings.shaderId = ( *pm )->shaderId;
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
