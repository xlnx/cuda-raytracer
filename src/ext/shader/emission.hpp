#pragma once

#include <ext/util.hpp>
#include "lambert.hpp"

namespace koishi
{
namespace ext
{
struct Emission : Shader
{
	Emission( const Properties &props ) :
	  Shader( props ),
	  color( get( props, "color", float3{ 1, 1, 1 } ) ),
	  strength( get( props, "strength", 2.f ) ),
	  lambert( Factory<Shader>::create( Config(
		"Lambert", {} ) ) ),
	  emission( color * strength )
	{
	}

	KOISHI_HOST_DEVICE void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, uint target ) const override
	{
		varyings.emission = emission;
		lambert->execute( varyings, sampler, pool, target );
	}

	void writeNode( json &j ) const override
	{
		j[ "Emission" ][ "color" ] = color;
		j[ "Emission" ][ "strength" ] = strength;
	}

private:
	float3 color;
	float strength;
	poly::object<Shader> lambert;

public:
	const float3 emission;
};

}  // namespace ext

}  // namespace koishi
