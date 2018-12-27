#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct Glossy : Shader
{
	Glossy( const Properties &props ) :
	  Shader( props ),
	  distribution( Factory<SphericalDistribution>::create(
		get<Config>( props, "distribution" ) ) ),
	  color( get( props, "color", float3{ 1, 1, 1 } ) )
	{
	}

	KOISHI_HOST_DEVICE void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, ShaderTarget target ) const override
	{
		switch ( target )
		{
		case sample_wi_f_by_wo:
			varyings.wi = reflect( varyings.wo, distribution->sample( sampler.sample3() ) );
			varyings.f = H::isSame( varyings.wo, varyings.wi ) ? color : float3{ 0, 0, 0 };
			break;
		case compute_f_by_wi_wo:
			varyings.f = color * distribution->f( normalize( varyings.wo + varyings.wi ) );
			break;
		}
	}

	void writeNode( json &j ) const override
	{
		j[ "Glossy" ][ "color" ] = color;
		distribution->writeNode( j[ "Glossy" ][ "distribution" ] );
	}

private:
	poly::object<SphericalDistribution> distribution;
	float3 color;
};

}  // namespace ext

}  // namespace koishi