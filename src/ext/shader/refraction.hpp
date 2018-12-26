#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct Refraction : Shader
{
	Refraction( const Properties &props ) :
	  Shader( props ),
	  distribution( Factory<SphericalDistribution>::create(
		get<Config>( props, "distribution" ) ) ),
	  color( get( props, "color", float3{ 1, 1, 1 } ) ),
	  ior( get<float>( props, "ior" ) )
	{
	}

	KOISHI_HOST_DEVICE void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, uint target ) const override
	{
		switch ( target )
		{
		case sample_wi_f_by_wo:
		{
			auto h = distribution->sample( sampler.sample3() );
			auto e = varyings.wo.z > 0 ? 1.f / ior : ior;
			if ( varyings.wo.z < 0 ) h = -h;
			if ( refract( varyings.wo, varyings.wi, h, e ) )
				varyings.f = color;
			else
				varyings.f = float3{ 0, 0, 0 };
		}
		break;
		case compute_f_by_wi_wo:
			varyings.f = float3{ 0, 0, 0 };
			// varyings.f = color * distribution->f( normalize( varyings.wo + varyings.wi * ior ) );
			break;
		}
	}

	void writeNode( json &j ) const override
	{
		j[ "Refraction" ][ "color" ] = color;
		distribution->writeNode( j[ "Refraction" ][ "distribution" ] );
	}

private:
	poly::object<SphericalDistribution> distribution;
	float3 color;
	float ior;
};

}  // namespace ext

}  // namespace koishi