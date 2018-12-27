#pragma once

#include <ext/util.hpp>
#include <ext/distribution/cosine.hpp>

namespace koishi
{
namespace ext
{
struct Lambert : Shader
{
	Lambert( const Properties &props ) :
	  Shader( props ),
	  R( get( props, "R", float3{ 1, 1, 1 } ) ),
	  color( get( props, "color", float3{ 1, 1, 1 } ) ),
	  distribution( Factory<SphericalDistribution>::create( Config(
		"Cosine", {} ) ) )
	{
	}

	KOISHI_HOST_DEVICE void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, ShaderTarget target ) const override
	{
		switch ( target )
		{
		case sample_wi_f_by_wo:
			float pdf;
			varyings.wi = distribution->sample( sampler.sample3(), pdf );
			varyings.f = color * R * distribution->f( varyings.wi ) / pdf;
			break;
		case compute_f_by_wi_wo:
			varyings.f = color * R * distribution->f( varyings.wi );
			break;
		}
	}

	void writeNode( json &j ) const override
	{
		j[ "Lambert" ][ "R" ] = R;
		j[ "Lambert" ][ "color" ] = color;
	}

private:
	float3 R, color;
	poly::object<SphericalDistribution> distribution;
};

}  // namespace ext

}  // namespace koishi