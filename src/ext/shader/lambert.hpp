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
	  distribution( Factory<SphericalDistribution>::create( Config(
		"Cosine", {} ) ) )
	{
	}

	KOISHI_HOST_DEVICE void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, uint target ) const override
	{
		switch ( target )
		{
		case sample_wi_f_by_wo:
			float pdf;
			varyings.wi = distribution->sample( sampler.sample3(), pdf );
			varyings.f = R * distribution->f( varyings.wi ) / pdf;
			break;
		case compute_f_by_wi_wo:
			varyings.f = R * distribution->f( varyings.wi );
			break;
		}
	}

	void writeNode( json &j ) const override
	{
		j[ "Lambert" ][ "R" ] = R;
	}

private:
	float3 R;
	poly::object<SphericalDistribution> distribution;
};

}  // namespace ext

}  // namespace koishi