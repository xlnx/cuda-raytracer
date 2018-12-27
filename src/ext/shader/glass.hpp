#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct Glass : Shader
{
	Glass( const Properties &props ) :
	  Shader( props ),
	  color( get( props, "color", float3{ 1, 1, 1 } ) ),
	  ior( get<float>( props, "ior" ) ),
	  fac( Factory<Scala<float>>::create(
		Config( "Fresnel", Properties{ { "ior", ior } } ) ) ),
	  distribution( Factory<SphericalDistribution>::create(
		get<Config>( props, "distribution" ) ) )
	{
	}

	KOISHI_HOST_DEVICE void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, ShaderTarget target ) const override
	{
		if ( sampler.sample() < fac->compute( varyings, pool ) )
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
		else
		{
			switch ( target )
			{
			case sample_wi_f_by_wo:
			{
				auto h = distribution->sample( sampler.sample3() );
				auto e = varyings.wo.z > 0 ? 1.f / ior : ior;
				if ( varyings.wo.z < 0 ) h = -h;
				if ( refract( varyings.wo, varyings.wi, h, e ) )
					varyings.f = color / ( e < 1.f ? e * e : 1.f );
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
	}

	void writeNode( json &j ) const override
	{
		j[ "Glass" ][ "color" ] = color;
		j[ "Glass" ][ "ior" ] = ior;
		distribution->writeNode( j[ "Glass" ][ "distribution" ] );
	}

private:
	float3 color;
	float ior;
	poly::object<Scala<float>> fac;
	poly::object<SphericalDistribution> distribution;
};

}  // namespace ext

}  // namespace koishi