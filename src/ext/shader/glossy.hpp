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
	  Varyings &varyings, Sampler &sampler, Allocator &pool, uint target ) const override
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

	void print( std::ostream &os ) const override
	{
		nlohmann::json json = {
			{ "Glossy", {} }
		};
		os << json.dump();
	}

private:
	poly::object<SphericalDistribution> distribution;
	float3 color;
};

// struct MicrofacetTransmission : BxDF
// {
// 	KOISHI_HOST_DEVICE MicrofacetTransmission( const poly::object<SphericalDistribution> &distribution ) :
// 	  distribution( distribution )
// 	{
// 	}

// 	KOISHI_HOST_DEVICE float3 f( const solid &wo, const solid &wi ) const override
// 	{
// 		// return H::isSame( wo, wi ) ? R : float3{ 0, 0, 0 };
// 		return float3{ 1, 1, 1 };
// 	}

// 	KOISHI_HOST_DEVICE solid sample( const solid &wo, const float3 &u, float3 &f ) const override
// 	{
// 		float pdf;
// 		auto wh = distribution->sample( u, pdf );
// 		auto wi = refract( wo, wh, 0.7 );
// 		// f = H::isSame( wo, wi ) ? float3{ 1, 1, 1 } : float3{ 0, 0, 0 };
// 		return solid( wi );
// 	}

// private:
// 	const poly::object<SphericalDistribution> &distribution;
// };

}  // namespace ext

}  // namespace koishi