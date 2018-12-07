#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct MicrofacetBxDF : BxDF
{
	KOISHI_HOST_DEVICE MicrofacetBxDF( const poly::object<SphericalDistribution> &distribution ) :
	  distribution( distribution )
	{
	}

	KOISHI_HOST_DEVICE float3 f( const float3 &wo, const float3 &wi ) const override
	{
	}
	KOISHI_HOST_DEVICE float3 sample( const float3 &wo, const float3 &rn, float &pdf ) const override
	{
		return -reflect( wo, distribution->sample( rn, pdf ) );
	}

private:
	const poly::object<SphericalDistribution> &distribution;
};

struct MicrofacetMaterial : Material
{
	MicrofacetMaterial( const Properties &props ) :
	  distribution( Factory<SphericalDistribution>::create( get<Config>( props, "distribution" ) ) )
	{
	}

	KOISHI_HOST_DEVICE virtual void apply( Interreact &res, Allocator &pool ) const
	{
		res.bsdf = create<BSDF>( pool );
		res.bsdf->add<MicrofacetBxDF>( pool, distribution );
	}

	void print( std::ostream &os ) const
	{
		nlohmann::json json = {
			{ "Microfacet", {} }
		};
		os << json.dump();
	}

private:
	poly::object<SphericalDistribution> distribution;
};

}  // namespace ext

}  // namespace koishi