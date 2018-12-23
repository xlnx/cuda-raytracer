#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct MicrofacetReflection : BxDF
{
	KOISHI_HOST_DEVICE MicrofacetReflection( const poly::object<SphericalDistribution> &distribution ) :
	  distribution( distribution )
	{
	}

	KOISHI_HOST_DEVICE float3 f( const solid &wo, const solid &wi ) const override
	{
		return distribution->f( normalize( ( wo + wi ) * .5f ) );
	}

	KOISHI_HOST_DEVICE solid sample( const solid &wo, const float3 &u, float3 &f ) const override
	{
		float pdf;
		auto wi = reflect( wo, distribution->sample( u, pdf ) );
		f = H::isSame( wo, wi ) ? float3{ 1, 1, 1 } : float3{ 0, 0, 0 };
		return wi;
	}

private:
	const poly::object<SphericalDistribution> &distribution;
};

struct MicrofacetTransmission : BxDF
{
	KOISHI_HOST_DEVICE MicrofacetTransmission( const poly::object<SphericalDistribution> &distribution ) :
	  distribution( distribution )
	{
	}

	KOISHI_HOST_DEVICE float3 f( const solid &wo, const solid &wi ) const override
	{
		// return H::isSame( wo, wi ) ? R : float3{ 0, 0, 0 };
	}

	KOISHI_HOST_DEVICE solid sample( const solid &wo, const float3 &u, float3 &f ) const override
	{
		float pdf;
		auto wh = distribution->sample( u, pdf );
		auto wi = refract( wo, wh, 0.7 );
		// f = H::isSame( wo, wi ) ? float3{ 1, 1, 1 } : float3{ 0, 0, 0 };
		return solid( wi );
	}

private:
	const poly::object<SphericalDistribution> &distribution;
};

struct MicrofacetMaterial : Material
{
	MicrofacetMaterial( const Properties &props ) :
	  distribution( Factory<SphericalDistribution>::create( get<Config>( props, "distribution" ) ) ),
	  color( get( props, "color", float3{ .5f, .5f, .5f } ) )
	{
	}

	KOISHI_HOST_DEVICE virtual void apply( SurfaceInterreact &res, Allocator &pool ) const
	{
		res.bsdf = create<BSDF>( pool );
		res.bsdf->add<MicrofacetReflection>( pool, distribution );
		res.color = color;
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
	float3 color;
};

}  // namespace ext

}  // namespace koishi