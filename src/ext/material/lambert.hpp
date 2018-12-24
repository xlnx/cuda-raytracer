#pragma once

#include <ext/util.hpp>
#include <ext/distribution/cosine.hpp>

namespace koishi
{
namespace ext
{
struct LambertDiffuse : BxDF
{
	KOISHI_HOST_DEVICE LambertDiffuse( const float3 &R ) :
	  R( R )
	{
	}

	KOISHI_HOST_DEVICE float3 f( const solid &wo, const solid &wi ) const override
	{
		return R * distrib.f( wi );
	}

	KOISHI_HOST_DEVICE solid sample( const solid &wo, const float3 &u, float3 &f ) const override
	{
		float pdf;
		auto wi = distrib.sample( u, pdf );
		f = this->f( wo, wi ) / pdf;
		return wi;
	}

private:
	float3 R;
	CosDistribution distrib;
};

struct LambertMaterial : Material
{
	LambertMaterial( const Properties &props ) :
	  Material( props ),
	  R( get( props, "R", float3{ 1, 1, 1 } ) )
	{
	}

	KOISHI_HOST_DEVICE void apply( SurfaceInterreact &res, Allocator &pool ) const override
	{
		Material::apply( res, pool );
		res.bsdf = create<BSDF>( pool );
		res.bsdf->add<LambertDiffuse>( pool, R );
	}

	void print( std::ostream &os ) const override
	{
		nlohmann::json json = {
			{ "LambertMaterial", { { "R", R } } }
		};
		os << json.dump();
	}

private:
	float3 R;
};

}  // namespace ext

}  // namespace koishi