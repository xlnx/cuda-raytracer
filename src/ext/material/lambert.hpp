#pragma once

#include <ext/util.hpp>

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
		return H::isSame( wo, wi ) ? R : float3{ 0, 0, 0 };
	}

	KOISHI_HOST_DEVICE solid sample( const solid &wo, const float3 &u, float3 &f ) const override
	{
		auto wi = H::sampleCos( float2{ u.x, u.y } );  // sample lambert
		f = H::isSame( wo, wi ) ? R : float3{ 0, 0, 0 };
		return wi;
	}

private:
	float3 R;
};

struct LambertMaterial : Material
{
	LambertMaterial( const Properties &props ) :
	  R( get( props, "R", float3{ 0.5, 0.5, 0.5 } ) )
	{
	}

	KOISHI_HOST_DEVICE void apply( SurfaceInterreact &res, Allocator &pool ) const override
	{
		res.bsdf = create<BSDF>( pool );
		res.bsdf->add<LambertDiffuse>( pool, R );
	}

	void print( std::ostream &os ) const
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