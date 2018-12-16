#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct Specular : BxDF
{
	KOISHI_HOST_DEVICE float3 f( const normalized_float3 &wo, const normalized_float3 &wi ) const override
	{
		if ( reflect( wo, normalized_float3( float3{ 0, 0, 1 } ) ) == wi )
			return float3{ 1.f, 1.f, 1.f };
		else
			return float3{ 0, 0, 0 };
	}
	KOISHI_HOST_DEVICE normalized_float3 sample( const normalized_float3 &wo, const float3 &rn, float3 &f ) const override
	{
		f = float3{ 1, 1, 1 };
		return reflect( wo, normalized_float3( float3{ 0, 0, 1 } ) );
	}
};

struct SpecularMaterial : Material
{
	SpecularMaterial( const Properties &props )
	{
	}
	KOISHI_HOST_DEVICE void apply( Interreact &res, Allocator &pool ) const
	{
		res.bsdf = create<BSDF>( pool );
		res.bsdf->add<Specular>( pool );
	}
	void print( std::ostream &os ) const
	{
		nlohmann::json json = {
			{ "SpecularMaterial", {} }
		};
		os << json.dump();
	}
};

}  // namespace ext

}  // namespace koishi