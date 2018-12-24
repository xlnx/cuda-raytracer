#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct Specular : BxDF
{
	KOISHI_HOST_DEVICE float3 f( const solid &wo, const solid &wi ) const override
	{
		if ( reflect( wo, solid( float3{ 0, 0, 1 } ) ) == wi )
			return float3{ 1.f, 1.f, 1.f };
		else
			return float3{ 0, 0, 0 };
	}
	KOISHI_HOST_DEVICE solid sample( const solid &wo, const float3 &rn, float3 &f ) const override
	{
		f = float3{ 1, 1, 1 };
		return reflect( wo, solid( float3{ 0, 0, 1 } ) );
	}
};

struct SpecularMaterial : Material
{
	SpecularMaterial( const Properties &props ) :
	  Material( props )
	{
	}
	KOISHI_HOST_DEVICE void apply( Input &input, Allocator &pool ) const
	{
		Material::apply( input, pool );
		input.bxdf = create<Specular>( pool );
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