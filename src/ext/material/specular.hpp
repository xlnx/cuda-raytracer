#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct Specular : BxDF
{
	KOISHI_HOST_DEVICE float3 f( const float3 &wo, const float3 &wi ) const override
	{
		return float3{ 0.f };
	}
	KOISHI_HOST_DEVICE float3 sample( const float3 &wo, float3 &wi, const float2 &rn, float &pdf ) const override
	{
		wi = -reflect( wo, float3{ 0, 0, -1 } );
		return f( wo, wi );
	}
};

struct SpecularMaterial : Material
{
	SpecularMaterial( const MaterialProps &props )
	{
	}
	KOISHI_HOST_DEVICE void apply( Interreact &res, Allocator &pool ) const
	{
		res.bsdf = create<BSDF>( pool );
		res.bsdf->add<Specular>( pool );
	}
	void print( std::ostream &os ) const
	{
		nlohmann::json json = nlohmann::json::object();
		json[ "SpecularMaterial" ] = nlohmann::json::object();
		os << json.dump();
	}
};

}  // namespace ext

}  // namespace koishi