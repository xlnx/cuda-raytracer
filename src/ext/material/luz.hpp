#pragma once

#include <ext/util.hpp>
#include "specular.hpp"

namespace koishi
{
namespace ext
{
struct LuzMaterial : Material
{
	LuzMaterial( const Properties &props ) :
	  emissive( get( props, "emissive", float3{ 1, 1, 1 } ) )
	{
	}

	KOISHI_HOST_DEVICE void apply( Interreact &res, Allocator &pool ) const override
	{
		res.bsdf = create<BSDF>( pool );
		res.bsdf->add<Specular>( pool );
		res.emissive = emissive;
	}

	void print( std::ostream &os ) const
	{
		nlohmann::json json = {
			{ "LuzMaterial", { "emissive", emissive } }
		};
		os << json.dump();
	}

private:
	float3 emissive;
};

}  // namespace ext

}  // namespace koishi