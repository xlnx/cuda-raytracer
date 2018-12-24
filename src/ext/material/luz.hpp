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
	  emissive( get( props, "color", float3{ 1, 1, 1 } ) * get( props, "strength", 2.f ) )
	{
	}

	KOISHI_HOST_DEVICE void apply( SurfaceInterreact &res, Allocator &pool ) const override
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

public:
	float3 emissive;
};

}  // namespace ext

}  // namespace koishi