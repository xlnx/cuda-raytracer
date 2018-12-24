#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct MixedMaterial : Material
{
	MixedMaterial( const Properties &props ) :
	  fac( get( props, "fac", .5f ) ),
	  materials{ Factory<Material>::create(
				   get<std::vector<Config>>( props, "materials" )[ 0 ] ),
				 Factory<Material>::create(
				   get<std::vector<Config>>( props, "materials" )[ 1 ] ) }
	{
	}

	KOISHI_HOST_DEVICE void apply( SurfaceInterreact &res, Allocator &pool ) const override
	{
		Sampler sampler;
		if ( sampler.sample() < fac )
		{
			materials[ 0 ]->apply( res, pool );
		}
		else
		{
			materials[ 1 ]->apply( res, pool );
		}
	}

	void print( std::ostream &os ) const override
	{
		nlohmann::json json = {
			{ "Mixed", { { "materials", {} } } }
		};
		os << json.dump();
	}

private:
	float fac;
	poly::object<Material> materials[ 2 ];
};

}  // namespace ext

}  // namespace koishi