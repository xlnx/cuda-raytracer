#pragma once

#include <core/meta/scala.hpp>
#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct MixedMaterial : Material
{
	MixedMaterial( const Properties &props ) :
	  fac( Factory<Scala<float>>::create( get<Config>( props, "fac" ) ) ),
	  materials{ Factory<Material>::create(
				   get<std::vector<Config>>( props, "materials" )[ 0 ] ),
				 Factory<Material>::create(
				   get<std::vector<Config>>( props, "materials" )[ 1 ] ) }
	{
	}

	KOISHI_HOST_DEVICE void apply( Input &input, Allocator &pool ) const override
	{
		materials[ input.sampler->sample() <
					   fac->compute( input, pool )
					 ? 1
					 : 0 ]
		  ->apply( input, pool );
	}

	void print( std::ostream &os ) const override
	{
		nlohmann::json json = {
			{ "Mixed", { { "materials", {} } } }
		};
		os << json.dump();
	}

private:
	poly::object<Scala<float>> fac;
	poly::object<Material> materials[ 2 ];
};

}  // namespace ext

}  // namespace koishi