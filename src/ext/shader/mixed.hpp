#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct Mixed : Shader
{
	Mixed( const Properties &props ) :
	  fac( Factory<Scala<float>>::create( get<Config>( props, "fac" ) ) ),
	  shaders{ Factory<Shader>::create(
				 get<std::vector<Config>>( props, "shaders" )[ 0 ] ),
			   Factory<Shader>::create(
				 get<std::vector<Config>>( props, "shaders" )[ 1 ] ) }
	{
	}

	KOISHI_HOST_DEVICE void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, uint target ) const override
	{
		shaders[ sampler.sample() < fac->compute( varyings, pool )
				   ? 1
				   : 0 ]
		  ->execute( varyings, sampler, pool, target );
	}

	void print( std::ostream &os ) const override
	{
		nlohmann::json json = {
			{ "Mixed", { { "shaders", {} } } }
		};
		os << json.dump();
	}

private:
	poly::object<Scala<float>> fac;
	poly::object<Shader> shaders[ 2 ];
};

}  // namespace ext

}  // namespace koishi