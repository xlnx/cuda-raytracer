#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct Mixed : Shader
{
	Mixed( const Properties &props ) :
	  Shader( props ),
	  fac( Factory<Scala<float>>::create( get<Config>( props, "fac" ) ) ),
	  shaders{ Factory<Shader>::create(
				 get<std::vector<Config>>( props, "shaders" )[ 0 ] ),
			   Factory<Shader>::create(
				 get<std::vector<Config>>( props, "shaders" )[ 1 ] ) }
	{
	}

	KOISHI_HOST_DEVICE void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, ShaderTarget target ) const override
	{
		auto f = fac->compute( varyings, pool );
		shaders[ sampler.sample() < f ? 1 : 0 ]->execute(
		  varyings, sampler, pool, target );
	}

	void writeNode( json &j ) const override
	{
		fac->writeNode( j[ "Mixed" ][ "fac" ] );
		shaders[ 0 ]->writeNode( j[ "Mixed" ][ "shaders" ][ 0 ] );
		shaders[ 1 ]->writeNode( j[ "Mixed" ][ "shaders" ][ 1 ] );
	}

private:
	poly::object<Scala<float>> fac;
	poly::object<Shader> shaders[ 2 ];
};

}  // namespace ext

}  // namespace koishi