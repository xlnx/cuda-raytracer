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
	  shader_a( Factory<Shader>::create(
				 get<std::vector<Config>>( props, "shaders" )[ 0 ] ) ),
	  shader_b( Factory<Shader>::create(
				 get<std::vector<Config>>( props, "shaders" )[ 1 ] ) )
	{
	}

	KOISHI_HOST_DEVICE void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, ShaderTarget target ) const override
	{
		auto f = fac->compute( varyings, pool );
		( sampler.sample() < f ? shader_b : shader_a )->execute(
		  varyings, sampler, pool, target );
	}

	void writeNode( json &j ) const override
	{
		fac->writeNode( j[ "Mixed" ][ "fac" ] );
		shader_a->writeNode( j[ "Mixed" ][ "shaders" ][ 0 ] );
		shader_b->writeNode( j[ "Mixed" ][ "shaders" ][ 1 ] );
	}

private:
	poly::object<Scala<float>> fac;
	poly::object<Shader> shader_a, shader_b;
};

}  // namespace ext

}  // namespace koishi
