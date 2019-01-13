#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct Blackhole : Shader
{
	using Shader::Shader;

	KOISHI_HOST_DEVICE void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, ShaderTarget target ) const override
	{
		switch ( target )
		{
		case sample_wi_f_by_wo:
			varyings.wi = solid( float3{ 0, 0, 1 } );
		case compute_f_by_wi_wo:
			varyings.f = float3{ 0, 0, 0 };
		}
	}

	void writeNode( json &j ) const override
	{
		j = "Blackhole";
	}
};

}  // namespace ext

}  // namespace koishi