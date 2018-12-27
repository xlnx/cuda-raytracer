#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct Transparent : Shader
{
	Transparent( const Properties &props ) :
	  Shader( props ),
	  color( get( props, "color", float3{ 1, 1, 1 } ) )
	{
	}

	KOISHI_HOST_DEVICE void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, ShaderTarget target ) const override
	{
		switch ( target )
		{
		case sample_wi_f_by_wo:
			varyings.wi = -varyings.wo;
			varyings.f = color;
			break;
		case compute_f_by_wi_wo:
			varyings.f = float3{ 0, 0, 0 };
			break;
		}
	}

	void writeNode( json &j ) const override
	{
		j[ "Transparent" ][ "color" ] = color;
	}

private:
	float3 color;
};

}  // namespace ext

}  // namespace koishi