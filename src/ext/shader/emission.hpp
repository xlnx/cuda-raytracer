#pragma once

#include <ext/util.hpp>

namespace koishi
{
namespace ext
{
struct Emission : Shader
{
	Emission( const Properties &props ) :
	  Shader( props ),
	  color( get( props, "color", float3{ 1, 1, 1 } ) ),
	  strength( get( props, "strength", 2.f ) ),
	  emission( color * strength )
	{
	}

	KOISHI_HOST_DEVICE void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, uint target ) const override
	{
		varyings.emission = emission;
	}

	void print( std::ostream &os ) const override
	{
	}

private:
	float3 color;
	float strength;

public:
	const float3 emission;
};

}  // namespace ext

}  // namespace koishi
