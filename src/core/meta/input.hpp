#pragma once

#include <core/basic/ray.hpp>
#include <core/misc/sampler.hpp>
#include "bsdf.hpp"

namespace koishi
{
namespace core
{
struct LocalInput
{
	normalized_float3 u, v, n;
	float3 p;

	normalized_float3 wo;

	uint matid;

	Sampler *sampler;

public:
	KOISHI_HOST_DEVICE Ray emitRay( const solid &w ) const
	{
		Ray r;
		constexpr float eps = 4e-3;
		r.o = p + w * eps, r.d = w;
		return r;
	}

	KOISHI_HOST_DEVICE Seg emitSeg( const float3 &p ) const
	{
		Seg s;
		auto d = p - this->p;
		s.d = normalize( d );
		auto nwo = normalize( s.d );
		constexpr float eps = 4e-3;
		auto diff = nwo * eps;
		s.o = this->p + diff, s.t = length( d - 2 * diff );
		return s;
	}

	KOISHI_HOST_DEVICE solid local( const normalized_float3 &w ) const
	{
		return normalize( float3{ dot( static_cast<const float3 &>( w ), u ),
								  dot( static_cast<const float3 &>( w ), v ),
								  dot( static_cast<const float3 &>( w ), n ) } );
	}

	KOISHI_HOST_DEVICE normalized_float3 global( const solid &w ) const
	{
		return normalize( w.x * u + w.y * v + w.z * n );
	}
};

struct Input : LocalInput
{
	BxDF *bxdf;

	float3 color{ 0, 0, 0 };

	float3 emissive{ 0, 0, 0 };
};

}  // namespace core

}  // namespace koishi