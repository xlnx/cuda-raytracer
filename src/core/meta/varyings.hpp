#pragma once

#include <core/basic/basic.hpp>

namespace koishi
{
namespace core
{
struct LocalVaryings
{
	normalized_float3 u, v, n;
	float3 p;

	solid wo;

	uint shaderId;

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

struct Varyings : LocalVaryings
{
	solid wi = solid( float3{ 0, 0, 0 } );
	float3 f{ 0, 0, 0 };

	float3 emission{ 0, 0, 0 };
};

}  // namespace core

}  // namespace koishi