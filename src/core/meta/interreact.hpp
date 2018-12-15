#pragma once

#include "primitive.hpp"
#include "bsdf.hpp"

namespace koishi
{
namespace core
{
struct Interreact
{
	float2 uv;

	float3 n, p, u, v;

	BSDF *bsdf = nullptr;

	bool isNull = true;

	float3 color{ 0, 0, 0 };

	float3 emissive{ 0, 0, 0 };

public:
	KOISHI_HOST_DEVICE operator bool() const
	{
		return !isNull;
	}

	KOISHI_HOST_DEVICE Ray emitRay( const float3 &w ) const
	{
		Ray r;
		auto nwo = normalize( w );
		constexpr float eps = 1e-3;
		r.o = p + nwo * eps, r.d = nwo;
		return r;
	}

	KOISHI_HOST_DEVICE Seg emitSeg( const float3 &p ) const
	{
		Seg s;
		s.d = p - this->p;
		auto nwo = normalize( s.d );
		constexpr float eps = 1e-3;
		auto diff = nwo * eps;
		s.o = this->p + diff, s.d -= 2 * diff;
		return s;
	}

	KOISHI_HOST_DEVICE float3 local( const float3 &w ) const
	{
		return float3{ dot( w, u ), dot( w, v ), dot( w, n ) };
	}

	KOISHI_HOST_DEVICE float3 global( const float3 &w ) const
	{
		return w.x * u + w.y * v + w.z * n;
	}

	KOISHI_HOST_DEVICE float3 world( const float3 &w ) const
	{
		return u * w.x + v * w.y + n * w.z;
	}
};

}  // namespace core

}  // namespace koishi