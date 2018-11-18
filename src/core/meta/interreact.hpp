#pragma once

#include "mesh.hpp"
#include "bsdf.hpp"

namespace koishi
{
namespace core
{
struct Interreact
{
	double2 uv;

	double3 n, p, u, v;

	const Mesh *mesh;

	BSDF *bsdf = nullptr;

	bool isNull = true;

public:
	KOISHI_HOST_DEVICE operator bool() const
	{
		return !isNull;
	}

	KOISHI_HOST_DEVICE Ray emitRay( const double3 &wo ) const
	{
		Ray r;
		auto nwo = normalize( wo );
		constexpr double eps = 1e-3;
		r.o = p + nwo * eps, r.d = nwo;
		return r;
	}

	KOISHI_HOST_DEVICE double3 local( const double3 &w ) const
	{
		return double3{ dot( w, u ), dot( w, v ), dot( w, n ) };
	}

	KOISHI_HOST_DEVICE double3 world( const double3 &w ) const
	{
		return u * w.x + v * w.y + n * w.z;
	}
};

}  // namespace core

}  // namespace koishi