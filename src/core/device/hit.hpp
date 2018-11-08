#pragma once

#include <vec/vmath.hpp>
#include <core/allocator.hpp>
#include "bsdf.hpp"
#include "mesh.hpp"

namespace koishi
{
namespace core

{
namespace dev
{
class Scene;
class Mesh;

struct Hit
{
	friend class dev::Scene;

	double2 uv;

	double3 n, p;

	dev::Mesh *mesh;

	dev::BSDF *bsdf = nullptr;

public:
	KOISHI_HOST_DEVICE operator bool() const
	{
		return !isNull;
	}

	KOISHI_HOST_DEVICE core::Ray emitRay( const double3 &wo ) const
	{
		core::Ray r;
		auto nwo = normalize( wo );
		constexpr double eps = 1e-3;
		r.o = p + nwo * eps, r.d = nwo;
		return r;
	}

private:
	bool isNull = true;
};

}  // namespace dev

}  // namespace core

}  // namespace koishi