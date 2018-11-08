#pragma once

#include <vec/vmath.hpp>
#include <core/allocator.hpp>
#include "mesh.hpp"

namespace koishi
{
namespace core
{
namespace dev
{
struct Scene
{
	KOISHI_HOST_DEVICE dev::Hit intersect( const core::Ray &r, Allocator &pool ) const
	{
		dev::Mesh *pm;
		core::Hit hit;
		for ( uint i = 0; i != N; ++i )
		{
			core::Hit hit1;
			if ( mesh[ i ].intersect( r, 1, hit1 ) && hit1.t < hit.t )
			{
				hit = hit1;
				pm = mesh + i;
			}
		}
		dev::Hit res;
		if ( hit )
		{
			res.mesh = pm;
			res.isNull = false;
			res.n = core::interplot( pm->normals[ pm->indices[ hit.id ] ],
									 pm->normals[ pm->indices[ hit.id + 1 ] ],
									 pm->normals[ pm->indices[ hit.id + 2 ] ],
									 hit.uv );
			res.p = r.o + r.d * hit.t;
			res.uv = hit.uv;
			pm->material->fetchTo( res, pool );
		}
		return res;
	}

public:
	template <typename Target>
	KOISHI_HOST_DEVICE static Scene *create( const std::vector<core::Mesh> &mesh );
	template <typename Target>
	KOISHI_HOST_DEVICE static void destroy( Scene *scene );

private:
	dev::Mesh *mesh;
	uint N;
};

template <>
KOISHI_HOST_DEVICE Scene *Scene::create<Host>( const std::vector<core::Mesh> &mesh )
{
	Scene *scene = new Scene;
	scene->mesh = (dev::Mesh *)malloc( sizeof( dev::Mesh ) * mesh.size() );
	for ( uint i = 0; i != mesh.size(); ++i )
	{
		new ( &scene->mesh[ i ] ) dev::Mesh( mesh[ i ] );
	}
	scene->N = mesh.size();
	return scene;
}

template <>
KOISHI_HOST_DEVICE void Scene::destroy<Host>( Scene *scene )
{
	free( scene->mesh );
}

#if defined( KOISHI_USE_CUDA )

template <>
KOISHI_HOST_DEVICE static Scene *Scene::create<Device>( const std::vector<core::Mesh> &mesh )
{
}

#endif

}  // namespace dev

}  // namespace core

}  // namespace koishi