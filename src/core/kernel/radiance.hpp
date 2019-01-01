#pragma once

#include <core/basic/basic.hpp>
#include "kernel.hpp"

namespace koishi
{
namespace core
{
struct RadianceKernel : Kernel
{
	struct Slice : ProfileSlice
	{
		struct BounceRecord
		{
			Ray ray;
			float3 p, f, L;
		};

		Slice( const Properties &props ) :
		  ProfileSlice( props ),
		  bounces( get( props, "depth", 8u ) )
		{
		}

		void writeSlice( std::ostream &os ) const override
		{
			if ( L.x >= 1 && L.y >= 1 && L.z >= 1 )
			{
				os << "bounce: " << bounce << " / " << bounces.size() << std::endl;
				for ( int i = 0; i != std::min( bounce, bounces.size() ); ++i )
				{
					os << "ray: " << bounces[ i ].ray.o << bounces[ i ].ray.d << std::endl;
					os << "hit: " << bounces[ i ].p << std::endl;
					os << "f: " << bounces[ i ].f << std::endl;
					os << "L: " << bounces[ i ].L << std::endl;
				}
			}
		}
		void readSlice( std::istream &is ) const override
		{
		}

		std::size_t bounce;
		float3 L;
		poly::vector<BounceRecord> bounces;
	};

	using Kernel::Kernel;

	KOISHI_HOST_DEVICE float3 execute( Ray ray, const Scene &scene, Allocator &pool,
									   Sampler &rng, ProfileSlice *prof ) override;

	virtual poly::object<ProfileSlice> profileSlice( const Properties &props ) const override
	{
		return poly::make_object<Slice>( props );
	}
};

}  // namespace core

}  // namespace koishi
