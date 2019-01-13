#pragma once

#include <core/basic/basic.hpp>
#include "kernel.hpp"

namespace koishi
{
namespace core
{
struct RadianceKernel : Kernel
{
	struct Configuration : serializable<Configuration>
	{
		Property( uint, maxBounce, 16u );
	};

	struct Slice : ProfileSlice
	{
		struct BounceRecord
		{
			Ray ray;
			float3 f, L;
			float theta;
			uint shaderId;
		};

		Slice( const Properties &props ) :
		  ProfileSlice( props ),
		  bounces( get( props, "depth", 8u ) )
		{
		}

		void writeSlice( std::ostream &os ) const override
		{
			if ( L.x >= 1e2 && L.y >= 1e2 && L.z >= 1e2 )
			{
				os << "bounce: " << bounce << " / " << bounces.size() << std::endl;
				for ( int i = 0; i != std::min( bounce, bounces.size() ); ++i )
				{
					os << "ray: " << bounces[ i ].ray.o << bounces[ i ].ray.d << std::endl;
					os << "f: " << bounces[ i ].f << std::endl;
					os << "theta: " << bounces[ i ].theta << std::endl;
					os << "L: " << bounces[ i ].L << std::endl;
					os << "shader: " << bounces[ i ].shaderId << std::endl;
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

	RadianceKernel( const Properties &props );

	KOISHI_HOST_DEVICE float3 execute( Ray ray, const Scene &scene, Allocator &pool,
									   Sampler &rng, ProfileSlice *prof ) override;

	virtual poly::object<ProfileSlice> profileSlice( const Properties &props ) const override
	{
		return poly::make_object<Slice>( props );
	}

private:
	uint maxBounce;
};

}  // namespace core

}  // namespace koishi
