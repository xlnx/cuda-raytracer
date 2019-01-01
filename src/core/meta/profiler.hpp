#pragma once

#include <core/basic/basic.hpp>

namespace koishi
{
namespace core
{
struct HitProfile
{
	float3 rayOrigin, rayHit;
	normalized_float3 rayDir;
};

// struct ProfilePerRay{
// 	ProfilePerRay()
// };

struct Profiler;

struct ProfileSlice
{
	friend struct Profiler;

private:
	ProfileSlice() = default;

	ProfileSlice( poly::vector<HitProfile> &hits, uint begin, uint depth ) :
	  enable( true ),
	  hits( &hits ),
	  begin( begin ),
	  depth( depth )
	{
	}

public:
	KOISHI_HOST_DEVICE bool enabled( uint index ) const
	{
		return enable && index >= 0 && index < depth;
	}

	KOISHI_HOST_DEVICE HitProfile &operator[]( uint index ) const
	{
		KASSERT( index >= 0 && index < depth );
		return hits->operator[]( begin + index );
	}

private:
	bool enable = false;
	poly::vector<HitProfile> *hits;
	uint begin, depth;
};

struct Profiler : emittable
{
	struct Configuration : serializable<Configuration>
	{
		Property( bool, enable, true );
		Property( uint, depth, 8 );
		Property( uint4, area, uint4{ 0, 0, -1u, -1u } );
	};

	Profiler( const Properties &props, uint w, uint h, uint spp )
	{
		Configuration config = json( props );
		enable = config.enable;
		area = max( uint4{ 0, 0, 0, 0 }, min( config.area, uint4{ w, h, w, h } ) );
		this->spp = spp;
		depth = config.depth;
		areaSize = uint2{ area.z - area.x, area.w - area.y };
		hits.resize( areaSize.x * areaSize.y * depth * spp );

		if ( enable )
		{
			KINFO( profiler, "Enabled in", area );
		}
		else
		{
			KINFO( profiler, "Disabled" );
		}
	}

	KOISHI_HOST_DEVICE bool enabled() const
	{
		return enable;
	}

	KOISHI_HOST_DEVICE bool enabled( uint x, uint y ) const
	{
		auto pos = uint2{ x, y };
		return enable && pos.x >= area.x && pos.x < area.z &&
			   pos.y >= area.y && pos.y < area.w;
	}

	KOISHI_HOST_DEVICE ProfileSlice at( uint x, uint y, uint k )
	{
		if ( !enable ) return ProfileSlice();
		auto pos = uint2{ x, y } - uint2{ area.x, area.y };
		KASSERT( pos.x >= 0 && pos.x < areaSize.x &&
				 pos.y >= 0 && pos.y < areaSize.y &&
				 k >= 0 && k < spp );
		return ProfileSlice( hits, ( ( areaSize.x * pos.y + pos.x ) * spp + k ) * depth, depth );
	}

private:
	bool enable;
	uint4 area;
	uint spp;
	uint depth;
	uint2 areaSize;
	poly::vector<HitProfile> hits;
};

}  // namespace core

}  // namespace koishi