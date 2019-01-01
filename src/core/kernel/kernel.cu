#include "kernel.hpp"

namespace koishi
{
namespace core
{
Profiler::Profiler( const std::string &filename )
{
	// std::ifstream is( filename );
	// std::string kernelType;
	// uint4 area;
	// is >> kernelType >> this->spp;
	// is >> area.x >> area.y >> area.z >> area.w;
	// this->kernel = Factory<Kernel>::create( {} );
	// this->area = area;
	// areaSize = uint2{ area.z - area.x, area.w - area.y };
	// enable = true;
	// output = filename;
	// auto profileSize = areaSize.x * areaSize.y * spp;
	// for ( int i = 0; i != profileSize; ++i )
	// {
	// 	slices.emplace_back( kernel->profileSlice() );
	// }
}

Profiler::Profiler( const Properties &props, const poly::object<Kernel> &kernel,
					uint w, uint h, uint spp )
{
	this->spp = spp;
	this->kernel = &*kernel;
	Configuration config = json( props );
	enable = config.enable;
	output = config.output;
	area = max( uint4{ 0, 0, 0, 0 },
				min( config.area, uint4{ w, h, w, h } ) );
	areaSize = uint2{ area.z - area.x, area.w - area.y };
	if ( enable )
	{
		KINFO( profiler, "Enabled in", area );
		auto profileSize = areaSize.x * areaSize.y * spp;
		for ( int i = 0; i != profileSize; ++i )
		{
			slices.emplace_back( kernel->profileSlice( config.props ) );
		}
		KINFO( profiler, "Allocoated ", profileSize, "profile slices" );
	}
	else
	{
		KINFO( profiler, "Disabled" );
	}
}

Profiler::~Profiler()
{
	KINFO( profiler, "Writting profile" );
	std::ofstream of( output );
	for ( uint y = 0; y != areaSize.y; ++y )
	{
		for ( uint x = 0; x != areaSize.x; ++x )
		{
			for ( uint k = 0; k != spp; ++k )
			{
				of << "at " << uint3{ x + area.x, y + area.y, k } << ": ";
				slices[ ( areaSize.x * y + x ) * spp + k ]->writeSlice( of );
			}
		}
	}
	KINFO( profiler, "Written to '" + output + "'" );
}

}  // namespace core

}  // namespace koishi