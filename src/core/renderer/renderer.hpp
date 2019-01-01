#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <util/image.hpp>
#include <util/config.hpp>
#include <core/basic/basic.hpp>
#include <core/tracer/tracer.hpp>

namespace koishi
{
namespace core
{
struct Renderer
{
	struct Configuration : serializable<Configuration>
	{
		Property( Properties, scene );
		Property( Properties, profiler, [] { 
			Properties x; x["enable"] = false; return x; }() );
		Property( Config, tracer, Config( "CPUMulticore", {} ) );
	};

	Renderer( uint w, uint h );

	void render( const std::string &path,
				 const std::string &dest, uint spp );

private:
	uint w, h;
};

}  // namespace core

}  // namespace koishi
