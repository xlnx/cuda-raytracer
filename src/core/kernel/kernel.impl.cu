#include "normal.hpp"
#include "radiance.hpp"
#include "custom.hpp"

namespace koishi
{
namespace core
{
static volatile int kernel_map[] = {
	Factory<Kernel>::reg<NormalKernel>( "Normal" ),
	Factory<Kernel>::reg<CustomKernel>( "Custom" ),
	Factory<Kernel>::reg<RadianceKernel>( "Radiance" )
};
}

}  // namespace koishi