#include "normal.hpp"
#include "radiance.hpp"
#include "visibility.hpp"
#include "halfVector.hpp"

namespace koishi
{
namespace core
{
static volatile int kernel_map[] = {
	Factory<Kernel>::reg<NormalKernel>( "Normal" ),
	Factory<Kernel>::reg<VisibilityKernel>( "Visibility" ),
	Factory<Kernel>::reg<HalfVectorKernel>( "HalfVector" ),
	Factory<Kernel>::reg<RadianceKernel>( "Radiance" )
};
}

}  // namespace koishi