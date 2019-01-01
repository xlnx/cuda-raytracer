#include "cpuMulticore.hpp"
#include "cudaSingleGPU.hpp"

namespace koishi
{
namespace core
{
static volatile int tracer_map[] = {
	Factory<Tracer>::reg<CPUMulticoreTracer>( "CPUMulticore" )
	// Factory<Tracer>::reg<CudaSingleGPUTracer>( "CudaSingleGPU" )
};
}

}  // namespace koishi