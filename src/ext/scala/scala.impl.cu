#include <ext/util.hpp>
#include "fresnel.hpp"
#include "constant.hpp"

namespace koishi
{
namespace ext
{
static volatile int scala_map[] = {
	Factory<Scala<float>>::reg<Fresnel>( "Fresnel" ),

	Factory<Scala<float>>::reg<Constant<float>>( "Constant" ),
	Factory<Scala<float1>>::reg<Constant<float1>>( "Constant" ),
	Factory<Scala<float2>>::reg<Constant<float2>>( "Constant" ),
	Factory<Scala<float3>>::reg<Constant<float3>>( "Constant" ),
	Factory<Scala<float4>>::reg<Constant<float4>>( "Constant" )
};

}

}  // namespace koishi