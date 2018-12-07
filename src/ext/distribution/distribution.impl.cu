#include "beckmann.hpp"
#include "cosine.hpp"

namespace koishi
{
namespace ext
{
static volatile int distribution_impl[] = {
	Factory<SphericalDistribution>::reg<CosDistribution>( "Cosine" ),
	Factory<SphericalDistribution>::reg<BeckmannDistribution>( "Beckmann" )
};

}  // namespace ext

}  // namespace koishi