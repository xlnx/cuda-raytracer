#include "lambert.hpp"
#include "specular.hpp"
#include "microfacet.hpp"

namespace koishi
{
namespace ext
{
static volatile int material_impl[] = {
	Factory<Material>::reg<LambertMaterial>( "Lambert" ),
	Factory<Material>::reg<SpecularMaterial>( "Specular" ),
	Factory<Material>::reg<MicrofacetMaterial>( "Microfacet" )
};

}  // namespace ext

}  // namespace koishi