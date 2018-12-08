#include "lambert.hpp"
#include "specular.hpp"
#include "microfacet.hpp"
#include "luz.hpp"

namespace koishi
{
namespace ext
{
static volatile int material_impl[] = {
	Factory<Material>::reg<LambertMaterial>( "Lambert" ),
	Factory<Material>::reg<SpecularMaterial>( "Specular" ),
	Factory<Material>::reg<MicrofacetMaterial>( "Microfacet" ),
	Factory<Material>::reg<LuzMaterial>( "Luz" )
};

}  // namespace ext

}  // namespace koishi