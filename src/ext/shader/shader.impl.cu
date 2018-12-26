#include <ext/util.hpp>
#include "lambert.hpp"
#include "glossy.hpp"
#include "refraction.hpp"
#include "emission.hpp"
#include "mixed.hpp"

namespace koishi
{
namespace ext
{
static volatile int bxdf_impl[] = {
	Factory<Shader>::reg<Emission>( "Emission" ),
	Factory<Shader>::reg<Lambert>( "Lambert" ),
	Factory<Shader>::reg<Glossy>( "GlossyBSDF" ),
	Factory<Shader>::reg<Refraction>( "RefractionBSDF" ),
	Factory<Shader>::reg<Mixed>( "Mixed" )
};

}  // namespace ext

}  // namespace koishi