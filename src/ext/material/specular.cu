#include "specular.hpp"

namespace koishi
{
namespace ext
{
static volatile int dummy = MaterialFactory::reg<SpecularMaterial>( "Specular" );
}

}  // namespace koishi
