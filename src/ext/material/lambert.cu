#include "lambert.hpp"

namespace koishi
{
namespace ext
{
static volatile int x = MaterialFactory::reg<LambertMaterial>( "Lambert" );
}

}  // namespace koishi