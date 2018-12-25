#pragma once

#include <vec/vec.hpp>

namespace koishi
{
namespace core
{
// clang-format off
constexpr uint sample_wi_f_by_wo =     0x00001000u;  // (wo) ->[wi, f]
constexpr uint compute_f_by_wi_wo =    0x00002000u;  // (wo, wi) -> f
// clang-format on

}  // namespace core

}  // namespace koishi