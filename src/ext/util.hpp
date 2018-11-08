#pragma once

#include <vec/vmath.hpp>
#include <core/allocator.hpp>
#include <core/dev/material.hpp>
#include <core/dev/bsdf.hpp>

namespace koishi
{
namespace ext
{
using Material = core::dev::Material;

using BSDF = core::dev::BSDF;

using BxDF = core::dev::BxDF;

}  // namespace ext

}  // namespace koishi