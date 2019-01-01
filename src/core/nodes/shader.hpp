#pragma once

#include <core/misc/sampler.hpp>
#include "node.hpp"

namespace koishi
{
namespace core
{
enum ShaderTarget
{
	sample_wi_f_by_wo = 0x00001000u,   // (wo) ->[wi, f]
	compute_f_by_wi_wo = 0x00002000u,  // (wo, wi) -> f
	target_default = 0x00000000u
};

struct Shader : Node
{
	using Node::Node;

	KOISHI_HOST_DEVICE virtual void execute(
	  Varyings &varyings, Sampler &sampler, Allocator &pool, ShaderTarget target = target_default ) const = 0;
};

}  // namespace core

}  // namespace koishi