/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "edge.h"
#include "node.hpp"

namespace signn{

static auto random_ = SimpleRandom<double>(-1.0, 1.0, DEBUG_MODE);

DirectedEdge::DirectedEdge(NodePtr tail, NodePtr head) : tail_(tail), head_(head), weight_(random_()), pre_weight_(weight_), delta_(0){}

double DirectedEdge::CalcWeightedScore() const
{
	auto tp = tail_.lock();
	if (tp) return tp->Score() * weight_;
	else assert(false, "class DirectedEdge: node link error");
}

}