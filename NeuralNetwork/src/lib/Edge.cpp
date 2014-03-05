/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "edge.h"
#include "node.hpp"
#include "external/SigUtil/lib/tool.hpp"

namespace signn{

static auto random_ = sig::SimpleRandom<double>(-1.0, 1.0, DEBUG_MODE);

template <class NodeData>
DirectedEdge<NodeData>::DirectedEdge(typename sig::Just<double>::type weight = sig::Nothing(SIG_DEFAULT_EDGE_WEIGHT))
	: tail_(nullptr), head_(nullptr), weight_(weight ? *weight : random_()), pre_weight_(weight_), delta_(0){}

template <class NodeData>
double DirectedEdge<NodeData>::CalcWeightedScore() const
{
	auto tp = tail_.lock();
	if (tp) return tp->Score() * weight_;
	else assert(false, "class DirectedEdge: node link error");
}

}