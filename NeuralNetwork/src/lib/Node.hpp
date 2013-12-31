/*
The MIT License(MIT)

Copyright(c) 2014 Akihiro Nishimura

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files(the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include "Edge.h"
#include "info.hpp"

namespace signn{
	
class Node
{
	const double threshold_;

	std::vector<DEdgePtr> in_;
	std::vector<DEdgePtr> out_;

	double score_;
	
	//cache
	double pre_activate_score_;

public:
	Node() : threshold_(threshold_theta), score_(0), pre_activate_score_(0){};
	~Node(){};

	void AddInEdge(DEdgePtr edge){ in_.push_back(edge); }
	void AddOutEdge(DEdgePtr edge){ out_.push_back(edge); }
	
	double AccumulateRawScore() const{
		return std::accumulate(in_.begin(), in_.end(), 0.0, [](double sum, decltype(*in_.begin())& e){ return e->CalcWeightedScore() + sum; }) - threshold_;
	}

	template <class ActivationFunc>
	void UpdateScore();

	void UpdateScore(double score){ score_ = score; }
	
	double Score() const{ return score_; }
	void Score(double v){ score_ = v; }

	double PreActivateScore() const{ return pre_activate_score_; }

	auto in_begin() ->decltype(in_.begin()){ return in_.begin(); }
	auto in_begin() const ->decltype(in_.cbegin()){ return in_.cbegin(); }

	auto in_end() ->decltype(in_.end()){ return in_.end(); }
	auto in_end() const ->decltype(in_.cend()){ return in_.cend(); }

	auto out_begin() ->decltype(out_.begin()){ return out_.begin(); }
	auto out_begin() const ->decltype(out_.cbegin()){ return out_.cbegin(); }

	auto out_end() ->decltype(out_.end()){ return out_.end(); }
	auto out_end() const ->decltype(out_.cend()){ return out_.cend(); }
};


template<class ActivationFunc>
void Node::UpdateScore()
{
	score_ = ActivationFunc::f(AccumulateRawScore());
}

}
