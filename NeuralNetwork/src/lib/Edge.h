/*
The MIT License(MIT)

Copyright(c) 2013 Akihiro Nishimura

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

#include "info.hpp"

namespace signn{
	
class DirectedEdge
{
	NodeWPtr tail_;
	NodeWPtr head_;

	double weight_;

	//cache
	double delta_;
	double pre_weight_;
	
public:
	DirectedEdge(NodePtr tail, NodePtr head);
	~DirectedEdge(){};


	double CalcWeightedScore() const;

	template <class ActivationFunc>
	double CalcDeltaWeight(double alpha, double error);

	template <class ActivationFunc>
	void UpdateWeight(double alpha, double error);


	void Weight(double v){ weight_ = v; }
	double Weight() const{ return weight_; }

	double PreWeight() const{ return pre_weight_; }

	void Delta(double v){ delta_ = v; }
	double Delta() const{ return delta_; }

	NodePtr HeadNode(){ return head_.lock(); }
	C_NodePtr HeadNode() const{ return head_.lock(); }

	NodePtr TailNode(){ return tail_.lock(); }
	C_NodePtr TailNode() const{ return tail_.lock(); }
};


template <class ActivationFunc>
double DirectedEdge::CalcDeltaWeight(double alpha, double error)
{
	pre_weight_ = weight_;
	delta_ = error * ActivationFunc::df(head_.lock()->RawScore());
	return alpha * tail_.lock()->Score() * delta_;
}


template <class ActivationFunc>
void DirectedEdge::UpdateWeight(double alpha, double error)
{
	pre_weight_ = weight_;
	delta_ = error * ActivationFunc::df(head_.lock()->RawScore());
	weight_ += alpha * tail_.lock()->Score() * delta_;
}

}

