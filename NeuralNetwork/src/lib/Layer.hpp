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

#include "Node.hpp"
//#include "Edge.h"
#include "info.hpp"

namespace signn{

class Layer
{
	const uint node_num_;
	std::vector<NodePtr> nodes_;

	FRIEND_WITH_LAYER;

private:
	virtual LayerPtr CloneImpl() const{ return std::shared_ptr<Layer>(new Layer(this->node_num_)); }

protected:
	Layer(uint node_num) : node_num_(node_num){ for (uint i = 0; i < node_num_; ++i) nodes_.push_back(std::make_shared<Node>()); }

	virtual void UpdateNodeScore(){ for (auto& n : nodes_) n->UpdateScore<Sigmoid>(); }
	
	void UpdateEdgeWeight(double alpha, double beta){
		for (auto& node : nodes_){
			auto const error = std::accumulate(node->out_begin(), node->out_end(), 0.0, [](double sum, DEdgePtr& e){
				return sum + e->PreWeight() * e->Delta();
			});

			for (auto edge = node->in_begin(), end = node->in_end(); edge != end; ++edge){
				(*edge)->UpdateWeight<Sigmoid>(alpha, beta, error);
			}
		}
	}

	std::vector<double> CalcEdgeWeight(double alpha){
		std::vector<double> new_weight;

		for (auto& node : nodes_){
			auto const error = std::accumulate(node->out_begin(), node->out_end(), 0.0, [](double sum, DEdgePtr& e){
				return sum + e->PreWeight() * e->Delta();
			});

			for (auto edge = node->in_begin(), end = node->in_end(); edge != end; ++edge){
				new_weight.push_back( (*edge)->CalcDeltaWeight<Sigmoid>(alpha, error) );
			}
		}
		return std::move(new_weight);
	}

	void RenewEdgeWeight(std::vector<double> const& delta, double weight_decay_rate)
	{
		int ct = -1;
		for (auto& node : nodes_){
			for (auto edge = node->in_begin(), end = node->in_end(); edge != end; ++edge){
				(*edge)->Weight((*edge)->Weight() * weight_decay_rate + delta[++ct]);
			}
		}
	}

	auto begin() ->decltype(nodes_.begin()){ return nodes_.begin(); }
	auto begin() const ->decltype(nodes_.cbegin()){ return nodes_.cbegin(); }

	auto end() ->decltype(nodes_.end()){ return nodes_.end(); }
	auto end() const ->decltype(nodes_.cend()){ return nodes_.cend(); }

	NodePtr operator[](uint index){ return nodes_[index]; }
	C_NodePtr operator[](uint index) const{ return nodes_[index]; }

public:
	virtual ~Layer(){};

	static LayerPtr MakeInstance(uint node_num){ return std::shared_ptr<Layer>(new Layer(node_num)); }

	LayerPtr CloneInitInstance() const{ return CloneImpl(); }

	uint NodeNum() const{ return node_num_; }
};

}
