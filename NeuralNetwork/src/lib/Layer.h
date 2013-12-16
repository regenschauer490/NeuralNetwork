#pragma once

#include "Node.h"
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

	virtual void UpdateNodeScore(){ for (auto& n : nodes_) n->UpdateScore<Sigmoid<double>>(); }
	
	void UpdateEdgeWeight(double alpha){
		for (auto& node : nodes_){
			auto const error = std::accumulate(node->out_begin(), node->out_end(), 0.0, [](double sum, DEdgePtr& e){
				return sum + e->PreWeight() * e->Delta();
			});

			for (auto edge = node->in_begin(), end = node->in_end(); edge != end; ++edge){
				(*edge)->UpdateWeight<Sigmoid<double>>(alpha, error);
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
				new_weight.push_back( (*edge)->CalcDeltaWeight<Sigmoid<double>>(alpha, error) );
			}
		}
		return std::move(new_weight);
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
