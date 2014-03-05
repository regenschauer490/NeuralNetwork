/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_LAYER_H
#define SIG_NN_LAYER_H

#include "node.hpp"

namespace signn{

template <class NodeData, class Edge>
class Layer
{
public:
	using LayerPtr_ = LayerPtr<NodeData, Edge>;
	using Node_ = Node<NodeData, Edge>;
	using NodePtr_ = NodePtr<NodeData, Edge>;
	using C_NodePtr_ = C_NodePtr<NodeData, Edge>;

private:
	const uint node_num_;
	std::vector<NodePtr_> nodes_;

	SIG_FRIEND_WITH_LAYER;

private:
	virtual LayerPtr_ CloneImpl() const{ return std::shared_ptr<Layer>(new Layer(this->node_num_)); }

protected:
	explicit Layer(uint node_num) : node_num_(node_num){ for (uint i = 0; i < node_num_; ++i) nodes_.push_back(std::make_shared<Node_>()); }

	//forward propagation
	virtual void UpdateNodeScore(){ for (auto& n : nodes_) n->UpdateScore<Sigmoid>(); }
	
	//back propagation for online
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

	//back propagation for batch (not renew weight)
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

	//renew weight (for batch)
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

	NodePtr_ operator[](uint index){ return nodes_[index]; }
	C_NodePtr_ operator[](uint index) const{ return nodes_[index]; }

public:
	virtual ~Layer(){};

	static LayerPtr_ MakeInstance(uint node_num){ return std::shared_ptr<Layer>(new Layer(node_num)); }

	LayerPtr_ CloneInitInstance() const{ return CloneImpl(); }

	uint NodeNum() const{ return node_num_; }
};

}
#endif
