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

