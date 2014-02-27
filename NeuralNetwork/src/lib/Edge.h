/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_EDGE_H
#define SIG_NN_EDGE_H

#include "info.hpp"

namespace signn{
	
class DirectedEdge
{
	NodeWPtr tail_;
	NodeWPtr head_;

	//parameter
	double weight_;

	//cache
	double delta_;
	double pre_weight_;
	
public:
	DirectedEdge(NodePtr tail, NodePtr head, typename sig::Just<double>::type weight = sig::Nothing(SIG_DEFAULT_EDGE_WEIGHT));
	~DirectedEdge(){};


	double CalcWeightedScore() const;

	template <class ActivationFunc>
	double CalcDeltaWeight(double alpha, double error);

	template <class ActivationFunc>
	void UpdateWeight(double alpha, double beta, double error);


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
	delta_ = error * ActivationFunc::df(head_.lock()->PreActivateScore());
	return alpha * tail_.lock()->Score() * delta_;
}


template <class ActivationFunc>
void DirectedEdge::UpdateWeight(double alpha, double beta, double error)
{
	pre_weight_ = weight_;
	delta_ = error * ActivationFunc::df(head_.lock()->PreActivateScore());
	weight_ = weight_ * beta + alpha * tail_.lock()->Score() * delta_;
}


class UndirectedEdge
{
public:
	using DEdge = DirectedEdge;

private:
	DEdge edge_;

public:
	UndirectedEdge(NodePtr tail, NodePtr head, boost::optional<double> weight = boost::none)
		: edge_(tail, head), weight_(is_random ? weight : random_()), pre_weight_(weight_), delta_(0){}
	~UndirectedEdge(){};

	double CalcWeightedScore() const;

	template <class ActivationFunc>
	double CalcDeltaWeight(double alpha, double error);

	template <class ActivationFunc>
	void UpdateWeight(double alpha, double beta, double error);


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

}
#endif

