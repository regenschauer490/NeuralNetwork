/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_EDGE_H
#define SIG_NN_EDGE_H

#include "info.hpp"
#include "util.hpp"

namespace signn{
	
template <class NodeData>
class DirectedEdge
{
public:
	using NodePtr_ = NodePtr<NodeData, DirectedEdge>;
	using C_NodePtr_ = C_NodePtr<NodeData, DirectedEdge>;
	using NodeWPtr_ = NodeWPtr<NodeData, DirectedEdge>;

private:
	//nodeとの間で循環参照となるのでweak_ptrを使用
	NodeWPtr_ tail_;
	NodeWPtr_ head_;

	//parameter
	double weight_;

	//cache
	double delta_;
	double pre_weight_;

	//ノード・エッジ間の連結操作関数をfriend指定 (これらの関数でのみ連結の制御を行う)
	SIG_FRIEND_WITH_NODE_AND_EDGE

private:
	void SetNode(NodePtr_ const& tail, NodePtr_ const& head){ tail_ = tail; head_ = head; }

public:
	explicit DirectedEdge(typename sig::Just<double>::type weight = sig::Nothing(SIG_DEFAULT_EDGE_WEIGHT))
		: weight_(weight ? sig::fromJust(weight) : GetRandNum(-1.0, 1.0)), pre_weight_(weight_), delta_(0){}
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

	NodePtr_ HeadNode(){ return head_.lock(); }
	C_NodePtr_ HeadNode() const{ return head_.lock(); }

	NodePtr_ TailNode(){ return tail_.lock(); }
	C_NodePtr_ TailNode() const{ return tail_.lock(); }
};

template <class NodeData>
double DirectedEdge<NodeData>::CalcWeightedScore() const
{
	auto tp = tail_.lock();
	if (tp) return tp->Score() * weight_;
	else assert(false);	//"class DirectedEdge: node link error"
}

template <class NodeData>
template <class ActivationFunc>
double DirectedEdge<NodeData>::CalcDeltaWeight(double alpha, double error)
{
	pre_weight_ = weight_;
	delta_ = error * ActivationFunc::df(head_.lock()->PreActivateScore());
	return alpha * tail_.lock()->Score() * delta_;
}


template <class NodeData>
template <class ActivationFunc>
void DirectedEdge<NodeData>::UpdateWeight(double alpha, double beta, double error)
{
	pre_weight_ = weight_;
	delta_ = error * ActivationFunc::df(head_.lock()->PreActivateScore());
	weight_ = weight_ * beta + alpha * tail_.lock()->Score() * delta_;
}


/*
template <class NodeData>
class UndirectedEdge
{
public:
	using NodePtr_ = NodePtr<UndirectedEdge, NodeData>;
	using C_NodePtr_ = C_NodePtr<UndirectedEdge, NodeData>;
	using NodeWPtr_ = NodeWPtr<UndirectedEdge, NodeData>;

private:
	DEdgePtr<NodeData> edge_;

private:
	void AddNode(NodePtr_& side, NodePtr_& snother_side){ edge_; }

public:
	UndirectedEdge(typename sig::Just<double>::type weight = sig::Nothing(SIG_DEFAULT_EDGE_WEIGHT)) : edge_(weight){}
	~UndirectedEdge(){};

	double CalcWeightedScore() const{ return edge_->CalcWeightedScore(); }

	template <class ActivationFunc>
	double CalcDeltaWeight(double alpha, double error){ return edge->CalcDeltaWeight(alpha, error); }

	template <class ActivationFunc>
	void UpdateWeight(double alpha, double beta, double error){ return edge_->UpdateWeight(alpha, beta, error); }


	void Weight(double v){ edge_->Weight(v); }
	double Weight() const{ return edge_->Weight(); }

	double PreWeight() const{ return edge_->PreWeight(); }

	void Delta(double v){ edge_->Delta(v); }
	double Delta() const{ return edge_->Delta(); }

	auto Nodes() ->std::tuple<NodePtr, NodePtr>{ return std::make_tuple(edge_->HeadNode(), edge_->TailNode()); }
	auto Nodes() const ->std::tuple<C_NodePtr, C_NodePtr>{ return std::make_tuple(edge_->HeadNode(), edge_->TailNode()); }
};*/

}
#endif

