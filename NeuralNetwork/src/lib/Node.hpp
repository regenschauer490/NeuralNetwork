/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_NODE_H
#define SIG_NN_NODE_H

#include "edge.h"

namespace signn{
	
template <class DataType, class Edge>
class Node;

template <class DataType>
class Node<DataType, DirectedEdge<DataType>>
{
public:
	using ParamType_ = ParamType<DataType>;
	using DEdgePtr_ = DEdgePtr<DataType>;

private:
	const double threshold_;

	std::vector<DEdgePtr_> in_;
	std::vector<DEdgePtr_> out_;

	DataType score_;
	
	//cache
	DataType pre_activate_score_;

	SIG_FRIEND_WITH_NODE_AND_EDGE
private:
	void AddInEdge(DEdgePtr_ edge){ in_.push_back(edge); }
	void AddOutEdge(DEdgePtr_ edge){ out_.push_back(edge); }

public:
	Node() : threshold_(threshold_theta), score_(), pre_activate_score_(score_){};
	~Node(){};

	
	DataType AccumulateRawScore() const{
		return std::accumulate(in_.begin(), in_.end(), 0.0, [](ParamType_ sum, decltype(*in_.begin())& e){ return e->CalcWeightedScore() + sum; }) - threshold_;
	}

	template <class ActivationFunc>
	void UpdateScore(){ score_ = ActivationFunc::f(AccumulateRawScore()); }		//enable only DataType := double 

	ParamType_ Score() const{ return score_; }		//return const T& if DataType is user defined types. otherwise return T.
	void Score(ParamType_ v){ score_ = v; }

	ParamType_ PreActivateScore() const{ return pre_activate_score_; }

	auto in_begin() ->decltype(in_.begin()){ return in_.begin(); }
	auto in_begin() const ->decltype(in_.cbegin()){ return in_.cbegin(); }

	auto in_end() ->decltype(in_.end()){ return in_.end(); }
	auto in_end() const ->decltype(in_.cend()){ return in_.cend(); }

	auto out_begin() ->decltype(out_.begin()){ return out_.begin(); }
	auto out_begin() const ->decltype(out_.cbegin()){ return out_.cbegin(); }

	auto out_end() ->decltype(out_.end()){ return out_.end(); }
	auto out_end() const ->decltype(out_.cend()){ return out_.cend(); }
};

/*
template <class DataType>
class Node<DataType, UndirectedEdge>
{
public:
	using ParamType = ParamType<DataType>;

private:
	const double threshold_;

	std::vector<UDEdgePtr> edge_;

	DataType score_;
	
	//cache
	DataType pre_activate_score_;

public:
	Node() : threshold_(threshold_theta), score_(0), pre_activate_score_(0){};
	~Node(){};

	void AddEdge(UDEdgePtr edge){ edge_.push_back(edge); }
	
	DataType AccumulateRawScore() const{
		return std::accumulate(edge_.begin(), edge_.end(), 0.0, [](ParamType sum, decltype(*in_.begin())& e){ return e->CalcWeightedScore() + sum; }) - threshold_;
	}

	template <class ActivationFunc>
	void UpdateScore();

	void UpdateScore(ParamType score){ score_ = score; }
	
	DataType Score() const{ return score_; }
	void Score(ParamType v){ score_ = v; }

	DataType PreActivateScore() const{ return pre_activate_score_; }

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
*/

}
#endif