/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_SOM_ONLINE_H
#define SIG_NN_SOM_ONLINE_H

#include "edge.h"

namespace signn{

template <size_t RefVecDim>
class SOMLayer
{
	using SOMLayerPtr_ = SOMLayerPtr<RefVecDim>;
	using Node_ = Node<DirectedEdge, std::array<double,RefVecDim>>;
	using NodePtr_ = NodePtr<Node_>;
	
	const uint row_num_;
	const uint col_num_;
	std::vector<std::vector<NodePtr_>> nodes_;
	
	SIG_FRIEND_WITH_LAYER;
private:
	SOMLayerPtr CloneImpl() const{ return std::make_shared<SOMLayer<RefVecDim>>(this->row_num_, this->col_num_)); }

public:
	SOMLayer(uint row_num, uint col_num);
};

template <size_t RefVecDim>
SOMLayer<RefVecDim>::SOMLayer(uint row_num, uint col_num) : row_num_(row_num), col_num_(col_num),
		nodes_(std::vector<std::vector<NodePtr>>(row_num, std::vector<NodePtr>(col_num)))
{
	auto CalcEdgeCost = [&](NodePtr_ const& nd, NodePtr_ const& na){
		using NV = decltype(*nd->Score().begin());
		auto delta = sig::ZipWith([&](NV vd, NV va){ return sig::DeltaAbs(va, vd); }, nd->Score(), na->Score());
		return std::accumulate(std::begin(delta), std::end(delta), 0.0);
	};
	
	for(uint rd=0; rd<row_num; ++rd){
		for(uint cd=0; cd<col_num; ++cd){
			for(uint ra=rd; ra<row_num; ++ra){
				for(uint ca=cd; ca<col_num; ++ca){
					auto edge = std::make_shared<DirectedEdge>(nodes_[nodes_[rd][cd], nodes_[ra][ca], CalcEdgeCost);
					nodes_[rd][cd]->AddOutEdge(edge);
					nodes_[ra][ca]->AddInEdge(edge);
				}
			}
		}
	}
}

/*
class HoneycamLayer : public SOMLayer
{
	const uint node_num_;
	std::vector<NodePtr> nodes_;
	
	SIG_FRIEND_WITH_LAYER;
private:
	virtual LayerPtr CloneImpl() const override{ return std::shared_ptr<Layer>(new HoneycamLayer()); }

public:
	HoneycamLayer(uint row_num, uint col_num) : Layer(row_num * col_num){}
};
*/
template <class InputInfo_, class OutputInfo_>
class SOM_Online
{
public:
	using DataFormat = DataFormat<InputInfo_, OutputInfo_>;

private: 
	
private:
	void Init();
	
public:
	SOM_Online(){}
	
	void Train(InputDataPtr input);
};

template <class InputInfo_, class OutputInfo_>
void SOM_Online<InputInfo_, OutputInfo_>::Init()
{
}

}
#endif