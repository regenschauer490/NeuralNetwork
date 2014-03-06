/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_SOM_ONLINE_H
#define SIG_NN_SOM_ONLINE_H

#include "layer_som.hpp"
#include "external/distance/distance.hpp"

namespace signn{

template <class InputInfo_>
class SOM_Online : public DataFormat<InputInfo_, OutputInfo_>
{
public:
	using DataFormat = DataFormat<InputInfo_, OutputInfo_>;
	using Layer_ = SOMLayer<InputInfo_::dim>;
	using LayerPtr_ = SOMLayerPtr<InputInfo_::dim>;
	using NodeData_ = typename Layer_::NodeData_;
	using NodePtr_ = typename Layer_::NodePtr_;
	using C_NodePtr_ = typename Layer_::C_NodePtr_;

private: 
	LayerPtr_ layer_;
	
private:
	void Init();

	//入力のベクトルと最も類似した参照ベクトルを探す
	C_NodePtr_ SearchSimilarity(NodeData_ const& input) const;
	
public:
	SOM_Online() : layer_(LayerPtr_(new Layer_(InputInfo_::dim))){}
	
	void Train(InputDataPtr input);
};

template <class InputInfo_, class OutputInfo_>
void SOM_Online<InputInfo_, OutputInfo_>::Init()
{
}

template <class InputInfo_, class OutputInfo_>
auto SOM_Online<InputInfo_, OutputInfo_>::SearchSimilarity(NodeData_ const& input) const
{
	sigdm::EuclideanDistance dist;

	for (auto const& row_nodes : layer_){
		for (auto const& node : row_nodes){
			dist(input, *node->Score(), )
		}
	}
}

}
#endif