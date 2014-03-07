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

template <class InputInfo_, size_t SideNodeNum>
class SOM_Online : public DataFormat<InputInfo_, OutputInfo<SOMLayerInfo<SideNodeNum>>>
{
public:
	using Layer_ = SOMLayer<InputInfo_::node_num>;
	using LayerPtr_ = SOMLayerPtr<InputInfo_::node_num>;
	using NodeData_ = typename Layer_::NodeData_;
	using NodePtr_ = typename Layer_::NodePtr_;
	using C_NodePtr_ = typename Layer_::C_NodePtr_;

private: 
	LayerPtr_ layer_;
	
private:
	void Init();

	//入力のベクトルと最も類似した参照ベクトルを探す
	C_NodePtr_ SearchSimilarity(InputData const& input) const;
	
public:
	SOM_Online() : layer_(LayerPtr_(new Layer_(InputInfo_::node_num))){}
	
	void Train(InputDataPtr input);
};

template <class InputInfo_, size_t SideNodeNum>
void SOM_Online<InputInfo_, SideNodeNum>::Init()
{
}

template <class InputInfo_, size_t SideNodeNum>
auto SOM_Online<InputInfo_, SideNodeNum>::SearchSimilarity(InputData const& input) const->C_NodePtr_
{
	typename sigdm::EuclideanDistance Dist;
	double min_dist = std::numeric_limits<double>::max();
	NodePtr_ nearest = nullptr;		//NodePtr_* にすべきか検討

	for (auto const& node : layer_){
		if (auto dist = Dist(input, *node->Score()) < min_dist){
			min_dist = dist;
			nearest = node;
		}
	}

	return nearest;
}

}
#endif