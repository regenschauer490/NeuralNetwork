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
	using Layer_ = SOMLayer<InputInfo_::dim>;			//InputInfo_::dim : 参照ベクトルの次元
	using LayerPtr_ = SOMLayerPtr<InputInfo_::dim>;
	using C_LayerPtr_ = C_SOMLayerPtr<InputInfo_::dim>;
	using NodeData_ = typename Layer_::NodeData_;
	using NodePtr_ = typename Layer_::NodePtr_;
	using C_NodePtr_ = typename Layer_::C_NodePtr_;

private:
	//入力データ形式の指定
	struct InputProxy :
		public UnsupervisedProxy
	{};

private: 
	LayerPtr_ layer_;
	double alpha_;		//learning-rate
	
private:
	void Init();

	//指定参照ベクトルとその近傍の参照ベクトルを更新 (0 ≦ learning_rate ≦ 1)
	void RenewNeighbor(InputData const& input, NodePtr_ center, double learning_rate);

	//入力のベクトルと最も類似した参照ベクトルを探す
	NodePtr_ SearchSimilarity(InputData const& input) const;
	
public:
	SOM_Online() : layer_(LayerPtr_(new Layer_(SideNodeNum, SideNodeNum))), alpha_(som_learning_rate){}

	InputProxy MakeInputData() const{ return InputProxy(); }


	void Train(InputDataPtr input){
		RenewNeighbor(*input, SearchSimilarity(*input), alpha_);
	}

	//入力データと最も類似した参照ベクトルの座標(y, x)を返す
	auto NearestPosition(InputDataPtr input)  const->std::array<uint, 2>{
		return layer_->SearchPosition(SearchSimilarity(*input));
	}
};


template <class InputInfo_, size_t SideNodeNum>
void SOM_Online<InputInfo_, SideNodeNum>::Init()
{
}

template <class InputInfo_, size_t SideNodeNum>
void SOM_Online<InputInfo_, SideNodeNum>::RenewNeighbor(InputData const& input, NodePtr_ center, double learning_rate)
{
	auto MakeUpdateFunc = [&](double alpha) ->std::function<void(NodeData_&)>{
		return [&,alpha](NodeData_& pre_score){
			InputArrayType_ tmp;
			for (uint i = 0; i<InputInfo_::dim; ++i){
				tmp[i] = alpha * input.Input()[i];
			}

			//score' = score * (1 - alpha) + alpha * input
			sig::CompoundAssignment([](double& dest, double corr){ dest *= corr; }, pre_score, (1-alpha));			//score *= (1 - alpha)
			sig::CompoundAssignment([](double& dest, double in){ dest += in; }, pre_score, tmp);			//score += (input * alpha)
		};
	};

	center->UpdateScore(MakeUpdateFunc(learning_rate));		//中心の参照ベクトルを更新

	for(auto edge_it = center->out_begin(), end = center->out_end(); edge_it != end; ++edge_it){
		auto edge = *edge_it;
		double corr = learning_rate * std::exp(- edge->Weight());
		edge->TailNode()->UpdateScore(MakeUpdateFunc(corr));		//近傍の参照ベクトルを更新
	}
}

template <class InputInfo_, size_t SideNodeNum>
auto SOM_Online<InputInfo_, SideNodeNum>::SearchSimilarity(InputData const& input) const->NodePtr_
{
	typename sigdm::EuclideanDistance Dist;		//todo: 距離関数の選択をできるように拡張
	double min_dist = std::numeric_limits<double>::max();
	NodePtr_ nearest = nullptr;			//NodePtr_* にすべきか検討

	for (auto const& node : *static_cast<C_LayerPtr_>(layer_)){
		auto dist = Dist(node->Score(), input.Input());
		if (dist < min_dist){
			min_dist = dist;
			nearest = node;
		}
	}

	return nearest;
}

}
#endif