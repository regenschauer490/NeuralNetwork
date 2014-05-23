/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_SOM_IMPL_H
#define SIG_NN_SOM_IMPL_H

#include "layer_som.hpp"
#include "distance/distance.hpp"
#include "SigUtil/lib/iteration.hpp"

namespace signn{

template <class InputInfo_, size_t SideNodeNum>
class SOM_Impl : public DataFormat<InputInfo_, OutputInfo<SOMLayerInfo<SideNodeNum>>>
{
public:
	using Layer_ = SOMLayer<InputInfo_::dim>;			//InputInfo_::dim : 参照ベクトルの次元
	using LayerPtr_ = SOMLayerPtr<InputInfo_::dim>;
	using C_LayerPtr_ = C_SOMLayerPtr<InputInfo_::dim>;
	using NodeData_ = typename Layer_::NodeData_;
	using NodePtr_ = typename Layer_::NodePtr_;
	using C_NodePtr_ = typename Layer_::C_NodePtr_;
	using DataRange_ = typename Layer_::DataRange_;

public:
	// 入力データ形式の指定
	struct InputProxy :
		public UnsupervisedProxy
	{};

private: 
	LayerPtr_ layer_;
	double alpha_;		//learning-rate (0 ≦ alpha_ ≦ 1)
	
public:
	SOM_Impl(double learning_rate, DataRange_ ref_vector_init) : layer_(LayerPtr_(new Layer_(SideNodeNum, SideNodeNum, ref_vector_init))), alpha_(learning_rate){}
	
	// 入力データ作成を行うクラスを返す
	// そのクラスを通して、入力データを作成する (InputDataPtr型の入力データが得られる)
	static InputProxy MakeInputData(){ return InputProxy(); }

	// データセットから各要素(列)の値の範囲を求める
	static DataRange_ AnalyseRange(InputDataSet const& inputs);

	void LearningRate(double rate){ alpha_ = rate; }
	double LearningRate(){ return alpha_; }

	// InputDataPtrを与えて学習を行う
	void Train(InputDataPtr input){
		RenewNeighbor(*input, SearchSimilarity(*input), alpha_);
	}

	// 指定参照ベクトルとその近傍の参照ベクトルを更新
	void RenewNeighbor(InputData const& input, NodePtr_ center);

	// 入力のベクトルと最も類似した参照ベクトルを探す
	NodePtr_ SearchSimilarity(InputData const& input) const;

	// 入力データと最も類似した参照ベクトルの座標(y, x)を返す
	auto NearestPosition(InputDataPtr input)  const->std::array<uint, 2>{
		return layer_->Position(SearchSimilarity(*input));
	}

	//debug用
	sig::array<double, InputInfo_::dim> RefVector(uint x, uint y){ return layer_->Access(y, x)->Score(); }
};


template <class InputInfo_, size_t SideNodeNum>
auto SOM_Impl<InputInfo_, SideNodeNum>::AnalyseRange(InputDataSet const& inputs) ->DataRange_
{
	uint dim = InputInfo_::dim;
	DataRange_ range(dim, std::make_pair(0, 0));

	bool f = true;
	for (auto const& input : inputs){
		auto const& vec = input->Input();

		for (uint i=0; i<dim; ++i){
			if (f){
				std::get<0>(range[i]) = vec[i];
				std::get<1>(range[i]) = vec[i];
			}
			else{
				if (vec[i] < std::get<0>(range[i])){
					std::get<0>(range[i]) = vec[i];
				}
				else if (vec[i] > std::get<1>(range[i])){
					std::get<1>(range[i]) = vec[i];
				}
			}
		}
		f = false;
	}
	return range;
}

template <class InputInfo_, size_t SideNodeNum>
void SOM_Impl<InputInfo_, SideNodeNum>::RenewNeighbor(InputData const& input, NodePtr_ center)
{
	auto MakeUpdateFunc = [&](double alpha) ->std::function<void(NodeData_&)>{
		return [&,alpha](NodeData_& pre_score){
			InputArrayType_ tmp;
			for (uint i = 0; i<InputInfo_::dim; ++i){
				tmp[i] = alpha * input.Input()[i];
			}

			//score' = score * (1 - alpha) + alpha * input
			sig::compound_assignment([](double& dest, double corr){ dest *= corr; }, pre_score, (1-alpha));			//score *= (1 - alpha)
			sig::compound_assignment([](double& dest, double in){ dest += in; }, pre_score, tmp);			//score += (input * alpha)
		};
	};

	center->UpdateScore(MakeUpdateFunc(alpha_));		//中心の参照ベクトルを更新

	for(auto edge_it = center->out_begin(), end = center->out_end(); edge_it != end; ++edge_it){
		auto edge = *edge_it;
		double corr = alpha_ * std::exp(-0.01 * edge->Weight());
		edge->TailNode()->UpdateScore(MakeUpdateFunc(corr));		//近傍の参照ベクトルを更新
	}
}

template <class InputInfo_, size_t SideNodeNum>
auto SOM_Impl<InputInfo_, SideNodeNum>::SearchSimilarity(InputData const& input) const ->NodePtr_
{
	auto df = sigdm::CosineSimilarity();
	double min_dist = std::numeric_limits<double>::max();
	NodePtr_ nearest = nullptr;			//NodePtr_* にすべきか検討

	for (auto const& node : *static_cast<C_LayerPtr_>(layer_)){
		auto dist = *df(node->Score(), input.Input());
		if (dist < min_dist){
			min_dist = dist;
			nearest = node;
		}
	}

	return nearest;
}

}
#endif