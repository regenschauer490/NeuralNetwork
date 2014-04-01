/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_SOM_ONLINE_H
#define SIG_NN_SOM_ONLINE_H

#include "SOM_impl.hpp"

namespace signn{

template <class InputInfo_, size_t SideNodeNum>
class SOM_Online : public DataFormat<InputInfo_, OutputInfo<SOMLayerInfo<SideNodeNum>>>
{
public:
	using SOM_ = SOM_Impl<InputInfo_, SideNodeNum>;
	using Layer_ = typename SOM::Layer_;
	using LayerPtr_ =typename SOM::LayerPtr_;
	using C_LayerPtr_ = C_SOMLayerPtr<InputInfo_::dim>;

private:
	//入力データ形式の指定
	struct InputProxy :
		public UnsupervisedProxy
	{};

private: 
	SOM_ som_;
	
public:
	SOM_Online() : som_(som_learning_rate) {}

	//入力データ作成を行うクラスを返す
	//そのクラスを通して、入力データを作成する (InputDataPtr型の入力データが得られる)
	InputProxy MakeInputData() const{ return InputProxy(); }


/* InputDataPtrを要素に持つコンテナを与えて学習を行う */
	
	//データ全体を学習する操作を何回繰り返すかを指定して実行
	template <class InputDataSet>
	void Train(InputDataSet const& inputs, uint iteration);


	//入力データと最も類似した参照ベクトルの座標(y, x)を返す
	auto NearestPosition(InputDataPtr input)  const->std::array<uint, 2>{
		return som_->NearestPosition(input);
	}
};


template <class InputInfo_, size_t SideNodeNum>
template <class InputDataSet>
void SOM_Online<InputInfo_, SideNodeNum>::Train(InputDataSet const& inputs, uint iteration){
	for (uint loop = 0; loop < iteration; ++loop){
		for (auto const& input : inputs){
			RenewNeighbor(*input, SearchSimilarity(*input), alpha_);
		}
	}
}

}
#endif