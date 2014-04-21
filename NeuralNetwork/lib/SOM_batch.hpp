/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_SOM_BATCH_H
#define SIG_NN_SOM_BATCH_H

#include "SOM_impl.hpp"

namespace signn{

template <class InputInfo_, size_t SideNodeNum, DistanceFunc DistFunc = DistanceFunc::Euclidean>
class SOM_Batch
{
public:
	using SOM = SOM_Impl<InputInfo_, SideNodeNum, DistFunc>;

	using InputDataPtr = typename SOM::InputDataPtr;	//入力データそのもの
	using InputDataSet = typename SOM::InputDataSet;	//入力データ集合
	using InputProxy = typename SOM::InputProxy;		//入力データ作成用クラス
	using DataRange = typename SOM::DataRange_;		//入力データベクトルの各要素の範囲指定
	
private: 
	InputDataSet dataset_;
	SOM som_;
	
private:
	//[0,1]に値を正規化
	void Normalization();

	//平均0, 分散σに値を標準化
	void Standardization();

public:
	SOM_Batch(InputDataSet const& inputs);

	// 入力データ作成を行うクラスを返す
	static InputProxy MakeInputData(){ return SOM::MakeInputData(); }

	// データ全体を学習する操作を何回繰り返すかを指定して実行
	void Train(uint iteration);

	// 入力データと最も類似した参照ベクトルの座標(y, x)を返す
	auto NearestPosition(InputDataPtr input)  const->std::array<uint, 2>{
		return som_.NearestPosition(input);
	}
};


template <class InputInfo_, size_t SideNodeNum, DistanceFunc DistFunc>
SOM_Batch<InputInfo_, SideNodeNum, DistFunc>::SOM_Batch(InputDataSet const& inputs) :
	dataset_(inputs),
	som_(som_learning_rate, SOM::AnalyseRange(inputs))
{}

template <class InputInfo_, size_t SideNodeNum, DistanceFunc DistFunc>
void SOM_Batch<InputInfo_, SideNodeNum, DistFunc>::Train(uint iteration){
	for (uint loop = 0; loop < iteration; ++loop){
		for (auto const& input : dataset_){
			som_.RenewNeighbor(*input, som_.SearchSimilarity(*input));
		}
	}
}

}
#endif