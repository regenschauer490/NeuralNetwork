/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_SOM_ONLINE_H
#define SIG_NN_SOM_ONLINE_H

#include "SOM_impl.hpp"

namespace signn{

template <class InputInfo_, size_t SideNodeNum, DistanceFunc DistFunc = DistanceFunc::Cosine>
class SOM_Online
{
public:
	using SOM = SOM_Impl<InputInfo_, SideNodeNum, DistFunc>;

	using InputDataPtr = typename SOM::InputDataPtr;	//入力データそのもの
	using InputDataSet = typename SOM::InputDataSet;	//入力データ集合
	using InputProxy = typename SOM::InputProxy;		//入力データ作成用クラス
	using DataRange = typename SOM::DataRange_;		//入力データベクトルの各要素の範囲指定
	
private: 
	SOM som_;
	
public:
	//
	SOM_Online(std::initializer_list<typename DataRange::value_type> input_range);
	SOM_Online(InputDataSet inputs) : som_(som_learning_rate, SOM::AnalyseRange(inputs)) {}
	
	// 入力データ作成を行うクラスを返す
	static InputProxy MakeInputData(){ return SOM::MakeInputData(); }

	// データを逐次的に与えて学習する
	void Train(InputDataPtr const& input){ som_.RenewNeighbor(*input, som_.SearchSimilarity(*input)); }
	
	// 入力データと最も類似した参照ベクトルの座標(y, x)を返す
	auto NearestPosition(InputDataPtr input)  const->std::array<uint, 2>{
		return som_.NearestPosition(input);
	}
};


template <class InputInfo_, size_t SideNodeNum, DistanceFunc DistFunc>
SOM_Online<InputInfo_, SideNodeNum, DistFunc>::SOM_Online(std::initializer_list<typename DataRange::value_type> input_range)
	: som_(som_learning_rate, [&](){
		DataRange ref_vector_init;
		for (auto const& e : input_range) ref_vector_init.push_back(e);
		return ref_vector_init;
	}())
{}

}
#endif