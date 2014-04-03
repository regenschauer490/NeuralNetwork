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
class SOM_Online
{
public:
	using SOM_ = SOM_Impl<InputInfo_, SideNodeNum>;
	using Layer_ = typename SOM_::Layer_;
	using LayerPtr_ =typename SOM_::LayerPtr_;
	using C_LayerPtr_ = C_SOMLayerPtr<InputInfo_::dim>;

	using InputDataPtr = typename SOM_::InputDataPtr;	//入力データそのもの
	using InputProxy = typename SOM_::InputProxy;		//入力データ作成用クラス
	using DataRange_ = typename SOM_::DataRange_;		//入力データベクトルの各要素の範囲指定
	
private: 
	SOM_ som_;
	
public:
	//
	SOM_Online(std::initializer_list<typename DataRange_::value_type> input_range);
	//SOM_Online(std::initializer_list<NodeData_> samples) : som_(som_learning_rate) {}

	// 入力データ作成を行うクラスを返す
	// そのクラスを通して、入力データを作成する (InputDataPtr型の入力データが得られる)
	InputProxy MakeInputData() const{ return InputProxy(); }

	// データを逐次的に与えて学習する
	void Train(InputDataPtr const& input){ som_.RenewNeighbor(*input, som_.SearchSimilarity(*input)); }
	
	// 入力データと最も類似した参照ベクトルの座標(y, x)を返す
	auto NearestPosition(InputDataPtr input)  const->std::array<uint, 2>{
		return som_.NearestPosition(input);
	}
};


template <class InputInfo_, size_t SideNodeNum>
SOM_Online<InputInfo_, SideNodeNum>::SOM_Online(std::initializer_list<typename DataRange_::value_type> input_range)
	: som_(som_learning_rate, [&](){
		DataRange_ ref_vector_init;
		for (auto const& e : input_range) ref_vector_init.push_back(e);
		return ref_vector_init;
	}())
{}

}
#endif