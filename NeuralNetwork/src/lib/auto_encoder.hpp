/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_AUTO_ENCODER_H
#define SIG_NN_AUTO_ENCODER_H

#include "MLP_online.hpp"

namespace signn{

template <class InputInfo_, uint HiddenDim, class OutputInfo_ = OutputInfo<OutputLayerType::BinaryClassification, InputInfo_::dim>>
class AutoEncoder : public DataFormat<InputInfo_, OutputInfo_>
{
	typedef Perceptron_Online<InputInfo_, OutputInfo_> Perceptron;

	LayerPtr hidden_;
	Perceptron ac_;

public:
	AutoEncoder(double learning_rate, double L2_regularization, double goal_mse = std::numeric_limits<double>::max()) : hidden_(Layer::MakeInstance(HiddenDim)), ac_(learning_rate, L2_regularization, std::vector<LayerPtr>{hidden_}, goal_mse){
		static_assert(InputInfo_::dim == OutputInfo_::dim, "invalid dimension: different dim between input and output");
	}

	double Train(InputDataPtr train_data, bool return_sqerror = false);

	OutputDataPtr Test(InputDataPtr test_data) const{ return ac_.Test(test_data); }

	void SaveParameter(std::wstring pass, bool select_optimal_state) const{ ac_.SaveParameter(pass, select_optimal_state); }

	void LoadParameter(std::wstring pass) const{ ac_.LoadParameter(pass); }

	//教師信号なし  (オートエンコーダ用)
	template<class Iter1, typename = decltype(*std::declval<Iter1&>(), void(), ++std::declval<Iter1&>(), void())>
	std::shared_ptr<InputData> MakeInputData(Iter1 input_begin, Iter1 input_end) const{
		uint i = 0;
		std::array<typename OutputInfo_::type, OutputInfo_::dim> teacher;
		for (auto in = input_begin; in != input_end; ++in, ++i) teacher[i] = *in;
		return std::make_shared<InputData>(input_begin, input_end, teacher.begin(), teacher.end(), false);
	}
};


template <class InputInfo_, uint HiddenDim, class OutputInfo_>
double AutoEncoder<InputInfo_, HiddenDim, OutputInfo_>::Train(InputDataPtr train_data, bool return_sqerror)
{
	if (train_data->IsTestData()){
		assert(false);
		auto& input = train_data->Input();
		auto& teacher = const_cast<InputData::TeacherDataArray&>(train_data->Teacher());

		for (uint i = 0; i < InputInfo_::dim; ++i) teacher[i] = input[i];
	}

	return ac_.Train(train_data, return_sqerror);
}

}
#endif