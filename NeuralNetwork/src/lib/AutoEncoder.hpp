/*
The MIT License(MIT)

Copyright(c) 2014 Akihiro Nishimura

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files(the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include "MLP_Online.hpp"

namespace signn{

template <class InputInfo_, uint HiddenDim, class OutputInfo_ = OutputInfo<OutputLayerType::BinaryClassification, InputInfo_::dim>>
class AutoEncoder : public DataFormat<InputInfo_, OutputInfo_>
{
	typedef Perceptron_Online<InputInfo_, OutputInfo_> Perceptron;

	LayerPtr hidden_;
	Perceptron ac_;

private:


public:
	AutoEncoder() : hidden_(Layer::MakeInstance(HiddenDim)), ac_(std::vector<LayerPtr>{hidden_}){ static_assert(InputInfo_::dim == OutputInfo_::dim, "invalid dimension: different dim between input and output"); }

	double Learn(InputDataPtr train_data, bool return_sqerror = false);

	OutputDataPtr Test(InputDataPtr test_data) const{ return ac_.Test(test_data); }

	void SaveParameter(std::wstring pass) const{ ac_.SaveParameter(pass); }

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
double AutoEncoder<InputInfo_, HiddenDim, OutputInfo_>::Learn(InputDataPtr train_data, bool return_sqerror)
{
	if (train_data->IsTestData()){
		assert(false);
		auto& input = train_data->Input();
		auto& teacher = const_cast<InputData::TeacherDataArray&>(train_data->Teacher());

		for (uint i = 0; i < InputInfo_::dim; ++i) teacher[i] = input[i];
	}

	return ac_.Learn(train_data, return_sqerror);
}

}
