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

#include "MLP_Impl.hpp"

namespace signn{
	
template <class InputInfo_, class OutputInfo_>
class Perceptron_Online : public DataFormat<InputInfo_, OutputInfo_>
{
	typedef MLP_Impl<InputInfo_, OutputInfo_> MLP;
	//typedef typename MLP::InputDataPtr InputDataPtr;
	//typedef typename MLP::OutputDataPtr OutputDataPtr;

	MLP mlp_;

	double min_mse_;						//minimum mean-square-error during iteration
	std::shared_ptr<MLP> optimal_state_;	//mlp state when mse is minimum
		
public:
	explicit Perceptron_Online(double learning_rate, double L2_regularization, std::vector<LayerPtr> hidden_layers, double goal_mse = std::numeric_limits<double>::max())
		: mlp_(learning_rate, L2_regularization, hidden_layers), min_mse_(goal_mse), optimal_state_(std::make_shared<MLP>(learning_rate, L2_regularization, hidden_layers)){}
	virtual ~Perceptron_Online(){};

	double Train(InputDataPtr train_data, bool check_mse = true);

	OutputDataPtr Test(InputDataPtr test_data) const;

	void SaveParameter(std::wstring pass, bool select_optimal_state) const{ select_optimal_state ? optimal_state_->SaveParameter(pass) : mlp_.SaveParameter(pass); };

	void LoadParameter(std::wstring pass){ mlp_.LoadParameter(pass); }

	void DebugWeight(std::vector<double> weight) const{ for (uint i=0; i<all_edges_.size(); ++i) all_edges_[i]->Weight(weight[i]); }
};


template <class InputInfo_, class OutputInfo_>
double Perceptron_Online<InputInfo_, OutputInfo_>::Train(InputDataPtr train_data, bool check_mse)
{
	mlp_.ForwardPropagation(*train_data);
	mlp_.BackPropagation(*train_data);

	if (check_mse){
		auto mse = mlp_.MeanSquareError(train_data->Teacher());
		if (mse < min_mse_){
			optimal_state_->CopyWeight(mlp_);
			min_mse_ = mse;
		}
		return mse;
	}
	else return -1;
}

/*
template <class InputInfo_, class OutputInfo_>
template<class Iter1, class Iter2>
double Perceptron_Online<InputInfo_, OutputInfo_>::Train(Iter1 input_begin, Iter1 input_end, Iter2 teacher_begin, Iter2 teacher_end)
{
	InputData input(input_begin, input_end, teacher_begin, teacher_end);
	return Train(input);
}

template <class InputInfo_, class OutputInfo_>
template<class Iter1>
double Perceptron_Online<InputInfo_, OutputInfo_>::Train(Iter1 input_begin, Iter1 input_end, typename OutputInfo_::type teacher)
{
	InputData input(input_begin, input_end, teacher);
	return Train(input);
}
*/

template <class InputInfo_, class OutputInfo_>
typename Perceptron_Online<InputInfo_, OutputInfo_>::OutputDataPtr Perceptron_Online<InputInfo_, OutputInfo_>::Test(InputDataPtr test_data) const
{
	mlp_.ForwardPropagation(*test_data);
	return std::make_shared<OutputData>(test_data, mlp_.OutputScore());
}

}
