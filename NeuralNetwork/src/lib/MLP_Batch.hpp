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
class Perceptron_Batch : public DataFormat<InputInfo_, OutputInfo_>
{
	typedef MLP_Impl<InputInfo_, OutputInfo_> MLP;

	const double alpha_;
	const double beta_;	//L2 regularization

	MLP mlp_;
	std::vector<MLP> copy_mlp_;

	double min_mse_;						//minimum mean-square-error during iteration
	std::shared_ptr<MLP> optimal_state_;	//mlp state when mse is minimum
	
public:
	explicit Perceptron_Batch(std::vector<LayerPtr> hidden_layers, double goal_mse = std::numeric_limits<double>::max()) : alpha_(learning_rate), beta_(L2_regularization), mlp_(learning_rate, L2_regularization, hidden_layers),
		min_mse_(goal_mse), optimal_state_(std::make_shared<MLP>(learning_rate, L2_regularization, hidden_layers))
	{
		for (uint i = 0; i < THREAD_NUM; ++i){
			copy_mlp_.emplace_back(alpha_, beta_, hidden_layers);
			copy_mlp_[i].CopyWeight(mlp_);
		}
	}
	virtual ~Perceptron_Batch(){};

	double Train(std::vector<InputDataPtr> const& inputs);

	OutputDataPtr Test(InputDataPtr test_data) const;

	void SaveParameter(std::wstring pass, bool select_optimal_state) const{ select_optimal_state ? optimal_state_->SaveParameter(pass) : mlp_.SaveParameter(pass); };
};



template <class InputInfo_, class OutputInfo_>
double Perceptron_Batch<InputInfo_, OutputInfo_>::Train(std::vector<InputDataPtr> const& inputs)
{
	const uint div_size = inputs.size() / THREAD_NUM;
	std::vector< std::future< std::tuple< double, std::vector< std::vector<double >>> >> task;
	double mse = 0;
	std::vector<std::vector<double>> delta_weight;

	auto LearnImpl = [](MLP& mlp, std::vector<InputDataPtr>::const_iterator begin, std::vector<InputDataPtr>::const_iterator end)
	{
		std::vector< std::vector<double>> result;
		double l_mse;
		bool first = true;

		while(true){
			mlp.ForwardPropagation(**begin);
			auto d_weight = mlp.BackPropagation_Delay(**begin);

			if (first){
				l_mse = mlp.MeanSquareError((*begin)->Teacher());
				result = std::move(d_weight);
				first = false;
			}
			else{
				l_mse += mlp.MeanSquareError((*begin)->Teacher());

				for (uint l = 0; l < d_weight.size(); ++l){
					for (uint n = 0; n < d_weight[l].size(); ++n) result[l][n] += d_weight[l][n];
				}
			}

			if (begin == end) break;
			++begin;
		}

		return std::make_tuple(l_mse, std::move(result));
	};

	
	for (uint i = 0, w = 0, we = div_size-1; i<THREAD_NUM; ++i, w += div_size, we += div_size){
		if (i == THREAD_NUM-1) we = inputs.size() - 1;
		task.push_back(std::async(std::launch::async, LearnImpl, copy_mlp_[i], inputs.cbegin() + w, inputs.cbegin() + we));
	}

	std::vector<std::tuple < double, std::vector < std::vector<double >>>> result;
	for (auto& t : task){
		result.push_back(t.get());
	}

	for (auto& r : result){
		mse += std::get<0>(r);

		auto& d_weight = std::get<1>(r);
		if (delta_weight.empty()) delta_weight = std::move(d_weight);
		else{
			for (uint l = 0; l < d_weight.size(); ++l){
				for (uint n = 0; n < d_weight[l].size(); ++n){
					delta_weight[l][n] += d_weight[l][n];
				}
			}
		}
	}
	
	for (auto& dw : delta_weight){
		for (auto& dwn : dw) dwn = dwn / inputs.size();
	}
	mse /= inputs.size();

	mlp_.RenewEdgeWeight(delta_weight);

	for (auto& cpy : copy_mlp_) cpy.CopyWeight(mlp_);
	
	if (mse < min_mse_){
		optimal_state_->CopyWeight(mlp_);
		min_mse_ = mse;
	}

	return mse;
}

template <class InputInfo_, class OutputInfo_>
typename Perceptron_Batch<InputInfo_, OutputInfo_>::OutputDataPtr Perceptron_Batch<InputInfo_, OutputInfo_>::Test(InputDataPtr test_data) const
{
	mlp_.ForwardPropagation(*test_data);
	return std::make_shared<OutputData>(test_data, mlp_.OutputScore());
}

}
