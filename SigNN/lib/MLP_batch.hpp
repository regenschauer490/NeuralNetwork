/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_MLP_BATCH_H
#define SIG_NN_MLP_BATCH_H

#include "MLP_impl.hpp"

namespace signn{
	
template <class InputInfo_, class OutputInfo_>
class Perceptron_Batch : public DataFormat<InputInfo_, OutputInfo_>
{
public:
	using MLP_ = MLP_Impl<InputInfo_, OutputInfo_>;
	using LayerPtr_ = typename MLP_::LayerPtr_;

private:
	const double alpha_;
	const double beta_;	//L2 regularization

	MLP_ mlp_;
	std::vector<MLP_> copy_mlp_;

	double min_mse_;						//minimum mean-square-error during iteration
	std::shared_ptr<MLP_> optimal_state_;	//mlp state when mse is minimum

private:
	//入力データ形式の指定
	struct InputProxy :
		public RawVectorProxy,
		public std::conditional<std::is_same<OutputType_, bool>::value, ClassificationProxy, RegressionProxy>::type,
		public UnsupervisedProxy
	{};

private:
	void ParameterCopy2Slave(){
		for (auto& copy : copy_mlp_){
			copy.CopyWeight(mlp_);
		}

	}

public:
	Perceptron_Batch(double learning_rate, double L2_regularization, std::vector<LayerPtr_> hidden_layers, double goal_mse = std::numeric_limits<double>::max())
		: alpha_(learning_rate), beta_(L2_regularization), mlp_(learning_rate, L2_regularization, hidden_layers),
		min_mse_(goal_mse), optimal_state_(std::make_shared<MLP_>(learning_rate, L2_regularization, hidden_layers))
	{
		for (uint i = 0; i < THREAD_NUM; ++i){
			copy_mlp_.emplace_back(alpha_, beta_, hidden_layers);
			copy_mlp_[i].CopyWeight(mlp_);
		}
	}

	static LayerPtr_ MakeMidLayer(uint node_num){ return MLP_::MakeMidLayer(node_num); }

	InputProxy MakeInputData() const{ return InputProxy(); }


	double Train(std::vector<InputDataPtr> const& inputs);

	OutputDataPtr Test(InputDataPtr test_data) const;

	void SaveParameter(std::wstring pass, bool select_optimal_state) const{ select_optimal_state ? optimal_state_->SaveParameter(pass) : mlp_.SaveParameter(pass); };

	void LoadParameter(std::wstring pass){
		mlp_.LoadParameter(pass); 
		ParameterCopy2Slave(); 
	}
};



template <class InputInfo_, class OutputInfo_>
double Perceptron_Batch<InputInfo_, OutputInfo_>::Train(std::vector<InputDataPtr> const& inputs)
{
	const uint div_size = inputs.size() / THREAD_NUM;
	std::vector< std::future< std::tuple< double, std::vector< std::vector<double >>> >> task;
	double mse = 0;
	std::vector<std::vector<double>> delta_weight;

	auto LearnImpl = [](MLP_& mlp, std::vector<InputDataPtr>::const_iterator begin, std::vector<InputDataPtr>::const_iterator end)
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
#endif
