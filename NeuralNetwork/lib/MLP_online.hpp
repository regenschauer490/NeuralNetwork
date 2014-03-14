/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_MLP_ONLINE_H
#define SIG_NN_MLP_ONLINE_H

#include "MLP_impl.hpp"

#undef min
#undef max

namespace signn{
	
template <class InputInfo_, class OutputInfo_>
class Perceptron_Online : public DataFormat<InputInfo_, OutputInfo_>
{
public:
	using MLP_ = MLP_Impl<InputInfo_, OutputInfo_>;
	using Layer_ = typename MLP_::Layer_;
	using LayerPtr_ = typename MLP_::LayerPtr_;
	
private:
	MLP_ mlp_;

	double min_mse_;						//minimum mean-square-error during iteration
	std::shared_ptr<MLP_> optimal_state_;	//mlp state when mse is minimum
		
public:
	Perceptron_Online(double learning_rate, double L2_regularization, std::vector<LayerPtr_>&& hidden_layers, double goal_mse = std::numeric_limits<double>::max())
		: mlp_(learning_rate, L2_regularization, hidden_layers), min_mse_(goal_mse), optimal_state_(std::make_shared<MLP_>(learning_rate, L2_regularization, hidden_layers)){}

	static LayerPtr_ MakeMidLayer(uint node_num){ return MLP_::MakeMidLayer(node_num); }


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
#endif