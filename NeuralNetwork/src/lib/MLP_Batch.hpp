/*
The MIT License(MIT)

Copyright(c) 2013 Akihiro Nishimura

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

#include "InputLayer.hpp"
#include "OutputLayer.hpp"
#include "ParameterPack.hpp"
#include "Node.h"
#include "Edge.h"
#include "info.hpp"
#include "info.hpp"

namespace signn{
	
template <class InputInfo_, class OutputInfo_>
class Perceptron_Batch
{
public:
	class InputData
	{
		std::array<typename InputInfo_::type, InputInfo_::dim> input_;
		std::array<typename OutputInfo_::type, OutputInfo_::dim> teacher_;

	public:
		template<class Iter1, class Iter2>
		InputData(Iter1 input_first, Iter1 input_last, Iter2 teacher_first, Iter2 teacher_last){
			uint i,j;
			for (i = 0; i < InputInfo::dim && input_first != input_last; ++i, ++input_first) input_[i] = *input_first;
			for (j = 0; j < OutputInfo::dim && teacher_first != teacher_last; ++j, ++taecher_first) teacher_[j] = *teacher_first;
			assert(i == InputInfo::dim && j == OutputInfo::dim && input_first == input_last && teacher_first == teacher_las, "invalid input data");
		}

		template<class Iter1>
		InputData(Iter1 input_first, Iter1 input_last, typename OutputInfo_::type teacher){
			uint i;
			for (i = 0; i < InputInfo_::dim && input_first != input_last; ++i, ++input_first) input_[i] = *input_first;
			teacher_[0] = teacher;
			assert(i == InputInfo_::dim && 1 == OutputInfo_::dim && input_first == input_last, "invalid input data");
		}

		template<class Iter1>
		InputData(Iter1 input_first, Iter1 input_last){
			uint i;
			for (i = 0; i < InputInfo_::dim && input_first != input_last; ++i, ++input_first) input_[i] = *input_first;
			assert(i == InputInfo_::dim && 1 == OutputInfo_::dim && input_first == input_last, "invalid input data");
		}

		std::array<typename InputInfo_::type, InputInfo_::dim> const& Input() const{ return input_; }
		std::array<typename OutputInfo_::type, OutputInfo_::dim> const& Teacher() const{ return teacher_; }
	};

private:
	struct MLP_Impl{
		InputLayerPtr<InputInfo_> in_layer_;
		OutputLayerPtr<OutputInfo_> out_layer_;
		std::vector<LayerPtr> layers_;	//all layers
	
		//ParameterPack parameters_;
		double alpha_;

		//cache
		std::vector< std::vector<DEdgePtr>> all_edges_;
	
	public:
		MLP_Impl(){}
		MLP_Impl(std::vector<LayerPtr> hidden_layers) :
			in_layer_(InputLayerPtr<InputInfo_>(new InputLayer<InputInfo_>())),
			out_layer_(OutputLayerPtr<OutputInfo_>(new typename LayerTypeMap<OutputInfo_::e_layertype>::layertype<OutputInfo_>())),
			alpha_(learning_rate)
		{
			layers_.push_back(in_layer_);
			for (auto& l : hidden_layers) layers_.push_back(l->CloneInitInstance());
			layers_.push_back(out_layer_);
			MakeLink();
		}

		void CopyWeight(MLP_Impl const& src){
			for (uint l=0; l< all_edges_.size(); ++l){
				for (uint e = 0; e < all_edges_[l].size(); ++e) all_edges_[l][e]->Weight(src.all_edges_[l][e]->Weight());
			}
		}

		void MakeLink();

		void ForwardPropagation(InputData const& input);

		std::vector< std::vector<double>> BackPropagation(InputData const& input);
	};

private:
	MLP_Impl mlp_;
	std::array<MLP_Impl, THREAD_NUM> copy_mlp_;

public:
	Perceptron_Batch(std::vector<LayerPtr> hidden_layers) : mlp_(hidden_layers){
		for (uint i = 0; i < THREAD_NUM; ++i){
			copy_mlp_[i] = MLP_Impl(hidden_layers);
			copy_mlp_[i].CopyWeight(mlp_);
		}
	}
	~Perceptron_Batch(){};

	double Learn(std::vector<InputData> const& inputs);

	template<class Iter1>
	C_OutputLayerPtr<OutputInfo_> Test(Iter1 input_first, Iter1 input_last);
};



template <class InputInfo_, class OutputInfo_>
void Perceptron_Batch<InputInfo_, OutputInfo_>::MLP_Impl::MakeLink()
{
	//make links between nodes
	auto MakeLink = [&](LayerPtr layer_prev, LayerPtr layer_next, std::vector<DEdgePtr>& cache){
		for (auto& l1 : *layer_prev){
			for (auto& l2 : *layer_next){
				auto edge = std::make_shared<DirectedEdge>(l1, l2);
				l1->AddOutEdge(edge);
				l2->AddInEdge(edge);
				cache.push_back(edge);
			}
		}
	};

	for (uint i = 1; i < layers_.size(); ++i){
		all_edges_.push_back(std::vector<DEdgePtr>());
		MakeLink(layers_[i - 1], layers_[i], all_edges_.back());
	}
}


template <class InputInfo_, class OutputInfo_>
void Perceptron_Batch<InputInfo_, OutputInfo_>::MLP_Impl::ForwardPropagation(InputData const& input)
{
	in_layer_->SetData(input.Input());
	for (auto& l : layers_){
		l->UpdateNodeScore();
	}
}


template <class InputInfo_, class OutputInfo_>
std::vector< std::vector<double>> Perceptron_Batch<InputInfo_, OutputInfo_>::MLP_Impl::BackPropagation(InputData const& input)
{
	const uint lsize = layers_.size()-1;
	std::vector< std::vector<double> > weight(lsize);
	auto const& teacher = input.Teacher();

	weight[lsize-1] = out_layer_->CalcEdgeWeight(alpha_, teacher);
	for (int i = layers_.size() - 2; i > 0; --i){
		weight[i-1] = layers_[i]->CalcEdgeWeight(alpha_);
	}
	
/*
	std::cout << out_layer_->operator[](0)->Score() << ", t: " << teacher[0] << ", se:" << pow(out_layer_->operator[](0)->Score() - teacher[0], 2) << std::endl;
	uint i = 0;
	for (auto& e : all_edges_){
		std::cout << ++i << " : " << e->PreWeight() << " -> " << e->Weight() << ", d: " << e->Weight() - e->PreWeight() << std::endl;
	}
	std::cout << std::endl;
*/

	return std::move(weight);
}


template <class InputInfo_, class OutputInfo_>
double Perceptron_Batch<InputInfo_, OutputInfo_>::Learn(std::vector<InputData> const& inputs)
{
	const uint div_size = inputs.size() / THREAD_NUM;
	std::vector< std::future< std::tuple< double, std::vector< std::vector<double >>> >> task;
	double mse = 0;

	auto LearnImpl = [](MLP_Impl& mlp, std::vector<InputData>::const_iterator begin, std::vector<InputData>::const_iterator end)
	{
		std::vector< std::vector<double>> result;
		double l_mse;
		bool first = true;

		while(true){
			mlp.ForwardPropagation(*begin);
			auto d_weight = mlp.BackPropagation(*begin);

			if (first){
				l_mse = mlp.out_layer_->SquareError(begin->Teacher());
				result = std::move(d_weight);
				first = false;
			}
			else{
				l_mse += mlp.out_layer_->SquareError(begin->Teacher());
				l_mse *= 0.5;

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
		auto& edge = mlp_.all_edges_;
		auto& d_weight = std::get<1>(r);

		mse += std::get<0>(r);

		for (uint l = 0; l < d_weight.size(); ++l){
			for (uint n = 0; n < d_weight[l].size(); ++n){
				edge[l][n]->Weight(edge[l][n]->Weight() + d_weight[l][n]);
			}
		}
	}

	for (auto& cpy : copy_mlp_) cpy.CopyWeight(mlp_);

	return mse / THREAD_NUM;
}

template <class InputInfo_, class OutputInfo_>
template<class Iter1>
C_OutputLayerPtr<OutputInfo_> Perceptron_Batch<InputInfo_, OutputInfo_>::Test(Iter1 input_first, Iter1 input_last)
{
	InputData input(input_first, input_last);
	mlp_.ForwardPropagation(input);
	return mlp_.out_layer_;
}

}
