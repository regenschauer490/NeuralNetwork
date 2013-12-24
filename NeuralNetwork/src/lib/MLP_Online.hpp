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
#include "Node.h"
#include "Edge.h"
#include "MLP_Base.hpp"

namespace signn{
	
template <class InputInfo_, class OutputInfo_>
class Perceptron_Online : public MLP_Base<InputInfo_, OutputInfo_>
{
	InputLayerPtr<InputInfo_> in_layer_;
	OutputLayerPtr<OutputInfo_> out_layer_;
	std::vector<LayerPtr> layers_;	//all layers
	
	//ParameterPack parameters_;
	double alpha_;
	double k_;

	//cache
	std::vector< std::vector<DEdgePtr>> all_edges_;
	
private:
	void MakeLink();

	void ForwardPropagation(InputData const& input) const;

	double BackPropagation(InputData const& input);

public:
	Perceptron_Online(std::vector<LayerPtr> hidden_layers) :
		in_layer_(InputLayerPtr<InputInfo_>(new InputLayer<InputInfo_>())), 
		out_layer_(OutputLayerPtr<OutputInfo_>(new typename LayerTypeMap<OutputInfo_::e_layertype>::layertype<OutputInfo_>())), alpha_(learning_rate)
	{
		layers_.push_back(in_layer_);
		for (auto& l : hidden_layers) layers_.push_back(l);
		layers_.push_back(out_layer_);
		MakeLink();
	}
	~Perceptron_Online(){};

	template<class Iter1, class Iter2>
	double Learn(Iter1 input_begin, Iter1 input_end, Iter2 teacher_begin, Iter2 teacher_end);

	template<class Iter1>
	double Learn(Iter1 input_begin, Iter1 input_end, typename OutputInfo_::type teacher);

	double Learn(InputData const& input);

	template<class Iter>
	C_OutputLayerPtr<OutputInfo_> Test(Iter input_begin, Iter input_end) const;

	void SaveParameter(std::wstring pass) const;

	void LoadParameter(std::wstring pass) const;

	void DebugWeight(std::vector<double> weight) const{ for (uint i=0; i<all_edges_.size(); ++i) all_edges_[i]->Weight(weight[i]); }
};


template <class InputInfo_, class OutputInfo_>
void Perceptron_Online<InputInfo_, OutputInfo_>::MakeLink()
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
void Perceptron_Online<InputInfo_, OutputInfo_>::ForwardPropagation(InputData const& input) const
{
	in_layer_->SetData(input.Input());
	for (auto& l : layers_){
		l->UpdateNodeScore();
	}
}


template <class InputInfo_, class OutputInfo_>
double Perceptron_Online<InputInfo_, OutputInfo_>::BackPropagation(InputData const& input)
{
	auto& teacher = input.Teacher();

	out_layer_->UpdateEdgeWeight(alpha_, teacher);
	for (int i = layers_.size() - 2; i >= 0; --i){
		layers_[i]->UpdateEdgeWeight(alpha_);
	}
	
/*
	std::cout << out_layer_->operator[](0)->Score() << ", t: " << teacher[0] << ", se:" << pow(out_layer_->operator[](0)->Score() - teacher[0], 2) << std::endl;
	uint i = 0;
	for (auto& e : all_edges_){
		std::cout << ++i << " : " << e->PreWeight() << " -> " << e->Weight() << ", d: " << e->Weight() - e->PreWeight() << std::endl;
	}
	std::cout << std::endl;
*/

	return pow((*out_layer_)[0]->Score() - teacher[0], 2);
}


template <class InputInfo_, class OutputInfo_>
double Perceptron_Online<InputInfo_, OutputInfo_>::Learn(InputData const& input)
{
	ForwardPropagation(input);
	return BackPropagation(input);

}

template <class InputInfo_, class OutputInfo_>
template<class Iter1, class Iter2>
double Perceptron_Online<InputInfo_, OutputInfo_>::Learn(Iter1 input_begin, Iter1 input_end, Iter2 teacher_begin, Iter2 teacher_end)
{
	InputData input(input_begin, input_end, teacher_begin, teacher_end);
	return Learn(input);
}

template <class InputInfo_, class OutputInfo_>
template<class Iter1>
double Perceptron_Online<InputInfo_, OutputInfo_>::Learn(Iter1 input_begin, Iter1 input_end, typename OutputInfo_::type teacher)
{
	InputData input(input_begin, input_end, teacher);
	return Learn(input);
}

template <class InputInfo_, class OutputInfo_>
template<class Iter>
C_OutputLayerPtr<OutputInfo_> Perceptron_Online<InputInfo_, OutputInfo_>::Test(Iter input_begin, Iter input_end) const
{
	InputData input(input_begin, input_end, 0);
	ForwardPropagation(input);
	return out_layer_->;
}

template <class InputInfo_, class OutputInfo_>
void Perceptron_Online<InputInfo_, OutputInfo_>::SaveParameter(std::wstring pass) const
{
	pass = File::DirpassTailModify(pass, true);

	for (uint l = 1; l < layers_.size(); ++l){
		File::RemakeFile(pass + L"weight" + std::to_wstring(l) + L".txt");
		for (auto const& node : *(layers_[l-1])){
			std::vector<double> weight;
			for (auto edge = node->out_begin(), end = node->out_end(); edge != end; ++edge){
				weight.push_back((*edge)->Weight());
			}
			File::SaveLineNum(weight, pass + L"weight" + std::to_wstring(l) + L".txt", File::WriteMode::append, ",");
		}
		++l;
	}
}

template <class InputInfo_, class OutputInfo_>
void Perceptron_Online<InputInfo_, OutputInfo_>::LoadParameter(std::wstring pass) const
{
	pass = File::DirpassTailModify(pass, true);
	for (uint l = 1; l < layers_.size(); ++l){
		std::vector<std::string> data;
		uint n = 0;
		File::ReadLine(data, pass + L"weight" + std::to_wstring(l) + L".txt");

		for (auto const& node : *(layers_[l-1])){
			uint e = 0;
			auto split = Split(data[n], ",");
			for (auto edge = node->out_begin(), end = node->out_end(); edge != end; ++edge){
				(*edge)->Weight(std::stod(split[e]));
				++e;
			}
			++n;
		}
		++l;
	}
}

}
