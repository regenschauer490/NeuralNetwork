/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_MLP_IMPL_H
#define SIG_NN_MLP_IMPL_H

#include "layer_input.hpp"
#include "layer_output.hpp"
#include "data_format.hpp"

namespace signn{

template <class InputInfo_, class OutputInfo_>
class MLP_Impl
{
	typedef DataFormat<InputInfo_, OutputInfo_> DataFormat;
	typedef typename DataFormat::InputData InputData;
	typedef typename DataFormat::OutputData OutputData;

	InputLayerPtr<InputInfo_> in_layer_;
	OutputLayerPtr<OutputInfo_> out_layer_;
	std::vector<LayerPtr> layers_;		//all layers

	const double alpha_;	//learning rate
	const double beta_;		//L2 regularization

	//cache
	std::vector< std::vector<DEdgePtr>> all_edges_;

public:
	MLP_Impl(){}
	explicit MLP_Impl(double alpha, double beta, std::vector<LayerPtr> hidden_layers) :
		in_layer_(InputLayerPtr<InputInfo_>(new InputLayer<InputInfo_>())),
		out_layer_(OutputLayerPtr<OutputInfo_>(new typename LayerTypeMap<OutputInfo_::e_layertype>::layertype<OutputInfo_>())),
		alpha_(alpha), beta_(beta)
	{
		layers_.push_back(in_layer_);
		for (auto& l : hidden_layers) layers_.push_back(l->CloneInitInstance());
		layers_.push_back(out_layer_);
		MakeLink();
	}

	void CopyWeight(MLP_Impl const& src){
		for (uint l = 0; l < all_edges_.size(); ++l){
			for (uint e = 0; e < all_edges_[l].size(); ++e) all_edges_[l][e]->Weight(src.all_edges_[l][e]->Weight());
		}
	}

	void RenewEdgeWeight(std::vector < std::vector < double >> const& delta){
		for (uint l = 0; l < delta.size(); ++l){
			layers_[l+1]->RenewEdgeWeight(delta[l], beta_);
		}
	}

	void MakeLink();


	void ForwardPropagation(InputData const& input) const;

	void BackPropagation(InputData const& input);

	std::vector< std::vector<double>> BackPropagation_Delay(InputData const& input);


	typename OutputData::OutputDataArray OutputScore() const{ return out_layer_->GetScore(); }

	double MeanSquareError(typename InputData::TeacherDataArray const& teacher) const{ return out_layer_->MeanSquareError(teacher); };


	void SaveParameter(std::wstring pass) const;

	void LoadParameter(std::wstring pass);

};


template <class InputInfo_, class OutputInfo_>
void MLP_Impl<InputInfo_, OutputInfo_>::MakeLink()
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
void MLP_Impl<InputInfo_, OutputInfo_>::ForwardPropagation(InputData const& input) const
{
	auto* tp = const_cast<MLP_Impl*>(this);		//要検討

	tp->in_layer_->SetData(input.Input());
	for (auto& l : tp->layers_){
		l->UpdateNodeScore();
	}
}


template <class InputInfo_, class OutputInfo_>
void MLP_Impl<InputInfo_, OutputInfo_>::BackPropagation(InputData const& input)
{
	assert(!input.IsTestData());

	out_layer_->UpdateEdgeWeight(alpha_, beta_, input.Teacher());
	for (int i = layers_.size() - 2; i > 0; --i){
		layers_[i]->UpdateEdgeWeight(alpha_, beta_);
	}
}

template <class InputInfo_, class OutputInfo_>
std::vector< std::vector<double>> MLP_Impl<InputInfo_, OutputInfo_>::BackPropagation_Delay(InputData const& input)
{
	assert(!input.IsTestData());

	const uint lsize = layers_.size() - 1;
	std::vector< std::vector<double> > weight(lsize);

	weight[lsize - 1] = out_layer_->CalcEdgeWeight(alpha_, input.Teacher());
	for (int i = lsize - 1; i > 0; --i){
		weight[i - 1] = layers_[i]->CalcEdgeWeight(alpha_);
	}

	return std::move(weight);
}

template <class InputInfo_, class OutputInfo_>
void MLP_Impl<InputInfo_, OutputInfo_>::SaveParameter(std::wstring pass) const
{
	pass = File::DirpassTailModify(pass, true);

	for (uint l = 1; l < layers_.size(); ++l){
		File::RemakeFile(pass + L"weight" + std::to_wstring(l) + L".txt");
		for (auto const& node : *(layers_[l - 1])){
			std::vector<double> weight;
			for (auto edge = node->out_begin(), end = node->out_end(); edge != end; ++edge){
				weight.push_back((*edge)->Weight());
			}
			File::SaveLineNum(weight, pass + L"weight" + std::to_wstring(l) + L".txt", File::WriteMode::append, ",");
		}
	}
}

template <class InputInfo_, class OutputInfo_>
void MLP_Impl<InputInfo_, OutputInfo_>::LoadParameter(std::wstring pass)
{
	pass = File::DirpassTailModify(pass, true);
	for (uint l = 1; l < layers_.size(); ++l){
		std::vector<std::string> data;
		uint n = 0;
		File::ReadLine(data, pass + L"weight" + std::to_wstring(l) + L".txt");

		for (auto const& node : *(layers_[l - 1])){
			uint e = 0;
			auto split = Split(data[n], ",");
			for (auto edge = node->out_begin(), end = node->out_end(); edge != end; ++edge){
				(*edge)->Weight(std::stod(split[e]));
				++e;
			}
			++n;
		}
	}
}


}