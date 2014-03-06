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
#include "external/SigUtil/lib/file.hpp"

namespace signn{

template <class InputInfo_, class OutputInfo_>
class MLP_Impl
{
public:
	using DataFormat_ = DataFormat<InputInfo_, OutputInfo_>;
	using InputData_ = typename DataFormat_::InputData;
//	using OutputData_ = typename DataFormat_::OutputData;
	using OutputArrayType_ = std::array<typename OutputInfo_::output_type, OutputInfo_::dim>;
	using NodeData_ = double;
	using DEdge_ = DirectedEdge<NodeData_>;
	using DEdgePtr_ = DEdgePtr<NodeData_>;
	using Layer_ = Layer<NodeData_, DEdge_>;						//実際の中間レイヤーの型
	using LayerPtr_ = LayerPtr<NodeData_, DEdge_>;
	using InputLayer_ = InputLayer<InputInfo_>;							//実際の入力レイヤーの型
	using InputLayerPtr_ = InputLayerPtr<InputInfo_>;
	using OutputLayer_ = typename OutputInfo_::template layer_map<OutputInfo_>;	//実際の出力レイヤーの型
	using OutputLayerPtr_ = OutputLayerPtr<OutputInfo_>;

private:
	InputLayerPtr_ in_layer_;
	OutputLayerPtr_ out_layer_;
	std::vector<LayerPtr_> layers_;		//all layers

	const double alpha_;	//learning rate
	const double beta_;		//L2 regularization

	//cache
	std::vector< std::vector<DEdgePtr_>> all_edges_;

public:
	MLP_Impl(){}
	MLP_Impl(double alpha, double beta, std::vector<LayerPtr_> hidden_layers) :
		in_layer_(InputLayerPtr_(new InputLayer_())),
		out_layer_(OutputLayerPtr_(new OutputLayer_())),
		alpha_(alpha), beta_(beta)
	{
		layers_.push_back(in_layer_);
		for (auto& l : hidden_layers) layers_.push_back(l->CloneInitInstance());
		layers_.push_back(out_layer_);
		MakeLink();
	}

	static LayerPtr_ MakeMidLayer(uint node_num){ return Layer_::MakeInstance(node_num); }


	void CopyWeight(MLP_Impl const& src){
		for (uint l = 0; l < all_edges_.size(); ++l){
			for (uint e = 0; e < all_edges_[l].size(); ++e) all_edges_[l][e]->Weight(src.all_edges_[l][e]->Weight());
		}
	}

	void RenewEdgeWeight(std::vector<std::vector<double>> const& delta){
		for (uint l = 0; l < delta.size(); ++l){
			layers_[l+1]->RenewEdgeWeight(delta[l], beta_);
		}
	}

	void MakeLink();
	template <class Matrix>
	void MakeLink(Matrix const& connection);	//層間の接続を表現(row:前の層, col:後ろの層)

	void ForwardPropagation(InputData_ const& input) const;

	void BackPropagation(InputData_ const& input);

	auto BackPropagation_Delay(InputData_ const& input) ->std::vector< std::vector<double>>;


	auto OutputScore() const->OutputArrayType_{ return out_layer_->GetScore(); }

	double MeanSquareError(OutputArrayType_ const& teacher) const{ return out_layer_->MeanSquareError(teacher); };


	void SaveParameter(std::wstring pass) const;

	void LoadParameter(std::wstring pass);

};


template <class InputInfo_, class OutputInfo_>
void MLP_Impl<InputInfo_, OutputInfo_>::MakeLink()
{
	//make links between nodes
	auto MakeLink = [&](LayerPtr_ layer_prev, LayerPtr_ layer_next, std::vector<DEdgePtr_>& cache){
		for (auto& departure : *layer_prev){
			for (auto& arrival : *layer_next){
				auto edge = std::make_shared<DEdge_>();
				Connect(departure, arrival, edge);
				cache.push_back(edge);
			}
		}
	};

	for (uint i = 1; i < layers_.size(); ++i){
		all_edges_.push_back(std::vector<DEdgePtr_>());
		MakeLink(layers_[i - 1], layers_[i], all_edges_.back());
	}
}

template <class InputInfo_, class OutputInfo_>
template <class Matrix>
void MLP_Impl<InputInfo_, OutputInfo_>::MakeLink(Matrix const& connection)
{
	for (uint i = 1; i < layers_.size(); ++i){
		all_edges_.push_back(std::vector<DEdgePtr_>());
		signn::MakeLink(layers_[i - 1], layers_[i], matrix);

		for(auto const& n : *layers_[i]){
			for(auto e = n->in_begin(), end = n->in_end(); e != end; ++e) all_edges_.push_back(*e);
		}
	}
}


template <class InputInfo_, class OutputInfo_>
void MLP_Impl<InputInfo_, OutputInfo_>::ForwardPropagation(InputData_ const& input) const
{
	auto* tp = const_cast<MLP_Impl*>(this);		//要検討

	tp->in_layer_->SetData(input.Input());
	for (auto& l : tp->layers_){
		l->UpdateNodeScore();
	}
}


template <class InputInfo_, class OutputInfo_>
void MLP_Impl<InputInfo_, OutputInfo_>::BackPropagation(InputData_ const& input)
{
	assert(!input.IsTestData());

	out_layer_->UpdateEdgeWeight(alpha_, beta_, input.Teacher());
	for (int i = layers_.size() - 2; i > 0; --i){
		layers_[i]->UpdateEdgeWeight(alpha_, beta_);
	}
}

template <class InputInfo_, class OutputInfo_>
std::vector< std::vector<double>> MLP_Impl<InputInfo_, OutputInfo_>::BackPropagation_Delay(InputData_ const& input)
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
	pass = sig::DirpassTailModify(pass, true);

	for (uint l = 1; l < layers_.size(); ++l){
		sig::FileClear(pass + L"weight" + std::to_wstring(l) + L".txt");
		for (auto const& node : *(layers_[l - 1])){
			std::vector<double> weight;
			for (auto edge = node->out_begin(), end = node->out_end(); edge != end; ++edge){
				weight.push_back((*edge)->Weight());
			}
			sig::SaveNum(weight, pass + L"weight" + std::to_wstring(l) + L".txt", sig::WriteMode::append, ",");
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
		sig::ReadLine(data, pass + L"weight" + std::to_wstring(l) + L".txt");

		for (auto const& node : *(layers_[l - 1])){
			uint e = 0;
			auto split = sig::Split(data[n], ",");
			for (auto edge = node->out_begin(), end = node->out_end(); edge != end; ++edge){
				(*edge)->Weight(std::stod(split[e]));
				++e;
			}
			++n;
		}
	}
}

}
#endif