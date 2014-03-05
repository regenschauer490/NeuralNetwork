/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_LAYER_OUTPUT_H
#define SIG_NN_LAYER_OUTPUT_H

#include "layer.hpp"

namespace signn{

template <class OutputInfo_>
class OutputLayer : public Layer<double, DirectedEdge<double>>
{
public:
	using NodeData_ = double;
	using OutputData_ = typename OutputInfo_::output_type;
	using Edge_ = DirectedEdge<NodeData_>;
	using LayerPtr_ = LayerPtr<NodeData_, Edge_>;

private:
	SIG_FRIEND_WITH_LAYER;

private:
	virtual LayerPtr_ CloneImpl() const override = 0;

protected:
	OutputLayer() : Layer(OutputInfo_::dim){};

	//forward propagation
	virtual void UpdateNodeScore() override = 0;

	//back propagation for online
	virtual void UpdateEdgeWeight(double alpha, double beta, std::array<NodeData_, OutputInfo_::dim> teacher_signals) = 0;

	//back propagation for batch (not renew weight)
	virtual std::vector<NodeData_> CalcEdgeWeight(double alpha, std::array<NodeData_, OutputInfo_::dim> teacher_signals) = 0;

	//convert node-value into output-value
	virtual OutputData_ OutputScore(NodeData_ raw_score) const = 0;

public:
	virtual ~OutputLayer(){};

	OutputLayerPtr<OutputInfo_> CloneInitInstance() const{ return std::static_pointer_cast<OutputLayer<OutputInfo_>>(CloneImpl()); }

	auto GetScore() const->std::array<OutputData_, OutputInfo_::dim>{
		std::array<OutputData_, dim_> score;
		for (uint i = 0; i < OutputInfo_::dim; ++i) score[i] = OutputScore(this->operator[](i)->Score());
		return score;
	}

	template<class Container>
	double MeanSquareError(Container const& teacher) const;

/*
	template<class Iter, typename = decltype(*std::declval<Iter&>(), void(), ++std::declval<Iter&>(), void())>
	double SquareError(Iter ans_vector_begin) const;

	double SquareError(typename OutputInfo_::type ans) const{
		static_assert(OutputInfo_::dim < 2, "error in OutputLayer::SquareError() : need OutputLayer_::dim < 2");
		return pow((*this)[0]->Score() - ans, 2);
	}
*/
};


template <class OutputInfo_>
template<class Container>
double OutputLayer<OutputInfo_>::MeanSquareError(Container const& teacher) const{
	return std::inner_product(begin(), end(), teacher.begin(), 0.0, std::plus<double>(), [](NodePtr_ const& v1, double v2){ return pow(v1->Score() - v2, 2); }) / dim_;
}

#define PP_UpdateNodeScore(ACTIVATE_FUNC)\
	void UpdateNodeScore() override{\
		for(auto& node : *this){\
			node->UpdateScore<ACTIVATE_FUNC>(); \
		}\
	}

#define PP_CalcEdgeWeight(ACTIVATE_FUNC)\
	std::vector<NodeData_> CalcEdgeWeight(double alpha, std::array<NodeData_, OutputInfo_::dim> teacher_signals) override{\
		std::vector<NodeData_> new_weight; \
		for (uint n = 0, node_num = OutputInfo_::dim; n < node_num; ++n){\
			auto& node = (*this)[n];\
			auto const error = teacher_signals[n] - node->Score();\
			\
			for (auto edge = node->in_begin(), end = node->in_end(); edge != end; ++edge){\
				new_weight.push_back( (*edge)->CalcDeltaWeight<ACTIVATE_FUNC>(alpha, error) ); \
			}\
		}\
		return std::move(new_weight);\
	}

#define PP_UpdateEdgeWeight(ACTIVATE_FUNC)\
	void UpdateEdgeWeight(double alpha, double beta, std::array<NodeData_, OutputInfo_::dim> teacher_signals) override{\
		for (uint n = 0, node_num = OutputInfo_::dim; n < node_num; ++n){\
			auto& node = (*this)[n]; \
			auto const error = teacher_signals[n] - node->Score(); \
			\
			for (auto edge = node->in_begin(), end = node->in_end(); edge != end; ++edge){\
				(*edge)->UpdateWeight<ACTIVATE_FUNC>(alpha, beta, error); \
			}\
		}\
	}


template <class OutputInfo_>
class RegressionLayer : public OutputLayer<OutputInfo_>
{
	SIG_FRIEND_WITH_LAYER;

private:
	virtual LayerPtr_ CloneImpl() const override{ return std::shared_ptr<Layer>(new RegressionLayer<OutputInfo_>()); }

	PP_UpdateNodeScore(Identity<NodeData_>);

	PP_CalcEdgeWeight(Identity<NodeData_>);
	PP_UpdateEdgeWeight(Identity<NodeData_>);

	OutputData_ OutputScore(NodeData_ raw_score) const override{ return raw_score; }

public:
	virtual ~RegressionLayer(){};

	OutputLayerPtr<OutputInfo_> CloneInitInstance() const{ return std::static_pointer_cast<RegressionLayer<OutputInfo_>>(CloneImpl()); }
};


template <class OutputInfo_>
class BinaryClassificationLayer : public OutputLayer<OutputInfo_>
{
	SIG_FRIEND_WITH_LAYER;

private:
	virtual LayerPtr_ CloneImpl() const override{ return std::shared_ptr<Layer>(new BinaryClassificationLayer<OutputInfo_>()); }

	PP_UpdateNodeScore(Sigmoid);

	PP_CalcEdgeWeight(Sigmoid);
	PP_UpdateEdgeWeight(Sigmoid);

	OutputData_ OutputScore(NodeData_ raw_score) const override{
		static_assert(std::is_same<OutputData_, bool>::value);
		return raw_score < 0.5 ? false : true; 
	}

public:
	virtual ~BinaryClassificationLayer(){};

	OutputLayerPtr<OutputInfo_> CloneInitInstance() const{ return std::static_pointer_cast<BinaryClassificationLayer<OutputInfo_>>(CloneImpl()); }
};


template <class OutputInfo_>
class MultiClassClassificationLayer : public OutputLayer<OutputInfo_>
{
	SIG_FRIEND_WITH_LAYER;

private:
	virtual LayerPtr_ CloneImpl() const override{ return std::shared_ptr<Layer>(new MultiClassClassificationLayer<OutputInfo_>()); }

	void UpdateNodeScore() override{
		std::vector<double> exp_raw_score;
		for (auto& node : *this){
			exp_raw_score.push_back( std::exp(node->AccumulateRawScore()) );
		}
		auto exp_sum = std::accumulate(exp_raw_score.begin(), exp_raw_score.end(), 0.0);

		for (uint n = 0, num = OutputInfo_::dim; n < num; ++n){
			(*this)[n]->Score(Softmax::f(exp_raw_score[n], exp_sum));
		}
	}

	std::vector<NodeData_> CalcEdgeWeight(double alpha, std::array<NodeData_, OutputInfo_::dim> teacher_signals) override{
		std::vector<NodeData_> new_weight;

		for (uint n = 0, node_num = OutputInfo_::dim; n < node_num; ++n){
			auto& node = (*this)[n]; 
			auto const error = teacher_signals[n] - node->Score(); 

			for (auto edge = node->in_begin(), end = node->in_end(); edge != end; ++edge){
				new_weight.push_back((*edge)->CalcDeltaWeight<Softmax> (alpha, error));
			}
		}
		return std::move(new_weight);
	}

	void UpdateEdgeWeight(double alpha, double beta, std::array<NodeData_, OutputInfo_::dim> teacher_signals) override{
		for (uint n = 0, node_num = NodeNum(); n < node_num; ++n){
			auto& node = (*this)[n]; 
			auto const error = teacher_signals[n] - node->Score(); 

			for (auto edge = node->in_begin(), end = node->in_end(); edge != end; ++edge){
				(*edge)->UpdateWeight<Softmax>(alpha, beta, error);
			}
		}
	}

	OutputData_ OutputScore(NodeData_ raw_score) const override{ return raw_score < 1.0 / OutputInfo_::dim ? false : true; }

public:
	virtual ~MultiClassClassificationLayer(){};

	OutputLayerPtr<OutputInfo_> CloneInitInstance() const{ return std::static_pointer_cast<MultiClassClassificationLayer<OutputInfo_>>(CloneImpl()); }
};

}
#endif