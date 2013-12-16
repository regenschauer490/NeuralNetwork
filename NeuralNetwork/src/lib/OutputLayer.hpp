#pragma once
#include "Layer.h"

namespace signn{

template <class OutputInfo_>
class OutputLayer : public Layer
{
	FRIEND_WITH_LAYER;

private:
	virtual LayerPtr CloneImpl() const override = 0;

protected:
	OutputLayer() : Layer(OutputInfo_::dim){};

	virtual void UpdateNodeScore() override = 0;

	virtual void UpdateEdgeWeight(double alpha, std::array<typename OutputInfo_::type, OutputInfo_::dim> teacher_signals) = 0;

	virtual std::vector<double> CalcEdgeWeight(double alpha, std::array<typename OutputInfo_::type, OutputInfo_::dim> teacher_signals) = 0;

	double SquareError(std::array<typename OutputInfo_::type, OutputInfo_::dim> const& teacher){
		return std::inner_product(begin(), end(), teacher.begin(), 0.0, std::plus<double>(), [](NodePtr& n, typename OutputInfo_::type v){ return pow(n->Score() - v, 2); });
	}

public:
	virtual ~OutputLayer(){};

	OutputLayerPtr<OutputInfo_> CloneInitInstance() const{ return std::static_pointer_cast<OutputLayer<OutputInfo_>>(CloneImpl()); }

	std::array<typename OutputInfo_::type, OutputInfo_::dim> GetScore() const{
		std::array<typename OutputInfo_::type, OutputInfo_::dim> score;
		for (uint i = 0; i < NodeNum(); ++i) score[i] = this->operator[](i)->Score();
		return score;
	}
};


#define PP_UpdateNodeScore(ACTIVATE_FUNC)\
	void UpdateNodeScore() override{\
		for(auto& node : *this){\
			node->UpdateScore<ACTIVATE_FUNC>(); \
		}\
	}

#define PP_CalcEdgeWeight(ACTIVATE_FUNC)\
	std::vector<double> CalcEdgeWeight(double alpha, std::array<typename OutputInfo_::type, OutputInfo_::dim> teacher_signals) override{\
		std::vector<double> new_weight;\
		for (uint n = 0, node_num = NodeNum(); n < node_num; ++n){\
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
	void UpdateEdgeWeight(double alpha, std::array<typename OutputInfo_::type, OutputInfo_::dim> teacher_signals) override{\
		for (uint n = 0, node_num = NodeNum(); n < node_num; ++n){\
			auto& node = (*this)[n]; \
			auto const error = teacher_signals[n] - node->Score(); \
			\
			for (auto edge = node->in_begin(), end = node->in_end(); edge != end; ++edge){\
				(*edge)->UpdateWeight<ACTIVATE_FUNC>(alpha, error); \
			}\
		}\
	}

template <class OutputInfo_>
class RegressionLayer : public OutputLayer<OutputInfo_>
{
	FRIEND_WITH_LAYER;

private:
	virtual LayerPtr CloneImpl() const override{ return std::shared_ptr<Layer>(new RegressionLayer<OutputInfo_>()); }

	PP_UpdateNodeScore(Identity<typename OutputInfo_::type>);

	PP_CalcEdgeWeight(Identity<typename OutputInfo_::type>);
	PP_UpdateEdgeWeight(Identity<typename OutputInfo_::type>);

public:
	virtual ~RegressionLayer(){};

	OutputLayerPtr<OutputInfo_> CloneInitInstance() const{ return std::static_pointer_cast<RegressionLayer<OutputInfo_>>(CloneImpl()); }
};


template <class OutputInfo_>
class BinaryClassificationLayer : public OutputLayer<OutputInfo_>
{
	FRIEND_WITH_LAYER;

private:
	virtual LayerPtr CloneImpl() const override{ return std::shared_ptr<Layer>(new BinaryClassificationLayer<OutputInfo_>()); }

	PP_UpdateNodeScore(Sigmoid<typename OutputInfo_::type>);

	PP_CalcEdgeWeight(Sigmoid<typename OutputInfo_::type>);
	PP_UpdateEdgeWeight(Sigmoid<typename OutputInfo_::type>);

public:
	virtual ~BinaryClassificationLayer(){};

	OutputLayerPtr<OutputInfo_> CloneInitInstance() const{ return std::static_pointer_cast<BinaryClassificationLayer<OutputInfo_>>(CloneImpl()); }
};


template <class OutputInfo_>
class MultiClassClassificationLayer : public OutputLayer<OutputInfo_>
{
	FRIEND_WITH_LAYER;

private:
	virtual LayerPtr CloneImpl() const override{ return std::shared_ptr<Layer>(new MultiClassClassificationLayer<OutputInfo_>()); }

	void UpdateNodeScore() override{
		std::vector<double> exp_raw_score;
		for (auto& node : *this){
			exp_raw_score.push_back( std::exp(node->AccumulateRawScore()) );
		}
		auto exp_sum = std::accumulate(exp_raw_score.begin(), exp_raw_score.end(), 0.0);

		for (uint n = 0, num = NodeNum(); n < num; ++n){
			(*this)[n]->UpdateScore(Softmax<typename OutputInfo_::type>::f(exp_raw_score[n], exp_sum));
		}
	}

	std::vector<double> CalcEdgeWeight(double alpha, std::array<typename OutputInfo_::type, OutputInfo_::dim> teacher_signals) override{
		std::vector<double> new_weight; 

		for (uint n = 0, node_num = NodeNum(); n < node_num; ++n){
			auto& node = (*this)[n]; 
			auto const error = teacher_signals[n] - node->Score(); 

			for (auto edge = node->in_begin(), end = node->in_end(); edge != end; ++edge){
				new_weight.push_back((*edge)->CalcDeltaWeight< Softmax<typename OutputInfo_::type>> (alpha, error));
			}
		}
		return std::move(new_weight);
	}

	void UpdateEdgeWeight(double alpha, std::array<typename OutputInfo_::type, OutputInfo_::dim> teacher_signals) override{
		for (uint n = 0, node_num = NodeNum(); n < node_num; ++n){
			auto& node = (*this)[n]; 
			auto const error = teacher_signals[n] - node->Score(); 

			for (auto edge = node->in_begin(), end = node->in_end(); edge != end; ++edge){
				(*edge)->UpdateWeight<Softmax<typename OutputInfo_::type>>(alpha, error);
			}
		}
	}

public:
	virtual ~MultiClassClassificationLayer(){};

	OutputLayerPtr<OutputInfo_> CloneInitInstance() const{ return std::static_pointer_cast<MultiClassClassificationLayer<OutputInfo_>>(CloneImpl()); }
};


}