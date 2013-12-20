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

	double SquareError_(std::array<typename OutputInfo_::type, OutputInfo_::dim> const& teacher) const{
		return Metrics::SquareError(begin(), end(), teacher.begin());
	}

public:
	virtual ~OutputLayer(){};

	OutputLayerPtr<OutputInfo_> CloneInitInstance() const{ return std::static_pointer_cast<OutputLayer<OutputInfo_>>(CloneImpl()); }

	std::array<typename OutputInfo_::type, OutputInfo_::dim> GetScore() const{
		std::array<typename OutputInfo_::type, OutputInfo_::dim> score;
		for (uint i = 0; i < NodeNum(); ++i) score[i] = this->operator[](i)->Score();
		return score;
	}

	template<class Iter>
	double SquareError(Iter ans_vector_begin) const;

	template<class Dummy>
	double SquareError(typename OutputInfo_::type ans) const;
};

template <class OutputInfo_>
template<class Iter>
double OutputLayer<OutputInfo_>::SquareError(Iter ans_vector_begin) const{
	static_assert(OutputInfo_::dim > 1, "error in OutputLayer::SquareError() : need OutputLayer_::dim > 1");
	return Metrics::SquareError(begin(), end(), ans_vector_begin);
}

template <class OutputInfo_>
template<class Dummy>
double OutputLayer<OutputInfo_>::SquareError(typename OutputInfo_::type ans) const{
	static_assert(OutputInfo_::dim < 2, "error in OutputLayer::SquareError() : need OutputLayer_::dim < 2");
	return pow((*this)[0]->Score() - ans, 2);
}


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