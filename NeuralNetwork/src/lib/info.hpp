﻿/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_INFO_H
#define SIG_NN_INFO_H

#include <algorithm>
#include <numeric>
#include <assert.h>
#include <vector>
#include <array>
#include <memory>
#include <functional>
#include <random>
#include <future>
#include "util.hpp"

namespace signn{

using sig::uint;

//parameter constant
const bool DEBUG_MODE = true;
const uint THREAD_NUM = 3;
const double threshold_theta = 0.5;
const double learning_rate_sample = 0.01;
const double L2__regularization_sample = 0.9999;

//parameter selection
enum class OutputLayerType { Regression, BinaryClassification, MultiClassClassification, MultiLabelClassification };


//forward declaration
template <class OutputInfo_>
class RegressionLayer;

template <class OutputInfo_>
class BinaryClassificationLayer;

template <class OutputInfo_>
class MultiClassClassificationLayer;

//type map (enum class -> class)
template <OutputLayerType From>
struct LayerTypeMap{};

template <>
struct LayerTypeMap<OutputLayerType::Regression>{
	template<class OutInfo> using layertype = RegressionLayer<OutInfo>;
	typedef double type;
};

template <>
struct LayerTypeMap<OutputLayerType::BinaryClassification>{
	template<class OInfo> using layertype = BinaryClassificationLayer<OInfo>;
	typedef bool type;
};

template <>
struct LayerTypeMap<OutputLayerType::MultiClassClassification>{
	template<class OInfo> using layertype = MultiClassClassificationLayer<OInfo>;
	typedef bool type;
};


//meta info
template <class Type, size_t Dim>
struct InputInfo{
	typedef Type type;
	static const size_t dim = Dim;
};

template <OutputLayerType LayerType, size_t Dim>
struct OutputInfo{
	typedef typename LayerTypeMap<LayerType>::type type;
	static const OutputLayerType e_layertype = LayerType;
	static const size_t dim = Dim;
};


//forward declaration
class Node;
typedef std::shared_ptr<Node> NodePtr;
typedef std::shared_ptr<Node const> C_NodePtr;
typedef std::weak_ptr<Node> NodeWPtr;
typedef std::weak_ptr<Node const> C_NodeWPtr;

class DirectedEdge;
typedef std::shared_ptr<DirectedEdge> DEdgePtr;

class Layer;
typedef std::shared_ptr<Layer> LayerPtr;
typedef std::shared_ptr<Layer const> C_LayerPtr;

template <class InputInfo_>
class InputLayer;
template <class InputInfo_>
using InputLayerPtr = std::shared_ptr<InputLayer<InputInfo_>>;

template <class OutputInfo_>
class OutputLayer;
template <class OutputInfo_>
using OutputLayerPtr = std::shared_ptr<OutputLayer<OutputInfo_>>;
template <class OutputInfo_>
using C_OutputLayerPtr = std::shared_ptr<OutputLayer<OutputInfo_> const>;


#define FRIEND_WITH_LAYER\
	template <class InputInfo, class OutputInfo> friend class MLP_Impl;\



//activation function
template <class T>
struct Identity
{
	static T f(T x) { return x; }
	static T df(T x) { return 1; }
};

struct Sigmoid
{
	static double f(double x) { return 1.0 / (1.0 + std::exp(-x)); }
	static double df(double x) { auto f_x = f(x); return f_x * (1.0 - f_x); }
};


struct Softmax
{
	static double f(double exp_x, double exp_sum){ return exp_x / exp_sum; }
	//static double df(double exp_x, double exp_sum) { return (exp_x * (sum - exp_x)) / std::pow(exp_sum, 2); }
	static double df(double x) { return 1; }
};

}
#endif