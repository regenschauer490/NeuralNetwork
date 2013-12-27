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

namespace signn
{

//parameter constant
const bool DEBUG_MODE = false;
const uint THREAD_NUM = 3;
const double threshold_theta = 0.5;
const double learning_rate = 0.01;

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
	template<class OInfo> using layertype = RegressionLayer<OInfo>;
	typedef double type;
};

template <>
struct LayerTypeMap<OutputLayerType::BinaryClassification>{
	template<class OInfo> using layertype = BinaryClassificationLayer<OInfo>;
	typedef int type;	//todo: test bool
};

template <>
struct LayerTypeMap<OutputLayerType::MultiClassClassification>{
	template<class OInfo> using layertype = MultiClassClassificationLayer<OInfo>;
	typedef int type;
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
	template <class InputInfo, class OutputInfo> friend class Perceptron_Batch;\
	template <class InputInfo, class OutputInfo> friend class Perceptron_Online;



//activation function
template <class T>
struct Identity
{
	static T f(T x) { return x; }
	static T df(T x) { return 1; }
};

template <class T>
struct Sigmoid
{
	static T f(T x) { return 1.0 / (1.0 + std::exp(-x)); }
	static T df(T x) { auto f_x = f(x); return f_x * (1.0 - f_x); }
};

template <class T>
struct Softmax
{
	static T f(T exp_x, T exp_sum){ return exp_x / exp_sum; }
	//static T df(T exp_x, T exp_sum) { return (exp_x * (sum - exp_x)) / std::pow(exp_sum, 2); }
	static T df(T x) { return 1; }
};



}