/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_INFO_H
#define SIG_NN_INFO_H

#include "external/SigUtil/lib/sigutil.hpp"

namespace signn{

using sig::uint;

//option
const bool DEBUG_MODE = true;
const uint THREAD_NUM = 3;

//parameters
const double learning_rate_sample = 0.01;
const double L2__regularization_sample = 0.9999;
const double threshold_theta = 0.5;
const double SIG_DEFAULT_EDGE_WEIGHT = 0.5;		//default edge weight(must not set same weight on MLP, AutoEncoder and so on)

const double som_learning_rate = 0.5;

//forward declaration
template <class OutputInfo_>
class RegressionLayer;

template <class OutputInfo_>
class BinaryClassificationLayer;

template <class OutputInfo_>
class MultiClassClassificationLayer;

template <size_t RefVecDim>
class SOMLayer;


//出力が実数値で、回帰を行いたい場合の設定
struct RegressionLayerInfo{
	template<class OutInfo> using layer_type = RegressionLayer<OutInfo>;
	using output_type = double;
	static constexpr size_t dim = 1;
};

//出力がバイナリ(bool)で、二値分類を行いたい場合の設定
struct BinaryClassifyLayerInfo{
	template<class OInfo> using layer_type = BinaryClassificationLayer<OInfo>;
	using output_type = bool;
	static constexpr size_t dim = 1;
};

//出力がバイナリ(bool)で、多値分類を行いたい場合の設定
template <size_t NodeNum>
struct MultiClassClassifyLayerInfo{
	template<class OInfo> using layer_type = MultiClassClassificationLayer<OInfo>;
	using output_type = bool;
	static constexpr size_t dim = NodeNum;
};

//出力がバイナリ(bool)で、マルチラベル分類を行いたい場合の設定
template <size_t NodeNum>
struct MultiLabelClassifyLayerInfo{
	template<class OInfo> using layer_type = BinaryClassificationLayer<OInfo>;
	using output_type = bool;
	static constexpr size_t dim = NodeNum;
};

//自己組織化マップ用の設定
template <size_t SideNodeNum>
struct SOMLayerInfo{
	template<class OInfo> using layer_type = SOMLayer<SideNodeNum>;
	using output_type = double;
	static constexpr size_t dim = SideNodeNum * SideNodeNum;
};

//出力レイヤーの任意設定
template < template <class> class OutputLayer, class ValueType, size_t NodeNum>
struct LayerInfo{
	template<class OInfo> using layer_type = OutputLayer<OInfo>;
	using output_type = ValueType;
	static constexpr size_t dim = NodeNum;
};



//input meta info
//Type: input data type
//dim: the number of nodes which this layer has
template <class Type, size_t NodeNum>
struct InputInfo{
	using type = Type;
	static constexpr size_t dim = NodeNum;
};

//output meta info
template <class LayerInfo_>
struct OutputInfo{
	using output_type = typename LayerInfo_::output_type;
	template<class OI> using layer_type = typename LayerInfo_::template layer_type<OI>;
	static constexpr size_t dim = LayerInfo_::dim;
};


//forward declaration
template <class T, class E> class Node;
template <class T, class E> using NodePtr = std::shared_ptr<Node<T,E>>;
template <class T, class E> using C_NodePtr = std::shared_ptr<Node<T,E> const>;
template <class T, class E> using NodeWPtr = std::weak_ptr<Node<T,E>>;
template <class T, class E> using C_NodeWPtr = std::weak_ptr<Node<T,E> const> ;

template <class ND> class DirectedEdge;
template <class ND> using DEdgePtr = std::shared_ptr<DirectedEdge<ND>> ;

template <class ND> class UndirectedEdge;
template <class ND> using UDEdgePtr = std::shared_ptr<UndirectedEdge<ND>>;


template <class ND, class E> class Layer;
template <class ND, class E> using LayerPtr = std::shared_ptr<Layer<ND,E>>;
template <class ND, class E> using C_LayerPtr = std::shared_ptr<Layer<ND,E> const>;

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

template <size_t RefVecDim>
using SOMLayerPtr = std::shared_ptr<SOMLayer<RefVecDim>>;
template <size_t RefVecDim>
using C_SOMLayerPtr = std::shared_ptr<const SOMLayer<RefVecDim>>;

#define SIG_FRIEND_WITH_LAYER\
	template <class InputInfo_, class OutputInfo_> friend class MLP_Impl;

#define SIG_FRIEND_WITH_SOMLAYER\
	template <class InputInfo_> friend class SOM_Online : public DataFormat<InputInfo_, OutputInfo_>;

#define SIG_FRIEND_WITH_NODE_AND_EDGE\
	template <class T> friend void Connect(NodePtr<T, DirectedEdge<T>> const&, NodePtr<T, DirectedEdge<T>> const&, DEdgePtr<T> const&);\
	template <class T> friend void Disconnect(C_NodePtr<T, DirectedEdge<T>> const&, C_NodePtr<T, DirectedEdge<T>> const&);


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