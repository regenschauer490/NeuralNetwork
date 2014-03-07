/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_LAYER_INPUT_H
#define SIG_NN_LAYER_INPUT_H

#include "layer.hpp"

namespace signn{

template <class InputInfo_>
class InputLayer : public Layer<double, DirectedEdge<double>>
{
public:
	using NodeData_ = double;
	using InputData_ = typename InputInfo_::type;
	using Edge_ = DirectedEdge<NodeData_>;
	using LayerPtr_ = LayerPtr<NodeData_, Edge_>;

	SIG_FRIEND_WITH_LAYER

private:
	InputLayer() : Layer(InputInfo_::node_num){};

	void UpdateNodeScore() override{}

	//implicit conversion from InputData_ to NodeData_(double)
	template <class = typename std::enable_if<std::is_same<typename std::common_type<NodeData_, InputData_>::type, NodeData_>::value>::type>
	void SetData(std::array<InputData_, InputInfo_::node_num> const& input){
		for (size_t i = 0, end = InputInfo_::node_num; i < end; ++i) (*this)[i]->Score(input[i]);
	}

	//explicit conversion from InputData_ to NodeData_(double) by convert function
	template <class = typename std::enable_if<!std::is_same<typename std::common_type<NodeData_, InputData_>::type, NodeData_>::value>::type>
	void SetData(std::array<InputData_, InputInfo_::node_num> const& input, std::function<NodeData_(InputData_)> const& convert){
		for (size_t i = 0, end = InputInfo_::node_num; i < end; ++i) (*this)[i]->Score(convert(input[i]));
	}

public:
	~InputLayer(){};
};

}
#endif