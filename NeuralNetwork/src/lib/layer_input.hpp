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
class InputLayer : public Layer
{
	FRIEND_WITH_LAYER

private:
	InputLayer() : Layer(InputInfo_::dim){};

	void UpdateNodeScore() override{}

	void SetData(std::array<typename InputInfo_::type, InputInfo_::dim> const& input){
		for (size_t i = 0, end = NodeNum(); i < end; ++i) (*this)[i]->Score(input[i]);
	}

public:
	~InputLayer(){};
};

}
#endif