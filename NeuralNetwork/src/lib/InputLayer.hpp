#pragma once

#include "Layer.h"

namespace signn{

template <class InputInfo_>
class InputLayer : public Layer
{
	FRIEND_WITH_LAYER

private:
	InputLayer() : Layer(InputInfo_::dim){};

	void UpdateNodeScore() override{}

	void SetData(std::array<typename InputInfo_::type, InputInfo_::dim> const& input){
		for (uint i = 0, end = NodeNum(); i < end; ++i) (*this)[i]->Score(input[i]);
	}

public:
	~InputLayer(){};
};

}