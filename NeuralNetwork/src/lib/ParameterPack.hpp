#pragma once

#include "info.hpp"

namespace signn{

template <class DataType>
class ParameterPack
{
	double node_threshold_;

	template <class InputInfo_, class OutputInfo_>
	friend class NewralNetwork;
public:
	ParameterPack(double node_threshold) : node_threshold_(node_threshold){};
	~ParameterPack(){};
};

}