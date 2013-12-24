#pragma once

#include "info.hpp"

template <class InputInfo_, class OutputInfo_>
class MLP_Base
{
public:
	class InputData
	{
		std::array<typename InputInfo_::type, InputInfo_::dim> input_;
		std::array<typename OutputInfo_::type, OutputInfo_::dim> teacher_;

	public:
		template<class Iter1, class Iter2>
		InputData(Iter1 input_begin, Iter1 input_end, Iter2 teacher_begin, Iter2 teacher_end){
			uint i, j;
			for (i = 0; i < InputInfo_::dim && input_begin != input_end; ++i, ++input_begin) input_[i] = *input_begin;
			for (j = 0; j < OutputInfo_::dim && teacher_begin != teacher_end; ++j, ++taecher_first) teacher_[j] = *teacher_begin;
			assert(i == InputInfo_::dim && j == OutputInfo_::dim && input_begin == input_end && teacher_begin == teacher_las, "invalid input data");
		}

		template<class Iter1>
		InputData(Iter1 input_begin, Iter1 input_end, typename OutputInfo_::type teacher){
			uint i, j = 0;
			for (i = 0; i < InputInfo_::dim && input_begin != input_end; ++i, ++input_begin) input_[i] = *input_begin;
			if (OutputInfo_::e_layertype == OutputLayerType::MultiClassClassification){
				assert(teacher < OutputInfo_::dim);
				for (; j < OutputInfo_::dim; ++j){
					if (teacher == j) teacher_[j] = 1;
					else teacher_[j] = 0;
				}
			}
			else teacher_[j++] = teacher;
			assert(i == InputInfo_::dim && j == OutputInfo_::dim && input_begin == input_end, "invalid input data");
		}

		/*template<class Iter1>
		InputData(Iter1 input_begin, Iter1 input_end){
		uint i;
		for (i = 0; i < InputInfo_::dim && input_begin != input_end; ++i, ++input_begin) input_[i] = *input_begin;
		assert(i == InputInfo_::dim && 1 == OutputInfo_::dim && input_begin == input_end, "invalid input data");
		}*/

		std::array<typename InputInfo_::type, InputInfo_::dim> const& Input() const{ return input_; }
		std::array<typename OutputInfo_::type, OutputInfo_::dim> const& Teacher() const{ return teacher_; }
	};

	class OutputData{

	};

public:
	MLP_Base();
	virtual ~MLP_Base(){};


};

