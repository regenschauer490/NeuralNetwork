#pragma once

#include "info.hpp"

namespace signn{

template <class InputInfo_, class OutputInfo_>
class MLP_Base
{
public:
	class InputData
	{
		const bool is_test_data_;
		std::array<typename InputInfo_::type, InputInfo_::dim> input_;
		std::array<typename OutputInfo_::type, OutputInfo_::dim> teacher_;

	public:
		template<class Iter1, class Iter2>
		InputData(Iter1 input_begin, Iter1 input_end, Iter2 teacher_begin, Iter2 teacher_end) : is_test_data_(false){
			uint i, j;
			for (i = 0; i < InputInfo_::dim && input_begin != input_end; ++i, ++input_begin) input_[i] = *input_begin;
			for (j = 0; j < OutputInfo_::dim && teacher_begin != teacher_end; ++j, ++taecher_first) teacher_[j] = *teacher_begin;
			assert(i == InputInfo_::dim && j == OutputInfo_::dim && input_begin == input_end && teacher_begin == teacher_las, "invalid input data");
		}

		template<class Iter1>
		InputData(Iter1 input_begin, Iter1 input_end, typename OutputInfo_::type teacher) : is_test_data_(false){
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

		template<class Iter1>
		InputData(Iter1 input_begin, Iter1 input_end) : is_test_data_(true){
			uint i;
			for (i = 0; i < InputInfo_::dim && input_begin != input_end; ++i, ++input_begin) input_[i] = *input_begin;
			assert(i == InputInfo_::dim && 1 == OutputInfo_::dim && input_begin == input_end, "invalid input data");
		}

		bool IsTestData() const{ return is_test_data_; }
		std::array<typename InputInfo_::type, InputInfo_::dim> const& Input() const{ return input_; }
		std::array<typename OutputInfo_::type, OutputInfo_::dim> const& Teacher() const{ return teacher_; }
	};

	class OutputData
	{
		const std::shared_ptr<InputData> input_;
		std::array<typename OutputInfo_::type, OutputInfo_::dim> estimate_;

	public:
		OutputData(std::shared_ptr<InputData> input, std::array<typename OutputInfo_::type, OutputInfo_::dim> estimate) : input_(input), estimate_(estimate){}

		template<class Iter, typename = decltype(*std::declval<Iter&>(), void(), ++std::declval<Iter&>(), void())>
		double SquareError(Iter ans_vector_begin) const{
			static_assert(OutputInfo_::dim > 1, "error in OutputLayer::SquareError() : required answer dimension is 1 (scalar)");
			return std::inner_product(estimate_.begin(), estimate_->Input().end(), ans_vector_begin, 0.0, std::plus<double>(), [](typename OutputInfo_::type v1, typename std::iterator_traits<Iter>::value_type v2){ return pow(v1 - v2, 2); });
		}

		double SquareError(typename OutputInfo_::type answer) const{
			static_assert(OutputInfo_::dim < 2, "error in OutputLayer::SquareError() : answer dimension requires more than 2 (vector)");
			return pow(estimate_[0] - answer, 2);
		}

		uint size() const{ return OutputInfo_::dim; }

		auto begin() const ->decltype(estimate_.cbegin()){ return estimate_.cbegin(); }

		auto end() const ->decltype(estimate_.cend()){ return estimate_.cend(); }

		typename OutputInfo_::type operator [](uint index) const{ return estimate_[index]; }
	};

public:
	MLP_Base(){};
	virtual ~MLP_Base(){};

	template<class Iter1, class Iter2, typename = decltype(*std::declval<Iter1&>(), void(), ++std::declval<Iter1&>(), void()), typename = decltype(*std::declval<Iter2&>(), void(), ++std::declval<Iter2&>(), void())>
	std::shared_ptr<InputData> MakeInputData(Iter1 input_begin, Iter1 input_end, Iter2 teacher_begin, Iter2 teacher_end) const{
		return std::make_shared<InputData>(input_begin, input_end, teacher_begin, teacher_end);
	}

	template<class Iter1, typename = decltype(*std::declval<Iter1&>(), void(), ++std::declval<Iter1&>(), void())>
	std::shared_ptr<InputData> MakeInputData(Iter1 input_begin, Iter1 input_end, typename OutputInfo_::type teacher) const{
		return std::make_shared<InputData>(input_begin, input_end, teacher);
	}

	template<class Iter1, typename = decltype(*std::declval<Iter1&>(), void(), ++std::declval<Iter1&>(), void())>
	std::shared_ptr<InputData> MakeInputData(Iter1 input_begin, Iter1 input_end) const{
		return std::make_shared<InputData>(input_begin, input_end);
	}
};

template <class InputInfo_, class OutputInfo_>
using InputDataPtr = std::shared_ptr<typename MLP_Base<InputInfo_, OutputInfo_>::InputData>;

template <class InputInfo_, class OutputInfo_>
using OutputDataPtr = std::shared_ptr<typename MLP_Base<InputInfo_, OutputInfo_>::OutputData>;


}