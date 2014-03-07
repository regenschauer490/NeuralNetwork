﻿/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_DATA_FORMAT_H
#define SIG_NN_DATA_FORMAT_H

#include "info.hpp"

namespace signn{
	
template <class InputInfo_, class OutputInfo_>
class DataFormat
{
public:
	using InputType_ = typename InputInfo_::type;
	using InputParamType_ = ParamType<InputType_>;
	using InputArrayType_ = std::array<InputType_, InputInfo_::node_num>;
	using OutputType_ = typename OutputInfo_::output_type;
	using OutputParamType_ = ParamType<OutputType_>;
	using OutputArrayType_ = std::array<OutputType_, OutputInfo_::node_num>;

public:
	class InputData
	{
		const bool is_test_data_;
		InputArrayType_ input_;
		OutputArrayType_ teacher_;

	public:
		//teacher:出力と同形式
		template<class Iter1, class Iter2>
		InputData(Iter1 input_begin, Iter1 input_end, Iter2 teacher_begin, Iter2 teacher_end, bool test_data) : is_test_data_(test_data){
			uint i, j;
			for (i = 0; i < InputInfo_::node_num && input_begin != input_end; ++i, ++input_begin) input_[i] = *input_begin;
			for (j = 0; j < OutputInfo_::node_num && teacher_begin != teacher_end; ++j, ++teacher_begin) teacher_[j] = *teacher_begin;
			
			assert(i == InputInfo_::node_num && j == OutputInfo_::node_num && input_begin == input_end && teacher_begin == teacher_end);
		}

		uint size() const{ return InputInfo_::node_num; }

		bool IsTestData() const{ return is_test_data_; }

		InputArrayType_ const& Input() const{ return input_; }
		OutputArrayType_ const& Teacher() const{ return teacher_; }
	};

	class OutputData
	{
		const std::shared_ptr<InputData const> input_;
		OutputArrayType_ estimate_;

	public:
		OutputData(std::shared_ptr<InputData const>& input, OutputArrayType_& estimate) : input_(input), estimate_(estimate){}

/*		double MeanSquareError() const{
			auto ans = input_->Teacher().begin();
			return std::inner_product(estimate_.begin(), estimate_.end(), ans, 0.0, std::plus<double>(), [](typename OutputInfo_::type v1, typename InputInfo_::type v2){ return pow(v1 - v2, 2); }) / OutputInfo_::node_num;
		}
*/
		template<class Iter, typename = decltype(*std::declval<Iter&>(), void(), ++std::declval<Iter&>(), void())>
		double MeanSquareError(Iter ans_vector_begin) const{
			static_assert(OutputInfo_::node_num > 1, "error in OutputLayer::MeanSquareError() : required answer dimension is 1 (scalar)");
			return std::inner_product(estimate_.begin(), estimate_.end(), ans_vector_begin, 0.0, std::plus<double>(), [](OutputParamType_ v1, typename std::iterator_traits<Iter>::value_type v2){ return pow(v1 - v2, 2); }) / OutputInfo_::node_num;
		}

		double MeanSquareError(OutputType_ answer) const{ return pow(estimate_[0] - answer, 2); }
		
		uint size() const{ return OutputInfo_::node_num; }

		auto begin() const ->decltype(estimate_.cbegin()){ return estimate_.cbegin(); }

		auto end() const ->decltype(estimate_.cend()){ return estimate_.cend(); }

		OutputType_ operator [](uint index) const{ return estimate_[index]; }
	};

public:
	using InputDataPtr = std::shared_ptr<typename DataFormat<InputInfo_, OutputInfo_>::InputData const>;

	using OutputDataPtr = std::shared_ptr<typename DataFormat<InputInfo_, OutputInfo_>::OutputData>;

public:
	DataFormat(){};
	virtual ~DataFormat(){};

	//数値ラベルをバイナリベクトルに変換 (ex: class_label = 2 -> [false, false, true, false, ...])
	template <class = typename std::enable_if<sig::Greater<OutputInfo_::node_num, 1>::value, void>::type>
	auto ConvertBinaryVector(uint class_label) const{
		assert(class_label < OutputInfo_::node_num);
		OutputArrayType_ binary_vec;
		for (uint i = 0; i < OutputInfo_::node_num; ++i){
			if (class_label == i) binary_vec[i] = true;
			else binary_vec[i] = false;
		}
		return binary_vec;
	}

	//teacher:出力と同形式
	template<class Iter1, class Iter2, typename = decltype(*std::declval<Iter1&>(), void(), ++std::declval<Iter1&>(), void()), typename = decltype(*std::declval<Iter2&>(), void(), ++std::declval<Iter2&>(), void())>
	InputDataPtr MakeInputData(Iter1 input_begin, Iter1 input_end, Iter2 teacher_begin, Iter2 teacher_end) const{
		return std::make_shared<InputData>(input_begin, input_end, teacher_begin, teacher_end, false);
	}

	//teacher:回帰の正解値
	template<class Iter1, typename = decltype(*std::declval<Iter1&>(), void(), ++std::declval<Iter1&>(), void()), class = typename std::enable_if<!std::is_same<OutputType_, bool>::value>::type>
	InputDataPtr MakeInputData(Iter1 input_begin, Iter1 input_end, OutputParamType_ teacher_value) const{
		static_assert(1 == OutputInfo_::node_num, "invalid input data");

		std::array<OutputType_, 1> teacher{ { teacher_value } };

		return std::make_shared<InputData>(input_begin, input_end, teacher.begin(), teacher.end(), false);
	}

	//teacher:分類のラベル (2値の場合{0,1}, 多値の場合は0から始まる連続した整数)
	template<class Iter1, typename = decltype(*std::declval<Iter1&>(), void(), ++std::declval<Iter1&>(), void()), class = typename std::enable_if<std::is_same<OutputType_, bool>::value>::type>
	InputDataPtr MakeInputData(Iter1 input_begin, Iter1 input_end, uint teacher_label) const{
		OutputArrayType_ teacher;

		if (std::is_same<typename OutputInfo_::layer_type<OutputInfo_>, MultiClassClassificationLayer<OutputInfo_>>::value){
			teacher = ConvertBinaryVector(teacher_label);
		}
		else teacher[0] = teacher_label;

		return std::make_shared<InputData>(input_begin, input_end, teacher.begin(), teacher.end(), false);
	}

	//教師信号なし  (テストデータ)
	template<class Iter1, typename = decltype(*std::declval<Iter1&>(), void(), ++std::declval<Iter1&>(), void())>
	InputDataPtr MakeInputData(Iter1 input_begin, Iter1 input_end) const{
		std::array<OutputType_, OutputInfo_::node_num> teacher;
		return std::make_shared<InputData>(input_begin, input_end, teacher.begin(), teacher.end(), true);
	}
};

}
#endif