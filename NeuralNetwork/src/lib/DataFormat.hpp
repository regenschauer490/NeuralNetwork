/*
The MIT License(MIT)

Copyright(c) 2014 Akihiro Nishimura

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

#include "info.hpp"

namespace signn{
	
template <class InputInfo_, class OutputInfo_>
class DataFormat
{
public:
	class InputData
	{
	public:
		typedef std::array<typename InputInfo_::type, InputInfo_::dim> InputDataArray;
		typedef std::array<typename OutputInfo_::type, OutputInfo_::dim> TeacherDataArray;

	private:
		const bool is_test_data_;
		InputDataArray input_;
		TeacherDataArray teacher_;

	public:
		//teacher:出力と同形式
		template<class Iter1, class Iter2>
		InputData(Iter1 input_begin, Iter1 input_end, Iter2 teacher_begin, Iter2 teacher_end, bool test_data) : is_test_data_(test_data){
			uint i, j;
			for (i = 0; i < InputInfo_::dim && input_begin != input_end; ++i, ++input_begin) input_[i] = *input_begin;
			for (j = 0; j < OutputInfo_::dim && teacher_begin != teacher_end; ++j, ++teacher_begin) teacher_[j] = *teacher_begin;
			
			assert(i == InputInfo_::dim && j == OutputInfo_::dim && input_begin == input_end && teacher_begin == teacher_end);
		}

		uint size() const{ return InputInfo_::dim; }
		bool IsTestData() const{ return is_test_data_; }

		InputDataArray const& Input() const{ return input_; }
		TeacherDataArray const& Teacher() const{ return teacher_; }
	};

	class OutputData
	{
	public:
		typedef std::array<typename OutputInfo_::type, OutputInfo_::dim> OutputDataArray;

	private:
		const std::shared_ptr<InputData const> input_;
		OutputDataArray estimate_;

	public:
		OutputData(std::shared_ptr<InputData const> input, OutputDataArray& estimate) : input_(input), estimate_(estimate){}

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
	typedef std::shared_ptr<typename DataFormat<InputInfo_, OutputInfo_>::InputData const> InputDataPtr;

	typedef std::shared_ptr<typename DataFormat<InputInfo_, OutputInfo_>::OutputData> OutputDataPtr;

public:
	DataFormat(){};
	virtual ~DataFormat(){};

	//teacher:出力と同形式
	template<class Iter1, class Iter2, typename = decltype(*std::declval<Iter1&>(), void(), ++std::declval<Iter1&>(), void()), typename = decltype(*std::declval<Iter2&>(), void(), ++std::declval<Iter2&>(), void())>
	InputDataPtr MakeInputData(Iter1 input_begin, Iter1 input_end, Iter2 teacher_begin, Iter2 teacher_end) const{
		return std::make_shared<InputData>(input_begin, input_end, teacher_begin, teacher_end, false);
	}

	//teacher:回帰の正解値
	template<class Iter1, typename = decltype(*std::declval<Iter1&>(), void(), ++std::declval<Iter1&>(), void()), class = typename std::enable_if<!std::is_same<typename OutputInfo_::type, bool>::value>::type>
	InputDataPtr MakeInputData(Iter1 input_begin, Iter1 input_end, typename OutputInfo_::type teacher_value) const{
		static_assert(1 == OutputInfo_::dim, "invalid input data");

		std::array<typename OutputInfo_::type, 1> teacher{ {teacher_value} };

		return std::make_shared<InputData>(input_begin, input_end, teacher.begin(), teacher.end(), false);
	}

	//teacher:分類のラベル
	template<class Iter1, typename = decltype(*std::declval<Iter1&>(), void(), ++std::declval<Iter1&>(), void()), class = typename std::enable_if<std::is_same<typename OutputInfo_::type, bool>::value>::type>
	InputDataPtr MakeInputData(Iter1 input_begin, Iter1 input_end, uint teacher_label) const{
		std::array<typename OutputInfo_::type, OutputInfo_::dim> teacher;

		if (OutputInfo_::e_layertype == OutputLayerType::MultiClassClassification){
			assert(teacher_label < OutputInfo_::dim);
			for (uint i = 0; i < OutputInfo_::dim; ++i){
				if (teacher_label == i) teacher[i] = true;
				else teacher[i] = false;
			}
		}
		else teacher[0] = teacher_label;

		return std::make_shared<InputData>(input_begin, input_end, teacher.begin(), teacher.end(), false);
	}

	//教師信号なし  (テストデータ)
	template<class Iter1, typename = decltype(*std::declval<Iter1&>(), void(), ++std::declval<Iter1&>(), void())>
	std::shared_ptr<InputData> MakeInputData(Iter1 input_begin, Iter1 input_end) const{
		std::array<typename OutputInfo_::type, OutputInfo_::dim> teacher;
		return std::make_shared<InputData>(input_begin, input_end, teacher.begin(), teacher.end(), true);
	}
};


}