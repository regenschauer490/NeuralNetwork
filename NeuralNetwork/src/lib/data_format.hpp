/*
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

/*		double MeanSquareError() const{
			auto ans = input_->Teacher().begin();
			return std::inner_product(estimate_.begin(), estimate_.end(), ans, 0.0, std::plus<double>(), [](typename OutputInfo_::type v1, typename InputInfo_::type v2){ return pow(v1 - v2, 2); }) / OutputInfo_::dim;
		}
*/
		template<class Iter, typename = decltype(*std::declval<Iter&>(), void(), ++std::declval<Iter&>(), void())>
		double MeanSquareError(Iter ans_vector_begin) const{
			static_assert(OutputInfo_::dim > 1, "error in OutputLayer::MeanSquareError() : required answer dimension is 1 (scalar)");
			return std::inner_product(estimate_.begin(), estimate_.end(), ans_vector_begin, 0.0, std::plus<double>(), [](typename OutputInfo_::type v1, typename std::iterator_traits<Iter>::value_type v2){ return pow(v1 - v2, 2); }) / OutputInfo_::dim;
		}

		double MeanSquareError(typename OutputInfo_::type answer) const{
			if (OutputInfo_::e_layertype == OutputLayerType::MultiClassClassification){
				assert(answer < OutputInfo_::dim);
				typename InputData::TeacherDataArray ans;
				for (uint i = 0; i < OutputInfo_::dim; ++i){
					if (answer == i)ans[i] = true;
					else ans[i] = false;
				}
				return std::inner_product(estimate_.begin(), estimate_.end(), ans.begin(), 0.0, std::plus<double>(), [](typename OutputInfo_::type v1, typename OutputInfo_::type v2){ return pow(v1 - v2, 2); }) / OutputInfo_::dim;
			}
			else return pow(estimate_[0] - answer, 2);
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
		typename InputData::TeacherDataArray teacher;

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
#endif