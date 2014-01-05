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

#include <iostream>
#include <fstream>
#include <sstream>
#include <locale>
#include <string>
#include <memory>
#include <random>
#include <functional>
#include <algorithm>
#include <numeric>
#include <type_traits>

/* namespace / typedef */
typedef unsigned long int uint;

namespace signn{
#undef min
	
	//初期化時に指定した範囲の一様分布乱数を発生させるクラス
	//デフォルト: 乱数生成器 -> メルセンヌツイスター
	template <class NumType, class Engine = std::mt19937>
	class SimpleRandom {
		Engine _engine;		//乱数生成アルゴリズム 
		typename std::conditional <
			std::is_integral<NumType>::value,
			std::uniform_int_distribution<int>,
			std::uniform_real_distribution<double>
		> ::type _dist;		//確率分布

	public:
		SimpleRandom(NumType min, NumType max, bool debug) : _engine(
			[debug](){
				std::random_device rnd;
				std::vector<uint> v(10);
				if (debug) std::fill(v.begin(), v.end(), 0);
				else std::generate(v.begin(), v.end(), std::ref(rnd));

				return Engine(std::seed_seq(v.begin(), v.end()));
		}()
			),
			_dist(min, max){}

		NumType operator()(){
			return _dist(_engine);
		}
	};

	//コンテナに格納された全文字列を結合して1つの文字列に(delimiterで区切り指定)
	template < class T, template < class T, class = std::allocator<T >> class Container >
	inline std::string CatStr(Container<T> const& container, std::string delimiter = "")
	{
		std::ostringstream ostream;

		for (auto const& src : container){
			ostream << src << delimiter;
		}
		return ostream.str();
	}

	//文字列をある文字を目印に分割する
	template < class String, template<class T, class Allocator = std::allocator<T >> class Container = std::vector >
	Container<String> Split(String src, typename std::common_type<String>::type const& delim, bool ignore_blank = true)
	{
		Container<String> result;
		int const mag = delim.size();
		int cut_at;

		while ((cut_at = src.find(delim)) != src.npos){
			if (!ignore_blank || cut_at > 0) result.push_back(src.substr(0, cut_at));
			src = src.substr(cut_at + mag);
		}
		if (!ignore_blank || src.length() > 0){
			result.push_back(src);
		}

		return std::move(result);
	}

	//浮動小数点型にも使える等値比較関数
	template <class T1, class T2>
	bool Equal(T1 v1, T2 v2)
	{
		const auto dmin = std::numeric_limits<std::common_type<T1, T2>::type>::min();

		return std::abs(v1 - v2) < dmin;
	}

	template<class Iter1, class Iter2>
	double Simirarlity(Iter1 xs_begin, Iter1 xs_end, Iter2 hs_begin, Iter2 hs_end, uint dilation)
	{
		uint size = 0;
		double result = 0;
		auto xs = xs_begin + dilation;
		auto hs = hs_begin;

		if (std::is_same<typename Iter1::value_type, bool>::value){
			for (; xs != xs_end && hs != hs_end; ++xs, ++hs, ++size){
				result += (*xs) == (*hs) ? 0 : 1;
			}
			for (xs = xs_begin; xs != xs_end && hs != hs_end; ++xs, ++hs, ++size){
				result += (*xs) == (*hs) ? 0 : 1;
			}
		}
		else{
			for (; xs != xs_end && hs != hs_end ; ++xs, ++hs, ++size){
				result += std::abs((*xs) - (*hs));
			}
			for (xs = xs_begin; xs != xs_end && hs != hs_end; ++xs, ++hs, ++size){
				result += std::abs((*xs) - (*hs));
			}
		}

		return result / size;
	}


/* 入出力 */
namespace File{

	//for type map
	template <class FILE_STRING> struct OFS_SELECTION{};
	template<> struct OFS_SELECTION<char const*>{
		typedef std::ofstream fstream;
		typedef std::ostreambuf_iterator<char> fstreambuf_iter;
	};
	template<> struct OFS_SELECTION<std::string>{
		typedef std::ofstream fstream;
		typedef std::ostreambuf_iterator<char> fstreambuf_iter;
	};
	template<> struct OFS_SELECTION<wchar_t const*>{
		typedef std::wofstream fstream;
		typedef std::ostreambuf_iterator<wchar_t> fstreambuf_iter;
	};
	template<> struct OFS_SELECTION<std::wstring>{
		typedef std::wofstream fstream;
		typedef std::ostreambuf_iterator<wchar_t> fstreambuf_iter;
	};

	template <class FILE_STRING> struct IFS_SELECTION{};
	template<> struct IFS_SELECTION<std::string>{
		typedef std::ifstream fstream;
		typedef std::istreambuf_iterator<char> fstreambuf_iter;
	};
	template<> struct IFS_SELECTION<std::wstring>{
		typedef std::wifstream fstream;
		typedef std::istreambuf_iterator<wchar_t> fstreambuf_iter;
	};

	template <class NUM> struct S2NUM_SELECTION{};
	template <> struct S2NUM_SELECTION<int>{
		int operator()(std::string s){ return std::stoi(s); }
	};
	template <> struct S2NUM_SELECTION<long>{
		long operator()(std::string s){ return std::stol(s); }
	};
	template <> struct S2NUM_SELECTION<long long>{
		long long operator()(std::string s){ return std::stoll(s); }
	};
	template <> struct S2NUM_SELECTION<unsigned int>{
		unsigned int operator()(std::string s){ return std::stoul(s); }
	};
	template <> struct S2NUM_SELECTION<unsigned long>{
		unsigned long operator()(std::string s){ return std::stoul(s); }
	};
	template <> struct S2NUM_SELECTION<unsigned long long>{
		unsigned long long operator()(std::string s){ return std::stoull(s); }
	};
	template <> struct S2NUM_SELECTION<float>{
		float operator()(std::string s){ return std::stof(s); }
	};
	template <> struct S2NUM_SELECTION<double>{
		double operator()(std::string s){ return std::stod(s); }
	};

	enum class WriteMode{ overwrite, append };

	inline void RemakeFile(std::wstring const& file_pass)
	{
		std::ofstream ofs(file_pass);
		ofs << "";
	}

	//-- Save Text

	template <class String>
	inline void SaveLine(String const& src, typename OFS_SELECTION<String>::fstream& ofs)
	{
		ofs << src << std::endl;
	}

	template <class String>
	inline void SaveLine(std::vector<String> const& src, typename OFS_SELECTION<String>::fstream& ofs)
	{
		typename OFS_SELECTION<String>::fstreambuf_iter streambuf_iter(ofs);
		for (auto const& str : src){
			std::copy(str.begin(), str.end(), streambuf_iter);
			streambuf_iter = '\n';
		}
	}

	//ファイルへ1行ずつ保存
	//args -> src: 保存対象, file_pass: 保存先のディレクトリとファイル名（フルパス）, open_mode: 上書き(overwrite) or 追記(append)
	template <class String>
	inline void SaveLine(String src, std::wstring const& file_pass, WriteMode mode)
	{
		static bool first = true;
		if (first){
			std::locale::global(std::locale(""));
			first = false;
		}

		std::ios::open_mode const open_mode = mode == WriteMode::overwrite ? std::ios::out : std::ios::out | std::ios::app;
		typename OFS_SELECTION<std::decay<String>::type>::fstream ofs(file_pass, open_mode);
		SaveLine(src, ofs);
	}
	template <class String>
	inline void SaveLine(std::vector<String> const& src, std::wstring const& file_pass, WriteMode mode)
	{
		static bool first = true;
		if (first){
			std::locale::global(std::locale(""));
			first = false;
		}

		std::ios::open_mode const open_mode = mode == WriteMode::overwrite ? std::ios::out : std::ios::out | std::ios::app;
		typename OFS_SELECTION<String>::fstream ofs(file_pass, open_mode);
		SaveLine(src, ofs);
	}

	template <class Num>
	inline void SaveLineNum(std::vector<Num> const& src, std::wstring const& file_pass, WriteMode mode, std::string delimiter = "\n")
	{
		SaveLine(CatStr(src, delimiter), file_pass, mode);
	}


	//-- Read Text

	template <class R>
	inline void ReadLine(std::vector<R>& empty_dest, typename IFS_SELECTION<R>::fstream& ifs, std::function< R(typename std::conditional<std::is_same<typename IFS_SELECTION<R>::fstream, std::ifstream>::value, std::string, std::wstring>::type)> const& conv = nullptr)
	{
		typename std::conditional<std::is_same<typename IFS_SELECTION<R>::fstream, std::ifstream>::value, std::string, std::wstring>::type line;

		while (ifs && getline(ifs, line)){
			conv ? empty_dest.push_back(conv(std::move(line))) : empty_dest.push_back(std::move(line));
		}
	}

	template <class R>
	inline void ReadLine(std::vector<R>& empty_dest, std::wstring const& file_pass, std::function< R(typename std::conditional<std::is_same<typename IFS_SELECTION<R>::fstream, std::ifstream>::value, std::string, std::wstring>::type)> const& conv = nullptr)
	{
		typename IFS_SELECTION<R>::fstream ifs(file_pass);
		if (!ifs){
			wprintf(L"file open error: %s \n", file_pass);
			return;
		}
		ReadLine(empty_dest, ifs);
	}

	template <class Num>
	inline void ReadLineNum(std::vector<Num>& empty_dest, std::wstring const& file_pass)
	{
		typename IFS_SELECTION<std::string>::fstream ifs(file_pass);
		S2NUM_SELECTION<Num> conv;
		std::string line;
		if (!ifs){
			wprintf(L"file open error: %s \n", file_pass);
			return;
		}
		while (ifs && getline(ifs, line)) empty_dest.push_back(conv(std::move(line)));
	}

		//ディレクトリ・ファイルパスの末尾に'/'or'\\'があるかチェックし、付けるか外すかどうか指定
		inline std::wstring DirpassTailModify(std::wstring const& directory_pass, bool const has_slash)
		{
			if (directory_pass.empty()) return directory_pass;

			auto tail = directory_pass.back();

			if (has_slash){
				//付ける場合
				if (tail == '/' || tail == '\\') return directory_pass;
				else return (directory_pass + L"/");
			}
			else{
				if (tail != '/' && tail != '\\') return directory_pass;
				else{
					auto tmp = directory_pass;
					tmp.pop_back();
					return tmp;
				}
			}

		};
		
	}

namespace Metrics{

	template < class T, template < class T, class = std::allocator<T >> class Container>
		inline double SquareError(Container<T> const& estimate, Container<T> const& answer) {
		return std::inner_product(estimate.begin(), estimate.end(), answer.begin(), 0.0, std::plus<double>(), [](T v1, T v2){ return pow(v1 - v2, 2); });
	}

	template < class T, template < class T, class = std::allocator<T>> class Container>
	double MeanSquareError(Container< Container<T>> const& estimates, Container< Container<T>> const& answers)
	{
		const uint dsize = std::min(estimates.size(), answers.size());
		double error = 0;

		for (uint i = 0; i < dsize; ++i){
			error += SquareError(estimates[i], answers[i]);
		}
		return error / dsize;
	}
}

}
