#pragma once

#include <iostream>
#include <sstream>
#include <locale>
#include <string>
#include <memory>
#include <random>
#include <functional>
#include <algorithm>
#include <numeric>

/* namespace / typedef */
typedef unsigned long int uint;

namespace signn{
#undef min()

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


/* 入出力 */
namespace File{

		//for type map
		template <class FILE_STRING> struct OFS_SELECTION{};
		template<> struct OFS_SELECTION<std::string>{
			typedef std::ofstream type;
		};
		template<> struct OFS_SELECTION<std::wstring>{
			typedef std::wofstream type;
		};

		template <class FILE_STRING> struct IFS_SELECTION{};
		template<> struct IFS_SELECTION<std::string>{
			typedef std::ifstream type;
		};
		template<> struct IFS_SELECTION<std::wstring>{
			typedef std::wifstream type;
		};

		enum class WriteMode{ overwrite, append };

		// Save Text

		template <class String>
		inline void SaveLine(String const& src, typename OFS_SELECTION<String>::type& ofs)
		{
			ofs << src << std::endl;
		}

		template <class String>
		inline void SaveLine(std::vector<String> const& src, typename OFS_SELECTION<String>::type& ofs)
		{
			std::for_each(src.begin(), src.end(), [&](String const& line){
				SaveLine(line, ofs);
			});
		}

		//ファイルへ1行ずつ保存
		//args -> src: 保存対象, file_pass: 保存先のディレクトリとファイル名（フルパス）, open_mode: 上書き(overwrite) or 追記(append)
		template <class String>
		inline void SaveLine(String const& src, std::wstring const& file_pass, WriteMode mode = WriteMode::overwrite)
		{
			static bool first = true;
			if (first){
				std::locale::global(std::locale(""));
				first = false;
			}

			std::ios::open_mode const open_mode = mode == WriteMode::overwrite ? std::ios::out : std::ios::out | std::ios::app;
			typename OFS_SELECTION<String>::type ofs(file_pass, open_mode);
			SaveLine(src, ofs);
		}
		template <class String>
		inline void SaveLine(std::vector<String> const& src, std::wstring const& file_pass, WriteMode mode = WriteMode::overwrite)
		{

			static bool first = true;
			if (first){
				std::locale::global(std::locale(""));
				first = false;
			}

			std::ios::open_mode const open_mode = mode == WriteMode::overwrite ? std::ios::out : std::ios::out | std::ios::app;
			typename OFS_SELECTION<String>::type ofs(file_pass, open_mode);
			SaveLine(src, ofs);
		}

		// int ver
		template <class Num>
		inline void SaveLineInt(std::vector<Num> const& src, std::wstring const& file_pass, std::ios::open_mode const open_mode = std::ios::out)
		{
			std::ofstream ofs(file_pass, open_mode);
			std::for_each(src.begin(), src.end(), [&](Num val){
				SaveLine(std::to_string(val), ofs);
			});
		}

		// Read Text

		template <class R, class Stream>
		inline void ReadLine(std::vector<R>& empty_dest, Stream& ifs, std::function< R(typename std::conditional<std::is_same<Stream, std::ifstream>::value, std::string, std::wstring>::type)> const& conv)
		{
			typename std::conditional<std::is_same<Stream, std::ifstream>::value, std::string, std::wstring>::type line;

			empty_dest.clear();
			//ifs.seekg(0, std::ios::beg).read(&empty_dest[0], static_cast<std::streamsize>(empty_dest.size()));
			while (ifs && getline(ifs, line)){ empty_dest.push_back(conv(std::move(line))); }
		}

		//読み込み失敗: empty_dest.size() == 0
		template <class String>
		inline void ReadLine(std::vector<String>& empty_dest, typename IFS_SELECTION<String>::type& ifs)
		{
			ReadLine<String>(empty_dest, ifs, [](String line){ return line; });
		}

		//読み込み失敗: empty_dest.size() == 0
		template <class String>
		inline void ReadLine(std::vector<String>& empty_dest, std::wstring const& file_pass)
		{
			typename IFS_SELECTION<String>::type ifs(file_pass);
			if (!ifs){
				wprintf(L"file open error: %s \n", file_pass);
				return;
			}
			ReadLine<String>(empty_dest, ifs);
		}


		// int ver
		template <class Num, class String = std::string>
		inline void ReadLineInt(std::vector<Num>& empty_dest, std::wstring const& file_pass)
		{
			typename IFS_SELECTION<String>::type ifs(file_pass);
			if (!ifs){
				wprintf(L"file open error: %s \n", file_pass);
				return;
			}
			ReadLine<Num>(empty_dest, ifs, [](String text){ return std::stol(text); });
		}

#ifdef ENABLE_BOOST

		//読み込み失敗: return -> maybe == nothing
		template <class String>
		inline maybe<std::vector<String>> ReadLine(typename IFS_SELECTION<String>::type& ifs)
		{
			typedef std::vector<String> R;
			R tmp;
			ReadLine<String>(tmp, ifs);
			return tmp.size() ? maybe<R>(std::move(tmp)) : nothing;
		}

		//読み込み失敗: return -> maybe == nothing
		template <class String>
		inline maybe<std::vector<String>> ReadLine(std::wstring const& file_pass)
		{
			typename IFS_SELECTION<String>::type ifs(file_pass);
			if (!ifs){
				std::wcout << L"file open error: " << file_pass << std::endl;
				return nothing;
			}
			return ReadLine<String>(ifs);
		}

		// int ver
		template <class Num, class String = std::string>
		inline maybe<std::vector<Num>> ReadLineInt(std::wstring const& file_pass)
		{
			std::vector<Num> tmp;
			ReadLineInt<Num, String>(tmp, file_pass);
			return tmp.size() ? maybe<std::vector<Num>>(std::move(tmp)) : nothing;
		}

#endif
		
	}

}
