#ifndef UTILITY_H
#define UTILITY_H

/* Last Update : 2013 12 15 */

#define ENABLE_BOOST

#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <locale>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <memory>
#include <random>
#include <functional>
#include <algorithm>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <numeric>
#include <regex>

#ifdef ENABLE_BOOST

#include <boost/optional.hpp>
#include <boost/format.hpp>
#include <boost/call_traits.hpp>
#include <boost/range.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/serialization/serialization.hpp>

#endif

/* namespace / typedef */
typedef unsigned long int uint;
typedef std::shared_ptr< std::string > StrPtr;
typedef std::shared_ptr< std::string const > C_StrPtr;
typedef std::shared_ptr< std::wstring > WStrPtr;
typedef std::shared_ptr< std::wstring const > C_WStrPtr;

#ifdef ENABLE_BOOST

template <typename T>
using maybe = boost::optional<T>;
//#define maybe boost::optional
auto const nothing = boost::none;

#endif

namespace sig{
#undef min()

/* ���^�֐��E���^�N���X */
	struct NullType{};

/* �R���e�i */

#ifdef ENABLE_BOOST
	//���I�m�ۂ����Œ蒷�z�� (�\�z��̃T�C�Y�ύX�s��)
	template <class T, class Allocator = std::allocator<T>>
	class FixedVector
	{
	public:
		typedef T value_type;
		typedef typename boost::call_traits<T>::param_type		param_type;
		typedef typename boost::call_traits<T>::reference		reference;
		typedef typename boost::call_traits<T>::const_reference	const_reference;
		typedef typename boost::call_traits<T>::value_type		result_type;

		typedef uint size_type;

	private:
		std::vector<T> _data;

	public:
		explicit FixedVector(Allocator const& alloc = Allocator()) : _data(alloc){}
		explicit FixedVector(size_type size, param_type value = T(), Allocator const& alloc = Allocator()) : _data(size, value, alloc){}
	//	explicit FixedVector(size_type count, T value) : _data(size, value){}
		template <class InputIter> FixedVector(InputIter first, InputIter last, Allocator const& alloc = Allocator()) : _data(first, last, alloc){}
		explicit FixedVector(std::vector<T> const& src) : _data(src){}
		explicit FixedVector(std::vector<T> && src) : _data(move(src)){}

		FixedVector(FixedVector const& src) : _data(src._data){}
		FixedVector(FixedVector && src) : _data(move(src._data)){}
		FixedVector(std::initializer_list<T> init, Allocator const& alloc = Allocator()) : _data(init){}

		FixedVector& operator=(FixedVector const& src){
			*this = FixedVector(src);
			return *this;
		}
		FixedVector& operator=(FixedVector && src){
			this->_data = move(src._data);
			return *this;
		}
		FixedVector& operator=(std::initializer_list<T> ilist){
			*this = FixedVector(ilist.begin(), ilist.end());
			return *this;
		}
		FixedVector& operator=(std::vector<T> const& src){
			this->_data = src;
			return *this;
		}
		FixedVector& operator=(std::vector<T> && src){
			this->_data = move(src);
			return *this;
		}
		
		auto begin()->decltype(_data.begin()){ return _data.begin(); }
		auto begin() const ->decltype(_data.begin()){ return _data.begin(); }
		auto cbegin() const ->decltype(_data.cbegin()){ return _data.cbegin(); }

		auto end()->decltype(_data.end()){ return _data.end(); }
		auto end() const ->decltype(_data.end()){ return _data.end(); }
		auto cend() const ->decltype(_data.cend()){ return _data.cend(); }

		auto rbegin()->decltype(_data.rbegin()){ return _data.rbegin(); }
		auto rbegin() const ->decltype(_data.rbegin()){ return _data.rbegin(); }
		auto crbegin() const ->decltype(_data.crbegin()){ return _data.crbegin(); }

		auto rend()->decltype(_data.rend()){ return _data.rend(); }
		auto rend() const ->decltype(_data.rend()){ return _data.rend(); }
		auto crend() const ->decltype(_data.crend()){ return _data.crend(); }

		reference at(size_type pos){ return _data.at(pos); }
		const_reference at(size_type pos) const{ return _data.at(pos); }

		reference operator [](size_type pos){ return _data[pos]; }
		const_reference operator [](size_type pos) const{ return _data[pos]; }

		reference front(){ return _data.front(); }
		const_reference front() const{ return _data.front(); }

		reference back(){ return _data.back(); }
		const_reference back() const{ return _data.back(); }

		bool empty() const{ return _data.empty(); }

		size_type size() const{ return _data.size(); }

		size_type max_size() const{ return _data.max_size; }

		void swap(FixedVector& other){ _data.swap(other); }
	};
#endif

/* �֐��^�v���O���~���O */

	//[a] -> (a -> r) -> [r]
	template < class R, class A, template < class T, class = std::allocator<T >> class Container>
		Container<R> Map(Container<A> const& list, std::function<typename std::common_type<R>::type(typename std::common_type<A>::type)> const& func)
		{
			Container<R> result;

			for (auto e : list) result.push_back(func(e));

			return std::move(result);
		}

	//[a] -> [b] -> (a -> b -> r) -> [r]
	//�߂�l�̌^R�́A�����I�Ɏw�肷��K�v����
	template < class R, class A, class B, template < class T, class = std::allocator<T>> class Container>
		Container<R> ZipWith(Container<A> const& list1, Container<B> const& list2, std::function<typename std::common_type<R>::type(typename std::common_type<A>::type, typename std::common_type<B>::type)> const& func)
		{
			const uint length = list1.size() < list2.size() ? list1.size() : list2.size();
			Container<R> result;

			uint i = 0;
			for (auto it1 = list1.begin(), it2 = list2.begin(), end1 = list1.end(), end2 = list2.end(); i < length; ++i, ++it1, ++it2) result.push_back(func(*it1, *it2));

			return std::move(result);
		}

#ifdef ENABLE_BOOST
	//[a] -> b -> (a -> b -> r) -> [r]
	//�߂�l�̌^R�́A�����I�Ɏw�肷��K�v����
	template < class R, class A, class B, template < class T, class = std::allocator<T >> class Container>
		Container<R> ZipWith(Container<A> const& list1, typename boost::call_traits<B>::param_type val, std::function<typename std::common_type<R>::type(typename std::common_type<A>::type, typename std::common_type<B>::type)> const& func)
	{
		Container<R> result;

		uint i = 0;
		for (auto it1 = list1.begin(), end1 = list1.end(); i < list1.size(); ++i, ++it1) result.push_back(func(*it1, val));

		return std::move(result);
	}
#endif

	//[a] -> b -> (a -> common<a,b> -> common<a,b>) -> common<a,b>
	//std::accumulate�Ƃ͈Ⴂ�A�����l�̌^B�ł͂Ȃ�A��B���ÖٓI�ɕϊ������^�ɏW��
	template < class A, class B, template < class T, class = std::allocator<T>> class Container>
		typename std::common_type<A, B>::type Accumulate(Container<A> const& list, B init, std::function<typename std::common_type<A>::type(typename std::common_type<A>::type, typename std::common_type<A, B>::type)> const& func){
			typename std::common_type<A, B>::type result = init;
			for(auto const& e : list)  result = func(e, result);
			return result;
		}


/* ���� */

	//���������Ɏw�肵���͈͂̈�l���z�����𔭐�������N���X
	//�f�t�H���g: ���������� -> �����Z���k�c�C�X�^�[
	template <class NumType, class Engine = std::mt19937>
	class SimpleRandom {
		Engine _engine;		//���������A���S���Y�� 
		typename std::conditional <
			std::is_integral<NumType>::value,
			std::uniform_int_distribution<int>,
			std::uniform_real_distribution<double>
		> ::type _dist;		//�m�����z

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


	//�d���̖������������𐶐�
	template < template < class T, class = std::allocator<T >> class Container = std::vector >
	Container<int> RandomUniqueNumbers(uint n, int min, int max, bool debug) {
		std::unordered_set<int> match;
		Container<int> result;
		static SimpleRandom<int> Rand(0, max - min, debug);

		int r;
		for (int i = 0; i < n; ++i){
			do{
				r = min + Rand();
			} while (match.find(r) != match.end());

			match.insert(r);
			result.push_back(r);
		}

		return std::move(result);
	}


/* �֗��֐� */

	//���������_�^�ɂ��g���铙�l��r�֐�
	template <class T1, class T2>
	bool Equal(T1 v1, T2 v2)
	{
		const auto dmin = std::numeric_limits<std::common_type<T1, T2>::type>::min();

		return std::abs(v1 - v2) < dmin;
	}

	//�\�[�g�O��index��ێ����ă\�[�g
	template <class T>
	std::vector< std::tuple<uint, T> > SortWithIndex(std::vector<T> const& vec, bool const small_to_large)
	{
		std::vector< std::tuple<uint, T> > result(vec.size());

		for (uint i = 0; i < vec.size(); ++i){
			std::get<0>(result[i]) = i;
			std::get<1>(result[i]) = vec[i];
		}
		if (small_to_large) std::sort(result.begin(), result.end(), [](std::tuple<uint, T> const& a, std::tuple<uint, T> const& b){ return std::get<1>(a) < std::get<1>(b); });
		else std::sort(result.begin(), result.end(), [](std::tuple<uint, T> const& a, std::tuple<uint, T> const& b){ return std::get<1>(b) < std::get<1>(a); });

		return std::move(result);
	}

	//�R���e�i�̗v�f���V���b�t��
	template <class T, template < class T, class = std::allocator<T >> class Container>
	void Shuffle(Container<T>& data)
	{
		static SimpleRandom<double> myrand(0.0, 1.0, false);
		std::random_shuffle(data.begin(), data.end(), [&](uint max)->uint{ return myrand() * max; });
	}

	//2�̃R���e�i�̗v�f��Ή������Ȃ���\�[�g
	template < class T1, class T2, template < class T, class = std::allocator < T >> class Container1, template < class T, class = std::allocator<T >> class Container2>
	void Shuffle(Container1<T1>& data1, Container2<T2>& data2)
	{
		uint size = std::min(data1.size(), data2.size());
		auto rnum = RandomUniqueNumbers(size, 0, size-1, false);
		auto copy1 = std::move(data1);
		auto copy2 = std::move(data2);

		data1.resize(copy1.size());
		data2.resize(copy2.size());
		for (uint i=0; i<size; ++i){
			data1[rnum[i]] = std::move(copy1[i]);
			data2[rnum[i]] = std::move(copy2[i]);
		}
	}

#ifdef ENABLE_BOOST

	//�����ɍœK�Ȓl�Ƃ���Index��T��.�@comp(��r�Ώےl, �b��ŏ��l)
	template < class T, class CMP, template < class T, class = std::allocator<T>> class Container >
	maybe<std::tuple<T, uint>> SearchIndex(Container<T> const& src, CMP comp)
	{
		if (src.empty()) return nothing;
		
		T val = src[0];
		uint index = 0;

		for (uint i = 0, size = src.size(); i < size; ++i){
			if (comp(src[i], val)){
				val = src[i];
				index = i;
			}
		}

		return std::make_tuple(val, index);
	}

#endif

	//�R���e�i�ւ̑�����Z ([a], [b], (a -> b -> a))
	template < class T1, class T2, template < class T, class = std::allocator<T>> class Container>
		void CompoundAssignment(Container<T1>& list1, Container<T2> const& list2, std::function<typename std::common_type<T1>::type(typename std::common_type<T1>::type, typename std::common_type<T2>::type)> const& op)
	{
		const uint length = list1.size() < list2.size() ? list1.size() : list2.size();

		for (uint i = 0; i < length; ++i) list1[i] = op(list1[i], list2[i]);
	}

	//�R���e�i�ւ̑�����Z ([a], b, (a -> b -> a))
	template < class T1, class T2, template < class T, class = std::allocator<T>> class Container>
		void CompoundAssignment(Container<T1>& list1, T2 const& v, std::function<typename std::common_type<T1>::type(typename std::common_type<T1>::type, typename std::common_type<T2>::type)> const& op)
	{
		for (uint i = 0, length = list1.size(); i < length; ++i) list1[i] = op(list1[i], v);
	}

	//�l���w���������
	template < class T, template < class T, class = std::allocator<T >> class Container = std::vector>
		Container<T> Fill(T const& value, uint count)
	{
		Container<T> tmp;
		tmp.reserve(count);
		for (uint i = 0; i < count; ++i) tmp.push_back(value);
		return std::move(tmp);
	}

	//�����֐���ʂ��Ēl�𐶐�����
	//args -> generator: �����֐�.�����̓��[�vindex
	template < class T, template < class T, class = std::allocator<T >> class Container = std::vector>
		Container<T> Generate(std::function<T(int)> const& generator, uint count)
	{
		Container<T> tmp;
		tmp.reserve(count);
		for (uint i = 0; i < count; ++i) tmp.push_back(generator(i));
		return std::move(tmp);
	}

/* �֗��A�C�e�� */

#ifdef ENABLE_BOOST
	//�p�[�Z���g�^
	class Percent
	{
		int _percent;

	public:
		explicit Percent(int percent) : _percent(percent){}

		int GetPercent() const{ return _percent; }
		double GetDouble() const{ return _percent * 0.01; }

		static Percent const& Unit(){ static const Percent unit(100); return unit; }

		Percent operator=(Percent src){ _percent = src._percent; return *this; }
		Percent operator=(int src){ _percent = src; return *this; }
		
		bool operator==(Percent obj) const{ return _percent == obj._percent; }

		bool operator!=(Percent obj) const{ return _percent != obj._percent; }

	private:
		friend class boost::serialization::access;

		template <class Archive>
		void serialize(Archive& ar, unsigned int version)
		{
			ar & _percent;
		}

		template <class Archive>
		friend void save_construct_data(Archive & ar, Percent const* p, unsigned int version){};

		template <class Archive>
		friend void load_construct_data(Archive & ar, Percent* p, unsigned int version){
			::new(p) Percent(0);
		}
	};

#endif

	//�^�C���E�H�b�`
	class TimeWatch{
		typedef std::chrono::system_clock::time_point TIME;
		TIME st, ed;
	
		void Init(){
			ed = std::chrono::system_clock::now();
			st = std::chrono::system_clock::now();
		}

	public:
		TimeWatch(){ Init(); }

		void ReStart(){ Init(); }

		void Stop(){
			ed = std::chrono::system_clock::now();
		}
		
		//template<class TimeUnit = std::chrono::milliseconds> VC++2012 ���Ή�
		template<class TimeUnit>
		long long GetTime(){
			return std::chrono::duration_cast<TimeUnit>(ed - st).count();
		}
	};
		
	//�q�X�g�O����
	//template <�v�f�̌^, �x��>
	template <class T, size_t BIN_NUM>
	class Histgram{
		T const _min;
		T const _max;
		T const _delta;
		std::array<uint, BIN_NUM+2> _count;	//[0]: x < min, [BIN_NUM-1]: max <= x
		uint _num;

	public:
		//�v�f�͈̔͂��w��
		Histgram(T min, T max) : _min(min), _max(max), _delta(static_cast<int>((max-min+1)/BIN_NUM)), _num(0){
			assert(_delta > 0); 
			for(auto& ct : _count) ct = 0;
		}

		//�v�f��bin�ɐU�蕪���ăJ�E���g
		void Count(T value){
			for (uint i = 0; i < BIN_NUM + 1; ++i){
				if(value < _delta*i + _min){
					++_num;
					++_count[i];
					return;
				}
			}
			++_count[BIN_NUM + 1];
		}

		template <template<class TT, class = std::allocator<TT>> class Container>
		void Count(Container<T> const& values){
			for (auto e : values) Count(e);
		}

		//bin�O�̗v�f�����݂�����
		bool IsOverRange() const{ return _count[0] || _count[BIN_NUM+1]; }

		//double GetAverage() const{ return std::accumulate(_count.begin(), _count.end(), 0, [](T total, T next){ return total + next; }) / static_cast<double>(_num); }

		//�p�x���擾
		std::array<uint, BIN_NUM> GetCount() const{
			std::array<uint, BIN_NUM> tmp;
			for(uint i=0; i<BIN_NUM; ++i) tmp[i] = _count[i+1];
			return std::move(tmp);
		}

#ifdef ENABLE_BOOST
		//bin�Ԗ�(0 �` BIN_NUM-1)�̕p�x���擾
		//return -> tuple<�p�x, �͈͍ŏ��l(�ȏ�), �͈͍ő�l(����)>
		maybe<std::tuple<uint,int,int>> GetCount(uint bin) const{ return bin < BIN_NUM ? maybe<std::tuple<uint,int,int>>(std::make_tuple(_count[bin+1], _delta*bin+_min, _delta*(bin+1)+_min)) : nothing; }
#else
		std::tuple<uint,int,int> GetCount(uint bin) const{ return bin < BIN_NUM ? std::make_tuple(_count[bin+1], _delta*bin+_min, _delta*(bin+1)+_min) : throw std::out_of_range("Histgram::Get, bin=" + std::to_string(bin)); }
#endif

		void Print() const{
			int const keta = log10(_max) + 1;
			int const ctketa = log10(*std::max_element(_count.begin(), _count.end())) + 1;
			T const dbar = _num < 100 ? 1.0 : _num*0.01;

			std::string offset1, offset2;
			if(keta < 3) offset1.append(3-keta, ' ');
			else offset2.append(keta-3, ' ');

			std::cout << "-- Histgram --" << std::endl;
			for(int i=0; i<BIN_NUM+2; ++i){
				if (i == 0) std::cout << std::endl << "[-��," << std::setw(keta) << _delta*i + _min << ")" << offset2 << "�F" << std::setw(ctketa) << _count[i] << " ";
				else if (i == BIN_NUM + 1) std::cout << std::endl << "[" << std::setw(keta) << _delta*(i - 1) + _min << ",+��)" << offset2 << "�F" << std::setw(ctketa) << _count[i] << " ";
				else std::cout << std::endl << "[" << std::setw(keta) << _delta*(i - 1) + _min << "," << std::setw(keta) << _delta*i + _min << ")" << offset1 << "�F" << std::setw(ctketa) << _count[i] << " ";
				
				
				for(int j=1; dbar*j <= _count[i] ; ++j) printf("|");
			}
		}
	};

/* �C���E�␳�E�ǉ��E�폜 */

	//�͈̓`�F�b�N�Ǝ����C��
	template <class T, class U>
	inline bool ModifyRange(T& val, U const& min, U const& max)
	{
		if(val<min){ val = min; return false; }
		if(val>max){ val = max; return false; }
		return true;
	}

	template <class T, class U>
	inline bool CheckRange(T const& val, U const& min, U const& max)
	{
		if(val<min){ return false; }
		if(val>max){ return false; }
		return true;
	}

	// �폜�֘A�̊֐��Q
	namespace Eraser{

		#define Sig_Eraser_RemoveDuplicates_Impl(Container) \
			auto end = std::unique(list.begin(), list.end());\
			auto removes = need_removes ? Container(end, list.end()) : Container();\
			list.erase(end, list.end())\


		//�R���e�i�̗v�f����d���������̂��폜
		//args -> list: �R���e�i, need_removes: �폜�����v�f��߂�l�Ŏ󂯎�邩, is_sorted: �R���e�i���\�[�g�ς݂� 
		//return -> �폜�v�f
		template <class T, template<class T, class = std::allocator<T>> class Container>
		inline Container<T> RemoveDuplicates(Container<T>& list, bool need_removes, bool is_sorted = false)
		{
			if (!is_sorted) std::sort(list.begin(), list.end());

			Sig_Eraser_RemoveDuplicates_Impl(Container<T>);

			return std::move(removes);
		}
		template < class T>
		inline std::list<T> RemoveDuplicates(std::list<T>& list, bool need_removes, bool is_sorted = false)
		{
			if (!is_sorted) list.sort();

			Sig_Eraser_RemoveDuplicates_Impl(std::list<T>);

			return std::move(removes);
		}
	
		#ifdef ENABLE_BOOST
		#define Sig_Eraser_ParamType1 typename boost::call_traits<T>::param_type
		#else
		#define Sig_Eraser_ParamType1 typename std::common_type<T>::type const&
		#endif

		//�R���e�i����w��v�f��1�폜
		//args -> list: �R���e�i, remove: �폜�v�f
		//return -> �폜�v�f�����݂�����
		template <class T, template<class T, class = std::allocator<T>> class Container >
		inline bool RemoveOne(Container<T>& list, Sig_Eraser_ParamType1 remove)
		{
			for(auto it =list.begin(), end = list.end(); it != end;){
				if(*it == remove){
					list.erase(it);
					return true;
				}
				else ++it;
			}
			return false;
		}

		//�R���e�i����q������𖞂����v�f��1�폜
		//args -> list: �R���e�i, remove_pred: �폜���ʊ֐�
		//return -> �폜�v�f�����݂�����
		template <class Pred, class T, template<class T, class = std::allocator<T>> class Container >
		inline bool RemoveOneIf(Container<T>& list, Pred remove_pred)
		{
			for(auto it =list.begin(), end = list.end(); it != end;){
				if(remove_pred(*it)){
					list.erase(it);
					return true;
				}
				else ++it;
			}
			return false;
		}

		//�R���e�i����w��v�f��S�폜
		//args -> list: �R���e�i, remove: �폜�v�f
		//return -> �폜�v�f�����݂�����
		template < class T, template < class T, class = std::allocator<T >> class Container >
		inline bool RemoveAll(Container<T>& list, Sig_Eraser_ParamType1 remove)
		{
			uint presize = list.size();
			if (!list.empty()) list.erase(std::remove(list.begin(), list.end(), remove), list.end());
			return presize != list.size();
		}

		//�R���e�i����q������𖞂����v�f��S�폜
		//args -> list: �R���e�i, remove_pred: �폜���ʊ֐�
		//return -> �폜�v�f�����݂�����
		template <class Pred, class T, template<class T, class = std::allocator<T>> class Container >
		inline bool RemoveAllIf(Container<T>& list, Pred remove_pred)
		{
			uint presize = list.size();
			if(!list.empty()) list.erase( std::remove_if(list.begin(), list.end(), remove_pred), list.end());
			return presize != list.size();
		}
	}

/* �����񏈗� */
	namespace String{

	#ifdef ENABLE_BOOST

		//HTML���Ƀ^�O���G���R�[�h�E�f�R�[�h����
		//��F�@<TAG>text<TAG>
		template <class String>
		class TagText
		{
			const String _tel;
			const String _ter;

		public:
			TagText(String tag_encloser_left, String tag_encloser_right) : _tel(tag_encloser_left), _ter(tag_encloser_right){};

			String Encode(String const& src, String const& tag) const{
				auto tag_str = _tel + tag + _ter;
				return tag_str + src + tag_str;
			}

			template < template < class T, class Allocator = std::allocator<T>> class Container = std::vector >
			String Encode(Container<String> const& src, Container<String> const& tag);

			maybe<String> Decode(String const& src, String const& tag) const{
				auto tag_str = _tel + tag + _ter;
				auto parse = Split(src, tag_str, false);
				return parse.empty() ? nothing : maybe<String>(parse[1]);
			}

			template < template < class T, class Allocator = std::allocator<T >> class Container = std::vector >
			maybe<Container<String>> Decode(String const& src, Container<String> const& tag);
		};

		template <class String>
		template < template < class T, class Allocator = std::allocator<T >> class Container = std::vector >
		String TagText<String>::Encode(Container<String> const& src, Container<String> const& tag)
		{
			String result;
			auto size = std::min( src.size(), tag.size());
			for (uint i = 0; i < size; ++i){
				result += Encode(src[i], tag[i]);
			}
			return result;
		}

		template <class String>
		template < template < class T, class Allocator = std::allocator<T >> class Container = std::vector >
		maybe<Container<String>> TagText<String>::Decode(String const& src, Container<String> const& tag)
		{
			Container<String> result;
			for (auto const& e : tag){
				if(auto d = Decode(src, e)) result.push_back(*d);
			}
			return result.empty() ? nothing : maybe<Container<String>>(std::move(result));
		}
	
	#endif

		//����������镶����ڈ�ɕ�������
		template < class String, template<class T, class Allocator = std::allocator<T>> class Container = std::vector >
		Container<String> Split(String src, typename std::common_type<String>::type const& delim, bool ignore_blank = true)
		{
			Container<String> result;
			int const mag = delim.size();
			int cut_at;

			while( (cut_at = src.find(delim)) != src.npos ){
				 if(!ignore_blank || cut_at > 0) result.push_back(src.substr(0, cut_at));
				 src = src.substr(cut_at + mag);
			}
			if(!ignore_blank || src.length() > 0){
				 result.push_back(src);
			}

			return std::move(result);
		}

		template < template<class STRING, class Allocator = std::allocator<STRING>> class Container = std::vector >
		Container<std::string> Split(char const* src, char const* delim)
		{
			return Split<std::string, Container>(std::string(src), delim);
		}

		template < template < class STRING, class Allocator = std::allocator<STRING >> class Container = std::vector >
		Container<std::wstring> Split(wchar_t const* src, wchar_t const* delim)
		{
			return Split<std::wstring, Container>(std::wstring(src), delim);
		}

	/*
	#ifdef ENABLE_BOOST

		//�R���e�i�Ɋi�[���ꂽ�S���������������1�̕������(delimiter�ŋ�؂�w��)
		template < class T, template < class T, class = std::allocator<T >> class Container >
		inline std::string CatStr(Container<T> const& container, std::string delimiter = "")
		{
			std::string tmp;

			for (auto const& src : container){
				tmp += (boost::format("%1%%2%") % src % delimiter).str();
			}
			return std::move(tmp);
		}
	#endif
	*/
	
		//�R���e�i�Ɋi�[���ꂽ�S���������������1�̕������(delimiter�ŋ�؂�w��)
		template < class T, template < class T, class = std::allocator<T >> class Container >
		inline std::string CatStr(Container<T> const& container, std::string delimiter = "")
		{
			std::ostringstream ostream;

			for (auto const& src : container){
				ostream << src << delimiter;
			}
			return ostream.str();
		}

		template < class T, template < class T, class = std::allocator<T >> class Container >
		inline std::wstring CatWStr(Container<T> const& container, std::wstring delimiter = L"")
		{
			std::wostringstream ostream;

			for (auto const& src : container){
				ostream << src << delimiter;
			}
			return ostream.str();
		}

		template <class STRING>
		struct Map2Regex{
			typedef void type;
		};
		template <>
		struct Map2Regex<std::string>{
			typedef std::regex type;
		};
		template <>
		struct Map2Regex<std::wstring>{
			typedef std::wregex type;
		};
	
		template <class STRING>
		struct Map2Smatch{
			typedef void type;
		};
		template <>
		struct Map2Smatch<std::string>{
			typedef std::smatch type;
		};
		template <>
		struct Map2Smatch<std::wstring>{
			typedef std::wsmatch type;
		};


		//expression�Ɋ܂܂�镶���Ɋւ��āA���K�\���̓��ꕶ�����G�X�P�[�v����
		template <class String>
		String RegexEscaper(String expression)
		{
			static const std::wregex escape_reg(L"([(){}\\[\\]|^?$.+*\\\\])");
			return std::regex_replace(expression, escape_reg, L"\\$1");
		}

		template <class String>
		typename Map2Regex<String>::type RegexMaker(String const& expression)
		{
			return typename Map2Regex<String>::type(RegexEscaper(expression));
		}

	#ifdef ENABLE_BOOST

		//std::regex_search �̃��b�p�֐��B
		//return -> maybe ? [�}�b�`�����ӏ��̏���][�}�b�`���̎Q�Ƃ̏���. 0�͑S��, 1�ȍ~�͎Q�Ɖӏ�] : nothing
		//��F
		//src = "test tes1 tes2"
		//expression = std::regex("tes(\\d)")
		//return -> [[tes, 1], [tes, 2]]
		template < class String, template<class T, class = std::allocator<T>> class Container = std::vector >
		maybe< Container< Container<String>>> RegexSearch(String src, typename Map2Regex<String>::type expression)
		{
			Container<Container<String>> d;
			maybe<Container<Container<String>>> result(d);
			typename Map2Smatch<String>::type match;

			while( std::regex_search(src, match, expression) ){
				result->push_back(Container<String>());
				for(auto const& m : match) (*result)[result->size()-1].push_back(m);
				src = match.suffix().str();
			}

			return result->empty() ? nothing : std::move(result);
		}

		//expression�Ɋ܂܂�镶���Ɋւ��āA���K�\���̓��ꕶ�����G�X�P�[�v���Ă��珈���i�����j
		template < class String, template < class T, class = std::allocator<T >> class Container = std::vector >
		maybe< Container< Container<String>>> RegexSearch(String src, String expression)
		{		
			return RegexSearch(src, RegexMaker(expression));
		}

	#endif

		//UTF-16 to Shift-JIS
		inline std::string WSTRtoSTR(const std::wstring &src)
		{
			size_t mbs_size = src.length() * MB_CUR_MAX + 1;
			if(mbs_size < 2 || src == L"\0") return std::string();
			char *mbs = new char[mbs_size];
			size_t num;

			wcstombs_s(&num, mbs, mbs_size, src.c_str(), src.length() * MB_CUR_MAX + 1);
			std::string dest(mbs);
			delete [] mbs;

			return std::move(dest);
		}

		inline std::vector<std::string> WSTRtoSTR(std::vector<std::wstring> const& strvec)
		{
			std::vector<std::string> result;
			for(auto const& str : strvec) result.push_back( WSTRtoSTR(str) );
			return std::move(result);
		}

		//Shift-JIS to UTF-16
		inline std::wstring STRtoWSTR(const std::string &src)
		{
			size_t wcs_size = src.length() + 1;
			if(wcs_size < 2|| src == "\0") return std::wstring();
			//std::cout << src << std::endl;
			wchar_t *wcs = new wchar_t[wcs_size];
			size_t num;

			mbstowcs_s(&num, wcs, wcs_size, src.c_str(), src.length() + 1);
			std::wstring dest(wcs);
			delete [] wcs;

			return std::move(dest);
		}

		inline std::vector<std::wstring> STRtoWSTR(std::vector<std::string> const& strvec)
		{
			std::vector<std::wstring> result;
			for(auto const& str : strvec) result.push_back( STRtoWSTR(str) );
			return std::move(result);
		}
	}
	
/* �W������ */

	//vector, list �̐ϏW�������߂�(�v�f����1��). [����v���FT::operator==()]
	template <class T, template<class T, class = std::allocator<T>> class Container>
	Container<T> SetIntersection(Container<T> const& src1, Container<T> const& src2)
	{
		Container<T> result;

		for(T const& e1 : src1){
			for(T const& e2 : src2){
				if(e1 == e2 && [&result, &e2]()->bool{
					for(T const& r : result){
						if(r == e2) return false;
					}
					return true;
				}()){
					result.push_back(e2);
				}
			}
		}
		return move(result);
	}

	//unordered_set �̐ϏW�������߂�.[����v���FT::operator==()]
	template <class T>
	std::unordered_set<T> SetIntersection(std::unordered_set<T> const& src1, std::unordered_set<T> const& src2)
	{
		std::unordered_set<T> result;

		for(T const& e1 : src1){
			for(T const& e2 : src2){
				if(e1 == e2) result.insert(e2);
			}
		}
		return move(result);
	}

	//unordered_map �̐ϏW�������߂�(bool key ? �L�[�Ŕ�r�y��1�����̗v�f���擾�z : ������v). [����v���FK::operator==(), V::operator==()]
	template <class K, class V>
	std::unordered_map<K,V> SetIntersection(std::unordered_map<K,V> const& src, std::unordered_map<K,V> const& other, bool const key)
	{
		std::unordered_map<K,V> result;

		for(auto const& e : src){
			for(auto const& o : other){
				if(key && e.first == o.first) result.insert(e);
				else if(e == o) result.insert(e);
			}
		}
		return move(result);
	}


	//vector, list �̍��W�������߂�(�v�f����1��). [����v���FT::operator==()]
	template <class T, template<class T, class = std::allocator<T>> class Container>
	Container<T> SetDifference(Container<T> const& src1, Container<T> const& src2)
	{
		Container<T> result, sum(src1);
		sum.insert(sum.end(), src2.begin(), src2.end());

		auto intersection = SetIntersection(src1, src2);

		for(T const& s : sum){
			if([&intersection, &s]()->bool{
				for(T const& i : intersection){
					if(s == i) return false;
				}
				return true;
			}() && [&result, &s]()->bool{
				for(T const& r : result){
					if(s == r) return false;
				}
				return true;
			}()	){
				result.push_back(s);
			}
		}
		return move(result);
	}

	//unordered_set �̍��W�������߂�.[����v���FT::operator==()]
	template <class T>
	std::unordered_set<T> SetDifference(std::unordered_set<T> const& src1, std::unordered_set<T> const& src2)
	{
		std::unordered_set<T> result, sum(src1);
		sum.insert(src2.begin(), src2.end());

		auto intersection = SetIntersection(src1, src2);

		for(T const& s : sum){
			if([&intersection, &s]()->bool{
				for(T const& i : intersection){
					if(s == i) return false;
				}
				return true;
			}()){
				result.insert(s);
			}
		}
		return move(result);
	}

	//unordered_map �̍��W�������߂�(bool key ? �L�[�Ŕ�r : ������v). [����v���FK::operator==(), V::operator==()]
	template <class K, class V>
	std::unordered_map<K,V> SetDifference(std::unordered_map<K,V> const& src1, std::unordered_map<K,V> const& src2, bool const key)
	{
		std::unordered_map<K,V> result, sum(src1);
		sum.insert(src2.begin(), src2.end());

		auto intersection = SetIntersection(src1, src2, key);

		for(auto const& s : sum){
			if([&intersection, &s, key]()->bool{
				for(auto const& i : intersection){
					if(key && s.first == i.first) return false;
					else if(!key && s == i) return false;
				}
				return true;
			}()){
				result.insert(s);
			}
		}
		return move(result);
	}



/* ���o�� */
	namespace File{

		//�f�B���N�g���E�t�@�C���p�X�̖�����'/'or'\\'�����邩�`�F�b�N���A�t���邩�O�����ǂ����w��
		std::wstring DirpassTailModify(std::wstring const& directory_pass, bool const has_slash)
		{
			if(directory_pass.empty()) return directory_pass;

			auto tail = directory_pass.back();

			if(has_slash){
				//�t����ꍇ
				if(tail == '/' || tail == '\\') return directory_pass;
				else return (directory_pass + L"/");
			}
			else{
				if(tail != '/' && tail != '\\') return directory_pass;
				else{
					auto tmp = directory_pass;
					tmp.pop_back();
					return tmp;
				}
			}

		};


		//�w��f�B���N�g���ɂ���t�@�C�������擾
		//args -> empty_dest: ��̃R���e�i, directry_pass: �ǂݍ��ݐ�̃t�H���_�p�X
		//�ǂݍ��ݎ��s: throw(std::invalid_argument)
		inline void GetFileNames(std::vector<std::wstring>& empty_dest, std::wstring const& directory_pass) throw(std::invalid_argument)
		{
			WIN32_FIND_DATA fd;
			auto pass = DirpassTailModify(directory_pass, true) + L"*.*";
			auto hFind = FindFirstFile(pass.c_str(), &fd);

			if (hFind == INVALID_HANDLE_VALUE){
				throw std::invalid_argument("error: invalid directory_pass");
			}
			else{
				do{
					//�t�H���_�͖���
					if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && !(fd.dwFileAttributes & FILE_ATTRIBUTE_HIDDEN))
					{
						empty_dest.push_back(std::wstring(fd.cFileName));
					}
				} while (FindNextFile(hFind, &fd));

				FindClose(hFind);
			}
		}

#ifdef ENABLE_BOOST

		//�w��f�B���N�g���ɂ���t�@�C�������擾
		//args -> directry_pass: �ǂݍ��ݐ�̃t�H���_�p�X
		//�ǂݍ��ݎ��s: return -> maybe == nothing
		inline maybe<std::vector<std::wstring>> GetFileNames(std::wstring const& directory_pass)
		{
			std::vector<std::wstring> tmp;
			try{
				GetFileNames(tmp, directory_pass);
			}
			catch (...){
				return nothing;
			}
			return tmp;
		}

#endif

		//�w��f�B���N�g���ɂ���t�@�C�������t���p�X�Ŏ擾
		//args -> empty_dest: ��̃R���e�i, directry_pass: �ǂݍ��ݐ�̃t�H���_�p�X 
		//�ǂݍ��ݎ��s: throw(std::invalid_argument)
		inline void GetFilePasses(std::vector<std::wstring>& empty_dest, std::wstring const& directory_pass) throw(std::invalid_argument)
		{
			auto pass = DirpassTailModify(directory_pass, true);
			std::vector<std::wstring> names;
			GetFileNames(names, pass);

			try{
				for (auto const& e : names) empty_dest.push_back(pass + e);
			}
			catch (...){
				throw std::invalid_argument("error: invalid directory_pass");
			}
		}

#ifdef ENABLE_BOOST

		//�w��f�B���N�g���ɂ���t�@�C�������t���p�X�Ŏ擾
		//args -> directry_pass: �ǂݍ��ݐ�̃t�H���_�p�X
		//�ǂݍ��ݎ��s: throw(std::invalid_argument)
		inline maybe<std::vector<std::wstring>> GetFilePasses(std::wstring const& directory_pass)
		{
			std::vector<std::wstring> tmp;
			try{
				GetFilePasses(tmp, directory_pass);
			}
			catch (...){
				return nothing;
			}
			return tmp;
		}

#endif

		//for type map
		template <class PASS_STRING> struct OFS_SELECTION{};
		template<> struct OFS_SELECTION<std::string>{
			typedef std::ofstream type;
		};
		template<> struct OFS_SELECTION<std::wstring>{
			typedef std::wofstream type;
		};

		template <class PASS_STRING> struct IFS_SELECTION{};
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

		//�t�@�C����1�s���ۑ�
		//args -> src: �ۑ��Ώ�, file_pass: �ۑ���̃f�B���N�g���ƃt�@�C�����i�t���p�X�j, open_mode: �㏑��(overwrite) or �ǋL(append)
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

		/*	inline void SaveLine(std::wstring const& src, std::wofstream& _ofs)
		{
		_ofs.imbue(std::locale("Japanese", LC_COLLATE));
		_ofs.imbue(std::locale("Japanese", LC_CTYPE));
		_ofs << src << std::endl;
		}

		inline void SaveLine(std::wstring const& src, std::wstring const& file_pass, std::wios::open_mode const open_mode = std::wios::out)
		{
		std::wofstream ofs(file_pass, open_mode);
		SaveLine(src, ofs);
		}

		inline void SaveLine(std::vector<std::wstring> const& src, std::wofstream& _ofs)
		{
		std::for_each(src.begin(), src.end(), [&](std::wstring const& line){
		_ofs << line << std::endl;
		} );
		}

		inline void SaveLine(std::vector<std::wstring> const& src, std::wstring const& file_pass, std::wios::open_mode const open_mode = std::wios::out)
		{
		std::wofstream _ofs(file_pass, open_mode);
		SaveLine(src, _ofs);
		}
		*/

		// Read Text

		template <class R, class Stream>
		inline void ReadLine(std::vector<R>& empty_dest, Stream& ifs, std::function< R(typename std::conditional<std::is_same<Stream, std::ifstream>::value, std::string, std::wstring>::type)> const& conv)
		{
			typename std::conditional<std::is_same<Stream, std::ifstream>::value, std::string, std::wstring>::type line;

			empty_dest.clear();
			//ifs.seekg(0, std::ios::beg).read(&empty_dest[0], static_cast<std::streamsize>(empty_dest.size()));
			while (ifs && getline(ifs, line)){ empty_dest.push_back(conv(std::move(line))); }
		}

		/*
		//�ǂݍ��ݎ��s: empty_dest.size() == 0
		inline void ReadLine(std::vector<std::string>& empty_dest, std::ifstream& _ifs)
		{
			empty_dest.clear();
			std::string line;
			while (_ifs && getline(_ifs, line)){ empty_dest.push_back(line); }
		}

		//�ǂݍ��ݎ��s: empty_dest.size() == 0
		template <class STRING>
		inline void ReadLine(std::vector<std::string>& empty_dest, STRING const& file_pass)
		{
			std::ifstream _ifs(file_pass);
			if (!_ifs){
				if (std::is_same<std::string, STRING>::value) printf("file open error: %s \n", file_pass);
				else wprintf(L"file open error: %s \n", file_pass);
				return;
			}
			ReadLine(empty_dest, _ifs);
		}
		*/

		//�ǂݍ��ݎ��s: empty_dest.size() == 0
		template <class String>
		inline void ReadLine(std::vector<String>& empty_dest, typename IFS_SELECTION<String>::type& ifs)
		{
			ReadLine<String>(empty_dest, ifs, [](String line){ return line; });
		}

		//�ǂݍ��ݎ��s: empty_dest.size() == 0
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

		//�ǂݍ��ݎ��s: return -> maybe == nothing
		template <class String>
		inline maybe<std::vector<String>> ReadLine(typename IFS_SELECTION<String>::type& ifs)
		{
			typedef std::vector<String> R;
			R tmp;
			ReadLine<String>(tmp, ifs);
			return tmp.size() ? maybe<R>(std::move(tmp)) : nothing;
		}

		//�ǂݍ��ݎ��s: return -> maybe == nothing
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
		/*
		//�ǂݍ��ݎ��s: return -> maybe == nothing
		inline maybe<std::vector<std::wstring>> ReadLine(std::wifstream& _ifs)
		{
			std::vector<std::wstring> tmp;
			ReadLine(tmp, _ifs);
			return tmp.size() ? maybe<std::vector<std::wstring>>(std::move(tmp)) : nothing;
		}

		//�ǂݍ��ݎ��s: return -> maybe == nothing
		inline maybe<std::vector<std::wstring>> ReadLine(std::wstring const& file_pass)
		{
			std::wifstream _ifs(file_pass);
			if (!_ifs){
				std::wcout << L"file open error: " << file_pass << std::endl;
				return nothing;
			}
			return ReadLine(_ifs);
		}
		*/

		// int ver
		template <class Num, class String = std::string>
		inline maybe<std::vector<Num>> ReadLineInt(std::wstring const& file_pass)
		{
			std::vector<Num> tmp;
			ReadLineInt<Num, String>(tmp, file_pass);
			return tmp.size() ? maybe<std::vector<Num>>(std::move(tmp)) : nothing;
		}

#endif

		//csv�ŕۑ�
		template <class Num>
		inline void SaveCSV(std::vector<std::vector<Num>> const& data, std::vector<std::string> const& row_names, std::vector<std::string> const& col_names, std::wstring const& out_fullpass)
		{
			std::ofstream ofs(out_fullpass);

			//first row: field name
			ofs << ",";
			for (uint i = 1; i < data[0].size() + 1; ++i){
				auto name = i-1 < col_names.size() ? col_names[i - 1] : "";
				ofs << i << ". " << name << ",";
			}
			ofs << "\n";

			//first col: field name
			for (uint j = 0; j < data.size(); ++j){
				auto name = j < row_names.size() ? row_names[j] : "";
				ofs << j+1 << ". " << name << ",";

				for (auto e : data[j]){
					ofs << e << ",";
				}
				ofs << "\n";
			}
		}
	}

/*	template < template<class T, class = std::allocator<T>> class Container >
	inline void Print(Container<std::string> const& container, char const* const delimiter = "\n")
	{
		std::copy(container.begin(), container.end(), std::ostream_iterator<std::string>(std::cout, delimiter));
	}

	template < template<class T, class = std::allocator<T>> class Container >
	inline void Print(Container<std::wstring> const& container, wchar_t const* const delimiter = L"\n")
	{
		std::copy(container.begin(), container.end(), std::ostream_iterator<std::wstring>(std::wcout, delimiter));
	}
*/

	inline void Print(std::string const& text, char const* const delimiter = "\n")
	{
		std::cout << text << delimiter;
	}

	inline void Print(std::wstring const& text, wchar_t const* const delimiter = L"\n")
	{
		std::wcout << text << delimiter;
	}

	template < class T, template < class T, class Allocator = std::allocator<T >> class Container, typename std::enable_if<!std::is_same<T, std::wstring>::value>::type*& = enabler>
	inline void Print(Container<T> const& container, char const* const delimiter = "\n")
	{
		std::copy(container.begin(), container.end(), std::ostream_iterator<T>(std::cout, delimiter));
	}

	template<template<class ...> class Container>
	inline void Print(Container<std::wstring> const& container, wchar_t const* const delimiter = L"\n")
	{
		std::copy(container.begin(), container.end(), std::ostream_iterator<std::wstring>(std::wcout, delimiter));
	}


}

#endif UTILITY