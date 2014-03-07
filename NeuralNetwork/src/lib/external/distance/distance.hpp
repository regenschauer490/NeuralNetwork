/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_DISTANCE_H
#define SIG_DISTANCE_H

#include "../SigUtil/lib/functional.hpp"

#undef max
#undef min

namespace sigdm{

	//ミンコフスキー距離
	template <size_t P>
	struct MinkowskiDistance
	{
		template <class C1, class C2>
		double operator()(C1 const& vec1, C2 const& vec2){
			using T = typename std::common_type<sig::container_traits<C1>::value_type, sig::container_traits<C2>::value_type>::type;
			auto delta = sig::ZipWith([&](T val1, T val2){ return std::pow(sig::DeltaAbsabs(val1, val2), P); }, vec1, vec2);
			
			return std::pow(std::accumulate(std::begin(delta), std::end(delta), 0.0, std::plus<double>()), 1.0 / P);
		}
	};

	//マンハッタン距離
	using ManhattanDistance = MinkowskiDistance<1>;

	//ユークリッド距離
	using EuclideanDistance = MinkowskiDistance<2>;

	//キャンベラ距離
	struct CanberraDistance
	{
		template < class T, template < class T_, class = std::allocator <T_ >> class Container>
		static double f(Container<T> const& vec1, Container<T> const& vec2){
			return sig::Accumulate(
				sig::ZipWith<double>(vec1, vec2, [](T val1, T val2){ return static_cast<double>(abs(val1 - val2)) / (abs(val1) + abs(val2)); }),
				0,
				std::plus<double>()
				);
		}
	};

	//バイナリ距離
	struct BinaryDistance
	{
		template < template < class T_, class = std::allocator <T_ >> class Container>
		static double f(Container<int> const& vec1, Container<int> const& vec2){
			int ether = 0, both = 0;
			for (auto it1 = vec1.begin(), it2 = vec2.begin(), end1 = vec1.end(), end2 = vec2.end(); it1 != end1 && it2 != end2; ++it1, ++it2){
				if (*it1 == 1 && *it2 == 1) ++both;
				else if (*it1 == 1 || *it2 == 1) ++ether;
			}
			return static_cast<double>(ether) / (ether + both);
		}

		template < template < class T_, class = std::allocator <T_ >> class Container>
		static double f(Container<bool> const& vec1, Container<bool> const& vec2){
			int ether = 0, both = 0;
			for (auto it1 = vec1.begin(), it2 = vec2.begin(), end1 = vec1.end(), end2 = vec2.end(); it1 != end1 && it2 != end2; ++it1, ++it2){
				if (*it1 && *it2) ++both;
				else if (*it1 || *it2) ++ether;
			}
			return static_cast<double>(ether) / (ether + both);
		}
	};
}
#endif