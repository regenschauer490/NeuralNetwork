/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_UTIL_H
#define SIG_NN_UTIL_H

#include <algorithm>
#include <numeric>
#include <assert.h>
#include <vector>
#include <array>
#include <memory>
#include <functional>
#include <random>
#include <cmath>
#include <future>
#include <type_traits>

#include "info.hpp"
#include "external/SigUtil/lib/sigutil.hpp"
#include "external/SigUtil/lib/tool.hpp"
#include "external/SigUtil/lib/calculation.hpp"
#include "external/SigUtil/lib/array.hpp"

#if SIG_ENABLE_BOOST
#include <boost/call_traits.hpp>
#endif

namespace signn
{
	using sig::enabler;
	using sig::ParamType;


	/* ノード・エッジ間の連結操作関数 */

	template <class T>
	void Connect(NodePtr<T, DirectedEdge<T>> const& departure, NodePtr<T, DirectedEdge<T>> const& arrival, DEdgePtr<T> const& edge)
	{
		departure->AddOutEdge(edge);
		arrival->AddInEdge(edge);
		edge->SetNode(departure, arrival);
	}

	/*
	template <class T>
	void Disconnect(C_NodePtr<T, DirectedEdge<T>> const& departure, C_NodePtr<T, DirectedEdge<T>> const& arrival)
	{
		for (auto oit = departure->out_begin(), oend = departure->out_end(); oit != oend; ++oit){
			for (auto iit = arrival->in_begin(), iend = arrival->in_end(); iit != iend; ++iit){
				if (*oit == *iit){
					auto edge = *oit;
					departure->RemoveOutEdge(edge);
					arrival->RemoveInEdge(edge);
				}
			}
		}
	}
	*/

	//class Matrix はランダムアクセス(operator[])可能であることが条件
	template <class T, template <class T_, class = std::allocator<T_>> class C, class Matrix>
	void MakeLink(C<NodePtr<T, DirectedEdge<T>>>& data, Matrix const& adjacency)
	{
		const uint size = data.size();
		
		for(uint r=0; r<size; ++r){
			auto& departure = data[r];
			for(uint c=0; c<size; ++c){
				if(adjacency[r][c]){
					auto& arrival = data[c];
					Connect(departure, arrival, std::make_shared<DirectedEdge<T>>());
				}
			}
		}
	}
	
	//class Matrix はランダムアクセス(operator[])可能であることが条件
	template <class T, template <class T_, class = std::allocator<T_>> class C, class Matrix>
	void MakeLink(C<NodePtr<T, DirectedEdge<T>>>& layer1, C<NodePtr<T, DirectedEdge<T>>>& layer2, Matrix const& connection)
	{
		const uint size = sig::Min(layer1.size(), layer2.size());
		
		for(uint r=0; r<size; ++r){
			auto& departure = layer1[r];
			for(uint c=0; c<size; ++c){
				if(adjacency[r][c]){
					auto& arrival = layer2[c];
					Connect(departure, arrival, std::make_shared<DirectedEdge>(departure, arrival));
				}
			}
		}
	}

	/**/

	template <class T, template <class T_, class = std::allocator<T_>> class C>
		inline double SquareError(C<T> const& estimate, C<T> const& answer) {
		return std::inner_product(estimate.begin(), estimate.end(), answer.begin(), 0.0, std::plus<double>(), [](T v1, T v2){ return pow(v1 - v2, 2); });
	}

	template <class T, template <class T_, class = std::allocator<T_>> class C>
	double MeanSquareError(C<C<T>> const& estimates, C<C<T>> const& answers)
	{
		const uint dsize = std::min(estimates.size(), answers.size());
		double error = 0;

		for (uint i = 0; i < dsize; ++i){
			error += SquareError(estimates[i], answers[i]);
		}
		return error / dsize;
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
			for (; xs != xs_end && hs != hs_end; ++xs, ++hs, ++size){
				result += std::abs((*xs) - (*hs));
			}
			for (xs = xs_begin; xs != xs_end && hs != hs_end; ++xs, ++hs, ++size){
				result += std::abs((*xs) - (*hs));
			}
		}

		return result / size;
	}

	inline double GetRandNum(double min, double max){
		static auto random = sig::SimpleRandom<double>(min, max, DEBUG_MODE);
		return random();
	}

}
#endif
