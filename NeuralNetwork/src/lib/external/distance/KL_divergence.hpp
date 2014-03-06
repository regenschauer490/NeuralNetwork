/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_KL_DIVERGENCE_H
#define SIG_KL_DIVERGENCE_H

#include "../SigUtil/lib/sigutil.hpp"

namespace sigdm{

//KL情報量
//条件：distribution[i] は正の値 かつ 総和が 1
//値域：[0, ∞)
template<template<class T, class = std::allocator<T>> class Container>
inline maybe<double> KL_Divergence(Container<double> const& distribution1, Container<double> const& distribution2)
{
	if(distribution1.size() != distribution2.size()) return nothing;
	 
	const auto Log2 = [](double n)->double{ return log(n) / log(2); };

	const auto KL = [&](Container<double> const& d1, Container<double> const& d2)->double{
		double sum = 0;
		for(uint i=0; i<d1.size(); ++i){
			sum += d1[i] * ( Log2(d1[i]) - Log2(d2[i]) );
		}
		return sum;
	};

	return KL(distribution1, distribution2);
}

}
#endif