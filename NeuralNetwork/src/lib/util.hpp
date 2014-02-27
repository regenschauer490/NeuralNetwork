/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIG_NN_UTIL_H
#define SIG_NN_UTIL_H

#include "external/SigUtil/lib/sigutil.hpp"

namespace signn
{

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
#endif
