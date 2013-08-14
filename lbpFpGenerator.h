#ifndef LBP_FP_GENERATOR_H
#define LBP_FP_GENERATOR_H

#include "lbp.hpp"
#include "histogram.hpp"

namespace lbpFp{

	Mat getFpHistogram(const Mat& src, int numPatterns);

}
#endif