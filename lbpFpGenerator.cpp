#include "stdafx.h"

#include "lbpFpGenerator.h"
using namespace cv;

Mat lbpFp::getFpHistogram(const Mat& src, int numPatterns){

	// initial values
	int radius = 1;
	int neighbors = 8;

	// matrices used
	Mat dst; // image after preprocessing
	Mat lbp; // lbp image

	cvtColor(src, dst, CV_BGR2GRAY);

	GaussianBlur(dst, dst, Size(7,7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
	// comment the following lines for original size
	resize(dst,dst,Size(), 0.5, 0.5);

	lbp::OLBP(dst, lbp); // use the original operator

	// now to show the patterns a normalization is necessary
	// a simple min-max norm will do the job...
	normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);

	Mat hist = lbp::histogram(dst,256);
	hist.convertTo(hist,CV_32FC1);
	int size = dst.rows * dst.cols;
	hist /= size;

	return hist;
}