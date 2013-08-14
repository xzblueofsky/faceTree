#ifndef FP_RECORDER_H
#define FP_RECORDER_H

#include <cv.h>
#include <fstream>
using namespace cv;
using namespace std;

namespace fpRecorder{
	
	//templated functions
	template <typename _TP>
	void writeToText_(const Mat& src,const string& destLoc);

	//wrapper functions
	void writeToText(const Mat& src, const string& destLoc);


	int writeMatrixToText(const Mat& src, string destLoc);
}

#endif