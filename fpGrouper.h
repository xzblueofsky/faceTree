#ifndef FP_GROUPER_H
#define FP_GROUPER_H

#include <opencv\cxcore.h>
#include <opencv\highgui.h>
#include "lbpFpGenerator.h"
#include "gaborFpGenerator.h"

using namespace cv;

namespace fpGrouper{
	Mat getLbpFpGroupMatrix(const vector<string>& srcImageLocList);

	Mat getGaborFpGroupMatrix(const vector<string>& srcImageLocList);
}

#endif