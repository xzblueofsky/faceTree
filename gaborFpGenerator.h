#ifndef GABOR_FP_GENERATOR_H
#define GABOR_FP_GENERATOR_H

#include <cv.h>
#include "histogram.hpp"
#include "cvgabor.h"

using namespace cv;
using namespace std;

namespace gaborFp{
	class GaborFp{
	private:
		Mat srcImage;
		const static double Sigma;
		const static double F;
		vector<Mat> gaborImageList;

		void getGaborImage(int thetaIndex,int scale, Mat& outPut);	//获得指定theta和scale的gabor滤波器卷积后的图像
		void getGaborImageList();	//获得全部gabor滤波器滤波后的图像序列0~8*PI/8,8方向；0~5,5个scale
		void downSampleGaborImage(Mat &gaborImage);	//rou=32降采样，将128*128图像将至4*4=16维。最终特征将是16*40=640维。
	public:
		GaborFp(Mat image):srcImage(image) { getGaborImageList(); }
		~GaborFp() {
			srcImage.release();
			for ( vector<Mat>::iterator it = gaborImageList.begin(); it != gaborImageList.end(); ++it ){
				it->release();
			}
			gaborImageList.clear();
		}

		Mat getGaborFp();	//获得Gabor特征，依次为8方向，5尺度的40张gabor滤波的图像，拼接成一个长向量作为特征
	};

	const double Sigma = 2*PI;
	const double F = sqrt(2.0);
	Mat getFpHistogram(const Mat& src);
}

#endif