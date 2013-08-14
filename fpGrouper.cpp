#include "stdafx.h"
#include "fpGrouper.h"

Mat fpGrouper::getLbpFpGroupMatrix(const vector<string>& srcImageLocList){
	Mat lbpFpMatrix = Mat(0,255,CV_32FC1);
	static int count=0;
	for (vector<string>::const_iterator it = srcImageLocList.begin(); it != srcImageLocList.end(); ++it){
		Mat image = imread(*it);
		Mat lbpFp = lbpFp::getFpHistogram(image,256);
		lbpFpMatrix.push_back(lbpFp);
		cout<<"lbp item count: "<<count++<<endl;
	}
	return lbpFpMatrix;
}

Mat fpGrouper::getGaborFpGroupMatrix(const vector<string>& srcImageLocList){
	Mat gaborFpMatrix;
	gaborFpMatrix.create(0,640,CV_8UC1);
	static int count = 0;
	for ( vector<string>::const_iterator it = srcImageLocList.begin(); it != srcImageLocList.end(); ++it ){
		Mat image = imread(*it);
		gaborFp::GaborFp gaborFp_(image);
		gaborFpMatrix.push_back(gaborFp_.getGaborFp());
		image.release();
		cout<<"gabor item count:"<<count++<<endl;
	}
	return gaborFpMatrix;
}