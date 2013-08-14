#include "stdafx.h"

#include "fpRecorder.h"

template <typename _TP>
void fpRecorder::writeToText_(const Mat &srcFp, const string &destLoc){
	ofstream ofs;
	ofs.open(destLoc,ios::out|ios::app);

	for (int i=0; i<srcFp.cols; i++ ){
		ofs<<srcFp.at<_TP>(0,i)<<"	";
	}

	ofs<<endl;
	ofs.close();
}

//now the wrapper functions
void fpRecorder::writeToText(const Mat &srcFp,const  string &destLoc){
	switch(srcFp.type()){
		case CV_8SC1: writeToText_<char>(srcFp, destLoc); break;
		case CV_8UC1: writeToText_<unsigned char>(srcFp, destLoc); break;
		case CV_16SC1: writeToText_<short>(srcFp, destLoc); break;
		case CV_16UC1: writeToText_<unsigned short>(srcFp, destLoc); break;
		case CV_32SC1: writeToText_<int>(srcFp, destLoc); break;
		case CV_32FC1: writeToText_<float>(srcFp, destLoc); break;
		case CV_64FC1: writeToText_<double>(srcFp, destLoc); break;
	}
}

int fpRecorder::writeMatrixToText(const Mat& src, string destLoc){
	ofstream ofs;
	ofs.open(destLoc,ios::app);
	for ( int i=0; i<src.rows; i++){
		for ( int j=0; j<src.cols; j++){
			ofs<<src.at<int>(i,j)<<"	";
		}
		ofs<<endl;
	}
	ofs.close();
	return 0;
}