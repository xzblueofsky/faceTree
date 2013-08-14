#include "stdafx.h"
#include "gaborFpGenerator.h"

const double gaborFp::GaborFp::Sigma = 2*PI;
const double gaborFp::GaborFp::F = sqrt(2.0);

Mat gaborFp::getFpHistogram(const Mat& src){
	
	//创建一个方向是PI/4而尺度是3的gabor
	double Sigma = 2*PI;
	double F = sqrt(2.0);
	CvGabor *gabor1 = new CvGabor;
	gabor1->Init(PI/2, 5, Sigma, F); 
	//获得实部并显示它
	IplImage *kernel = cvCreateImage( cvSize(gabor1->get_mask_width(), gabor1->get_mask_width()), IPL_DEPTH_8U, 1);  
	kernel = gabor1->get_image(CV_GABOR_REAL);  
	cvNamedWindow("Gabor Kernel", 1);  
	cvShowImage("Gabor Kernel", kernel);  
	cvWaitKey(0);
	//载入图片并显示
	Mat img;
	cvtColor(src, img, CV_BGR2GRAY);
	imshow("grayImage",img);
	waitKey(0);
	//获取载入图像的gabor滤波响应的实部并且显示
	IplImage *reimg = cvCreateImage(cvSize(img.cols,img.rows), IPL_DEPTH_8U, 1);  
	gabor1->conv_img(&IplImage(img), reimg, CV_GABOR_REAL);  
	cvNamedWindow("Real Response", 1);  
	cvShowImage("Real Response",reimg);  
	cvWaitKey(0);
	cvDestroyWindow("Real Response");
	//获取载入图像的gabor滤波响应的模并且显示
	IplImage *magimg = cvCreateImage(cvSize(img.cols,img.rows), IPL_DEPTH_8U, 1);  
	gabor1->conv_img(&IplImage(img), reimg, CV_GABOR_MAG);  
	cvNamedWindow("Magnitude Response", 1);  
	cvShowImage("Magnitude Response",reimg);
	cvWaitKey(0);

	return img;
}

void gaborFp::GaborFp::getGaborImage(int thetaIndex, int scale, Mat &outPut){

	CvGabor *gabor1 = new CvGabor;

	gabor1->Init(thetaIndex*PI/4, scale, Sigma, F);

	Mat img;
	cvtColor(srcImage, img, CV_BGR2GRAY);

	IplImage *maImg = cvCreateImage(cvSize(img.cols,img.rows), IPL_DEPTH_8U, 1);  
	gabor1->conv_img(&IplImage(img), maImg, CV_GABOR_MAG);
	
	Mat(maImg).copyTo(outPut);
	downSampleGaborImage(outPut);
	delete gabor1;
	cvReleaseImage(&maImg);
}

void gaborFp::GaborFp::downSampleGaborImage(Mat &gaborImage){
	const static int rou = 32;
	resize(gaborImage,gaborImage,Size(gaborImage.rows/rou,gaborImage.cols/rou));
}

void gaborFp::GaborFp::getGaborImageList(){
	for ( int scale=0; scale<5; ++scale ){
		for ( int thetaIndex=0; thetaIndex<8; ++thetaIndex ){
			Mat gaborImageUnit;
			getGaborImage(thetaIndex,scale,gaborImageUnit);
			gaborImageList.push_back(gaborImageUnit);
		}
	}
}

Mat gaborFp::GaborFp::getGaborFp(){
	Mat gaborFp;
	for ( vector<Mat>::iterator it = gaborImageList.begin(); it != gaborImageList.end(); ++it ){
		gaborFp.push_back(it->reshape(1,1));
	}
	gaborFp = gaborFp.reshape(1,1);
	return gaborFp;
}