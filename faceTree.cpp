// faceTree.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

#include "lbp.hpp"
#include "histogram.hpp"
#include "lbpFpGenerator.h"
#include <direct.h>
#include <io.h>
#include <fstream>
#include "fpRecorder.h"
#include "cvgabor.h"
#include "gaborFpGenerator.h"
#include "fpGrouper.h"
#include "global.h"
#include "zScoreObj.h"

extern "C"{
#include "cluster.h"
}
//#include <vld.h>	//检测内存泄露

using namespace std;
using namespace cv;

int getFileLocList( const string& dir, vector<string>& fileLocList );
void example_hierarchical(int nrows, int ncols, double** data, int** mask,double** distmatrix);
double** example_distance_gene(int nrows, int ncols, double** data, int** mask)
	/* Calculate the distance matrix between genes using the Euclidean distance. */
{ int i, j;
double** distMatrix;
double* weight = (double*)malloc(ncols*sizeof(double));
printf("============ Euclidean distance matrix between genes ============\n");
for (i = 0; i < ncols; i++) weight[i] = 1.0;
distMatrix = distancematrix(nrows, ncols, data, mask, weight, 'e', 0);
if (!distMatrix)
{ printf ("Insufficient memory to store the distance matrix\n");
free(weight);
return NULL;
}
printf("   Gene:");
for(i=0; i<nrows-1; i++) printf("%6d", i);
printf("\n");
for(i=0; i<nrows; i++)
{ printf("Gene %2d:",i);
for(j=0; j<i; j++) printf(" %5.2f",distMatrix[i][j]);
printf("\n");
}
printf("\n");
free(weight);
return distMatrix;
}

string zxitoa(int num){
	stringstream ss;
	ss<<num;
	string str;
	ss>>str;
	return str;
}

int _tmain(int argc, _TCHAR* argv[])
{
	cout<<"hello world"<<endl;
	string imageLoc = "H:/pic/lena.jpg";
	/*Mat image = imread(imageLoc);
	imshow("test",image);
	waitKey();*/

	//////////////////////////////////////////////////
	////	保存lbp特征单元测试
	//Mat lbpFpHist = lbpFp::getFpHistogram(image,256);
	//string loc = "D:/test.txt";
	//fpRecorder::writeToText(lbpFpHist,loc);

	//////////////////////////////////////////////////////
	////	抽取并保存全部图片的lbp特征至lbpFpList.txt
	//string srcDir = "E:/360/face/face_urls/";
	//vector<string> imageList;
	//getFileLocList(srcDir,imageList);

	//string lbpFpListLoc = "D:/lbpFpList.txt";
	//for ( vector<string>::iterator it = imageList.begin(); it !=imageList.end(); ++it){
	//	Mat image = imread(*it);
	//	Mat lbpFpHist = lbpFp::getFpHistogram(image,256);
	//	fpRecorder::writeToText(lbpFpHist,lbpFpListLoc);
	//}

	/////////////////////////////////////////////////////////////
	////	Gabor特征单元测试
	////创建一个方向是PI/4而尺度是3的gabor
	//double Sigma = 2*PI;
	//double F = sqrt(2.0);
	//CvGabor *gabor1 = new CvGabor; 
	//gabor1->Init(PI/2, 4, Sigma, F); 
	////获得实部并显示它
	//IplImage *kernel = cvCreateImage( cvSize(gabor1->get_mask_width(), gabor1->get_mask_width()), IPL_DEPTH_8U, 1);  
	//kernel = gabor1->get_image(CV_GABOR_REAL);  
	//cvNamedWindow("Gabor Kernel", 1);  
	//cvShowImage("Gabor Kernel", kernel);  
	//cvWaitKey(0);
	////载入图片并显示
	//Mat img;
	//cvtColor(image, img, CV_BGR2GRAY);
	//imshow("grayImage",img);
	//waitKey(0);
	////获取载入图像的gabor滤波响应的实部并且显示
	//IplImage *reimg = cvCreateImage(cvSize(img.cols,img.rows), IPL_DEPTH_8U, 1);  
	//gabor1->conv_img(&IplImage(img), reimg, CV_GABOR_REAL);  
	//cvNamedWindow("Real Response", 1);  
	//cvShowImage("Real Response",reimg);  
	//cvWaitKey(0);  
	//cvDestroyWindow("Real Response");
	////获取载入图像的gabor滤波响应的模并且显示
	//IplImage *magimg = cvCreateImage(cvSize(img.cols,img.rows), IPL_DEPTH_8U, 1);  
	//gabor1->conv_img(&IplImage(img), reimg, CV_GABOR_MAG);  
	//cvNamedWindow("Magnitude Response", 1);  
	//cvShowImage("Magnitude Response",reimg);
	//cvWaitKey(0);
	////这个响应可以被取样为8位的灰度图。如果你要原始的浮点类型的数据，你可以这样做
	//gabor1->conv_img(&IplImage(img), reimg, CV_GABOR_MAG);
	////然而，这些浮点数据是不能够以上面灰度图的形式简单的显示，但是它可以被保存在一个XML文件中。
	//cvSave( "reimg.xml", (IplImage*)reimg, NULL, NULL, cvAttrList(0,0)); 
	//cvDestroyWindow("Magnitude Response");  


	///////////////////////////////////////////////////////////
	//	20130804 begin
	///////////////////////////////////////////////////////////
	////	file rename 单元测试(已成功，不要再运行)
	//vector<string> imageFileList;
	//string imageDir = "H:/360/face_urls/";
	//getFileLocList(imageDir,imageFileList);
	//int index=0;
	//for(vector<string>::iterator it = imageFileList.begin(); it!=imageFileList.end(); ++it){
	//	index++;
	//	stringstream ss;
	//	ss<<index;
	//	string strIndex = ss.str();
	//	string newImageName = strIndex + ".jpg";
	//	rename((*it).c_str(),(imageDir+newImageName).c_str());
	//}

	//////////////////////////////////////////////////
	////	Mat数据结构测试
	//Mat test(Mat::ones(2,2,CV_8UC1));
	//test.convertTo(test,CV_32FC1);
	//test /= cvRound(3);
	//cout<<test<<endl;

	//////////////////////////////////////////////////
	////	保存lbp特征单元测试 ver 1.1
	//Mat lbpFpHist = lbpFp::getFpHistogram(image,256);
	//string loc = "D:/test.txt";
	//fpRecorder::writeToText(lbpFpHist,loc);

	////////////////////////////////////////////////////////
	////		generate image database lbp fp group matrix
	//string imageLocDir = "E:/360/face/face_urls/";
	//vector<string> imageLocList;
	//getFileLocList(imageLocDir,imageLocList);
	//Mat lbpFpGroupMatrix = fpGrouper::getLbpFpGroupMatrix(imageLocList);
	////	lbp fp group matrix to xml file
	//FileStorage fs("D:/lbpFpMatrix.xml",FileStorage::WRITE);
	//fs<<"lbpFpMatrix"<<lbpFpGroupMatrix;

	////	read fp group matrix
	//cout<<"begin"<<endl;
	//FileStorage fs("D:/lbpFpMatrix.xml",FileStorage::READ);
	//Mat lbpFpGroupMatrix;
	//cout<<"ready"<<endl;
	//fs["lbpFpMatrix"]>>lbpFpGroupMatrix;
	//cout<<"done"<<endl;
	//cout<<"rows:"<<lbpFpGroupMatrix.rows<<endl;
	//cout<<"cols:"<<lbpFpGroupMatrix.cols<<endl;
	//fs.release();

	//////////////////////////////////////////////////////////
	////	lbp PCA单元测试,取前16个最大的特征值，从256维降至16维，特征值比例80.12%
	//PCA pca(lbpFpGroupMatrix,Mat(),CV_PCA_DATA_AS_ROW,16);
	//cout<<"pca eigen values:"<<pca.eigenvalues<<endl;
	//fs.open("D:/lfgmEigenValues.xml",FileStorage::WRITE);
	//fs<<"lfgmEigenValues"<<pca.eigenvalues;
	//fs.release();

	//fs.open("D:/lfgmEigenVectors.xml",FileStorage::WRITE);
	//fs<<"lfgmEigenVectors"<<pca.eigenvectors;
	//fs.release();

	//// 特征值比例计算.step1:计算特征值的和
	//Mat tmpSum(1,1,pca.eigenvalues.type());
	//reduce(pca.eigenvalues,tmpSum,0,CV_REDUCE_SUM);
	//cout<<"tmpSum"<<tmpSum<<endl;
	//
	//fs.open("D:/eigenValuesSum.xml",FileStorage::WRITE);
	//fs<<"eigenValuesSum"<<tmpSum;
	//fs.release();
	////	特征值比例计算。step2:计算特征值所占比例
	//Mat eigenValueProp(1,pca.eigenvalues.cols,pca.eigenvalues.type());
	//eigenValueProp = pca.eigenvalues / tmpSum;
	//fs.open("D:/lfgmEigenValueProp.xml",FileStorage::WRITE);
	//fs<<"eigenValueProp"<<eigenValueProp;
	////	特征值比例计算。step3:计算特征值比例递增
	//Mat eigenValuePropAscend(eigenValueProp);
	//for (int i=1; i<eigenValuePropAscend.rows; i++ ){
	//	eigenValuePropAscend.at<float>(i,0) += eigenValuePropAscend.at<float>(i-1,0);
	//}
	//fs<<"eigenValuePropAscend"<<eigenValuePropAscend;
	//fs.release();

	////	计算pca映射后的特征矩阵,计算耗时很长，大约2小时左右
	//Mat lfgmProjected;
	//for ( int i=0; i<lbpFpGroupMatrix.rows; i++){
	//	pca.project(lbpFpGroupMatrix,lfgmProjected);
	//}
	//fs.open("D:/lfgmProjected.xml",FileStorage::WRITE);
	//fs<<"lfgmProjected"<<lfgmProjected;
	//fs.release();

	//////////////////////////////////////////////////////
	////	图像大小归一化到128*128
	//const string imageDir = "H:/360/face_urls/";
	//const string dstDir = "E:/360/face/face_urls_unified/";
	//const Size dstSize(128,128);
	//vector<string> imageLocList;
	//getFileLocList(imageDir,imageLocList);
	//for ( vector<string>::iterator it = imageLocList.begin(); it != imageLocList.end(); ++it ){
	//	Mat src = imread(*it);
	//	Mat dst;
	//	resize(src,dst,dstSize);
	//	imwrite(*it,dst);
	//}

	//////////////////////////////////////////////////////////////////
	////	GaborFpMatrix 生成单元测试
	//FileStorage fs;
	//const string imageDir = "H:/360/face_urls/";
	//vector<string> imageLocList;
	//getFileLocList(imageDir,imageLocList);
	//Mat gaborFpGroupMatrix;
	//gaborFpGroupMatrix = fpGrouper::getGaborFpGroupMatrix(imageLocList);
	//fs.open("H:/gaborFpGroupMatrix.xml",FileStorage::WRITE);
	//fs<<"gaborFpGroupMatrix"<<gaborFpGroupMatrix;
	//fs.release();

	///////////////////////////////////////////////////////////////////
	//	读入gabor特征矩阵
	//FileStorage fs;
	//fs.open(global::gabor::loc::fgmLoc,FileStorage::READ);
	//Mat gfgm;
	//fs[global::gabor::tag::fpGroupMatrix]>>gfgm;
	//fs.release();
	////	计算平均值
	//gfgm.convertTo(gfgm,CV_32F);
	//Mat gfgmMean;
	//reduce(gfgm,gfgmMean,0,CV_REDUCE_AVG);
	//fs.open(global::gabor::loc::meanLoc,FileStorage::WRITE);
	//fs<<global::gabor::tag::mean<<gfgmMean;
	//fs.release();
	////	gabor特征矩阵每一列都减去平均值
	//Mat meanMatrix;
	//for( int i=0; i<gfgm.rows; i++){
	//	meanMatrix.push_back(gfgmMean);
	//}
	//Mat test = gfgm - meanMatrix;
	//fs.open(global::test::loc::loc,FileStorage::WRITE);
	//cout<<global::test::loc::loc<<endl;
	//fs<<global::test::tag::tag<<test;
	//fs.release();

	////	计算gabor特征矩阵协方差
	//Mat gfgm;
	//FileStorage fs;
	//fs.open(global::gabor::loc::fgmLoc,FileStorage::READ);
	//fs[global::gabor::tag::fpGroupMatrix]>>gfgm;
	//gfgm.convertTo(gfgm,CV_32F);
	//Mat gfgmCov,mean;
	//calcCovarMatrix(gfgm,gfgmCov,mean,CV_COVAR_ROWS | CV_COVAR_NORMAL);
	//fs.open(global::gabor::loc::covarLoc,FileStorage::WRITE);
	//fs<<global::gabor::tag::covar<<gfgmCov;
	//fs.release();

	////	calculate and save normalization of gabor finger print group matrix 
	//Mat gfgm;	//decleration of 'Gabor Fingerprint Group Matrix'
	//FileStorage fs;
	//fs.open(global::gabor::loc::fgmLoc,FileStorage::READ);
	//fs[global::gabor::tag::fpGroupMatrix]>>gfgm;	//read data into gfgm
	//fs.release();
	//gfgm.convertTo(gfgm,CV_64F);

	//Mat covar;	//decleration of covariance matrix of gfgm
	//Mat mean; //decleration of mean of gfgm;
	//calcCovarMatrix(gfgm,covar,mean,CV_COVAR_ROWS | CV_COVAR_NORMAL);
	//covar /= static_cast<double>(gfgm.rows);
	//Mat stdDev;	//decleration of stdandard deviation of gfgm
	//stdDev.create(1,covar.cols,CV_64F);
	//for ( int i=0; i<covar.rows; i++){
	//	stdDev.at<double>(0,i) = sqrt(covar.at<double>(i,i));
	//	cout<<"calculate standard deviation of column "<<i<<endl;
	//}

	//Mat normGfgm;	// decleration of normalized gfgm
	//normGfgm.create(gfgm.rows,gfgm.cols,CV_64F);
	//gfgm.copyTo(normGfgm);
	//for ( int i=0; i<normGfgm.rows; i++ ){	//calculate normGfgm
	//	mean.convertTo(mean,normGfgm.type());
	//	normGfgm.row(i) -= mean;
	//	cout<<"substract mean from row "<<i<<endl;
	//}
	//for ( int i=0; i<normGfgm.cols; i++ ){
	//	normGfgm.col(i) /= stdDev.at<double>(0,i);	//calculate normGfgm
	//	cout<<"devide standard deviation from column "<<i<<endl;
	//}
	//
	//fs.open(global::gabor::loc::normGfgm,FileStorage::WRITE);	//save normGfgm
	//fs<<global::gabor::tag::normGfgm<<normGfgm;
	//fs.release();

	//////////////////////////////////////////////////////////////////
	////	LbpFpMatrix generation
	//FileStorage fs;
	//const string imageDir = "H:/360/face_urls/";
	//vector<string> imageLocList;
	//getFileLocList(imageDir,imageLocList);
	//Mat lbpFpGroupMatrix;
	//lbpFpGroupMatrix = fpGrouper::getLbpFpGroupMatrix(imageLocList);
	//fs.open(global::lbp::loc::fgmLoc,FileStorage::WRITE);
	//fs<<global::lbp::tag::fpGroupMatrix<<lbpFpGroupMatrix;
	//fs.release();

	////	calculate and save normalization of Lbp finger print group matrix 
	//Mat lfgm;	//decleration of 'Lbp Fingerprint Group Matrix'
	//FileStorage fs;
	//fs.open(global::lbp::loc::fgmLoc,FileStorage::READ);
	//fs[global::lbp::tag::fpGroupMatrix]>>lfgm;	//read data into lfgm
	//fs.release();
	//lfgm.convertTo(lfgm,CV_64F);

	//Mat covar;	//decleration of covariance matrix of lfgm
	//Mat mean; //decleration of mean of lfgm;
	//calcCovarMatrix(lfgm,covar,mean,CV_COVAR_ROWS | CV_COVAR_NORMAL);
	//covar /= static_cast<double>(lfgm.rows);
	//Mat stdDev;	//decleration of stdandard deviation of lfgm
	//stdDev.create(1,covar.cols,CV_64F);
	//for ( int i=0; i<covar.rows; i++){
	//	stdDev.at<double>(0,i) = sqrt(covar.at<double>(i,i));
	//	cout<<"calculate standard deviation of column "<<i<<endl;
	//}

	//Mat normlfgm;	// decleration of normalized lfgm
	//normlfgm.create(lfgm.rows,lfgm.cols,CV_64F);
	//lfgm.copyTo(normlfgm);
	//for ( int i=0; i<normlfgm.rows; i++ ){	//calculate normlfgm
	//	mean.convertTo(mean,normlfgm.type());
	//	normlfgm.row(i) -= mean;
	//	cout<<"substract mean from row "<<i<<endl;
	//}
	//for ( int i=0; i<normlfgm.cols; i++ ){
	//	normlfgm.col(i) /= stdDev.at<double>(0,i);	//calculate normlfgm
	//	cout<<"devide standard deviation from column "<<i<<endl;
	//}
	//
	//fs.open(global::lbp::loc::normLfgm,FileStorage::WRITE);	//save normlfgm
	//fs<<global::lbp::tag::normLfgm<<normlfgm;
	//fs.release();

	////test column concat
	//Mat test1 = Mat::zeros(2,2,CV_8UC1);
	//Mat test2 = Mat::ones(2,3,CV_8UC1);
	//cout<<"test1"<<test1<<endl;
	//cout<<"test2"<<test2<<endl;
	//
	//hconcat(test1,test2,test1);
	//cout<<"test1"<<test1<<endl;

	////gabor pca
	////640降至132维，80%能量,计算耗时8个小时
	//FileStorage fs;
	//fs.open(global::gabor::loc::normGfgm,FileStorage::READ);
	//Mat normGfgm;
	//fs[global::gabor::tag::normGfgm]>>normGfgm;
	//PCA pca(normGfgm,Mat(),CV_PCA_DATA_AS_ROW,132);
	//cout<<"pca eigen values:"<<pca.eigenvalues<<endl;

	//fs.open(global::gabor::pca::loc::eigenValues,FileStorage::WRITE);
	//fs<<global::gabor::pca::tag::eigenValues<<pca.eigenvalues;
	//fs.release();

	//fs.open(global::gabor::pca::loc::eigenVectors,FileStorage::WRITE);
	//fs<<global::gabor::pca::tag::eigenVectors<<pca.eigenvectors;
	//fs.release();

	//Mat gfgmProjected;
	//for ( int i=0; i<normGfgm.rows; i++){
	//	pca.project(normGfgm,gfgmProjected);
	//}
	//fs.open(global::gabor::pca::loc::projected,FileStorage::WRITE);
	//fs<<global::gabor::pca::tag::projected<<gfgmProjected;
	//fs.release();

	////lbp pca
	////256维降至10维，82%能量，计算耗时2小时
	//FileStorage fs;
	//fs.open(global::lbp::loc::normLfgm,FileStorage::READ);
	//Mat normLfgm;
	//fs[global::lbp::tag::normLfgm]>>normLfgm;
	//PCA pca(normLfgm,Mat(),CV_PCA_DATA_AS_ROW,10);
	//cout<<"pca eigen values:"<<pca.eigenvalues<<endl;

	//fs.open(global::lbp::pca::loc::eigenValues,FileStorage::WRITE);
	//fs<<global::lbp::pca::tag::eigenValues<<pca.eigenvalues;
	//fs.release();

	//fs.open(global::lbp::pca::loc::eigenVectors,FileStorage::WRITE);
	//fs<<global::lbp::pca::tag::eigenVectors<<pca.eigenvectors;
	//fs.release();

	//Mat lfgmProjected;
	//for ( int i=0; i<normLfgm.rows; i++){
	//	pca.project(normLfgm,lfgmProjected);
	//}
	//fs.open(global::lbp::pca::loc::projected,FileStorage::WRITE);
	//fs<<global::lbp::pca::tag::projected<<lfgmProjected;
	//fs.release();

	////generate z-score fingerprint matrix
	//Mat gaborPart,lbpPart;
	//FileStorage fs;
	//fs.open(global::gabor::pca::loc::projected,FileStorage::READ);
	//fs[global::gabor::pca::tag::projected]>>gaborPart;
	//fs.release();

	//fs.open(global::lbp::pca::loc::projected,FileStorage::READ);
	//fs[global::lbp::pca::tag::projected]>>lbpPart;
	//fs.release();

	//Mat zScore;
	//hconcat(gaborPart,lbpPart,zScore);
	//fs.open(global::z_score::loc::fileLoc,FileStorage::WRITE);
	//fs<<global::z_score::tag::tag<<zScore;
	//fs.release();

	////read zScore raw data from xml file
	// Mat zScore;
	// FileStorage fs;
	// fs.open(global::z_score::loc::fileLoc,FileStorage::READ);
	// fs[global::z_score::tag::tag]>>zScore;
	// fs.release();
	////read corresponding image name from image database folder
	// vector<string> imageLocList;
	// getFileLocList(global::dataBase::location,imageLocList);
	//// generage zScore object vector
	// vector<zScoreObj> zList;
	// for ( int i=0; i<imageLocList.size(); i++ ){
	//	zScoreObj zObj(imageLocList[i],zScore.row(i));
	//	zList.push_back(zObj);
	//	cout<<i<<"th zScoreObj is writed."<<endl;
	// }
	////	write zScore object vector to binary file
	// ofstream ofs;
	// ofs.open(global::z_score::loc::calcLoc,ios::binary);
	// int size1 = zList.size();
	// ofs.write((const char*)&size1,sizeof(int));
	// ofs.write((const char*)&zList[0],size1*sizeof(zScoreObj));
	// ofs.close();

	 /*vector<zScoreObj> zList1;
	 ifstream ifs;
	 ifs.open(global::z_score::loc::calcLoc,ios::binary);
	 int size2;
	 ifs.read((char*)&size2,sizeof(int));
	 zList1.resize(size2);
	 ifs.read((char*)&zList1[0],size2*sizeof(zScoreObj));
	 ifs.close();
	 for ( vector<zScoreObj>::iterator it = zList1.begin(); it!=zList1.end(); ++it){
		Mat image = imread(it->getImageLoc());
		imshow("test",image);
		waitKey();
	 }*/

	 vector<zScoreObj> zListTest;
	 ifstream ifs;
	 int size;
	 ifs.open(global::test::z_score::loc::calcLoc,ios::binary);
	 ifs.read((char*)&size,sizeof(int));
	 zListTest.resize(size);
	 ifs.read((char*)&zListTest[0],size*sizeof(zScoreObj));
	 ifs.close();

	 double** data= (double**)malloc(global::test::dataBase::sampleNum*sizeof(double*));
	 for ( int i=0; i<global::test::dataBase::sampleNum; i++ ){
		 data[i] = (double*)malloc(global::test::dataBase::fpDimension*sizeof(double));
	 }

	 for ( int i=0; i<global::test::dataBase::sampleNum; i++ ) {
		 for ( int j=0; j<global::test::dataBase::fpDimension; j++ ){
			 data[i][j] = zListTest[i].getData(j);
		 }
	 }

	double** distmatrix;

	const int nrows = global::test::dataBase::sampleNum;
	const int ncols = global::test::dataBase::fpDimension;
	int** mask = (int**)malloc(nrows*sizeof(int*));

	for (int i = 0; i < nrows; i++){ 
		mask[i] = (int*)malloc(ncols*sizeof(int));
	}
	for (int i = 0; i<nrows; i++){
		for ( int j=0; j<ncols; j++){
			mask[i][j] = 1;
		}
	}

	distmatrix = example_distance_gene(global::test::dataBase::sampleNum, global::test::dataBase::fpDimension, data, mask);
	if (distmatrix) example_hierarchical(nrows, ncols, data, mask, distmatrix);


	Node* tree;
	double* weight = (double*)malloc(ncols*sizeof(double));
	for (int i = 0; i < ncols; i++) weight[i] = 1.0;
	tree = treecluster(nrows, ncols, data, mask, weight, 0, 'e', 'm', 0);
	//int* clusterid = new int[global::test::dataBase::sampleNum];
	int clusterid[16];
	cuttree(global::test::dataBase::sampleNum,tree,8,clusterid);

	string imageDir = global::test::dataBase::location;
	string clusterDir = "E:/360/face/test/result/cluster/";
	for ( int i=0; i<16; i++ ){
		string imageLoc = imageDir + zxitoa(i+1) + ".jpg";
		Mat image = imread(imageLoc);
		string clusterSubDir = clusterDir + zxitoa(clusterid[i]);
		_mkdir(clusterSubDir.c_str());
		string clusterLoc = clusterSubDir + "/" +zxitoa(i+1) + ".jpg";
		imwrite(clusterLoc,image);
	}

	string test = zxitoa(15);
	cout<<test<<endl;
	system("pause");
	return 0;
}

int getFileLocList( const string& dir, vector<string>& fileLocList )
{
	long handle;
	_finddata_t fileInfo;
	{
		string vir_fir = dir + "*.*";
		handle = _findfirst(vir_fir.c_str(),&fileInfo);
		if ( handle==-1 )
		{
			cout<<dir<<" getFileList failed."<<endl;
			return -1;
		}
	}
	do 
	{
		if ( fileInfo.attrib==_A_SUBDIR || fileInfo.attrib==_A_HIDDEN )
		{
			continue;
		}

		string fileLoc = dir + '/' + fileInfo.name;
		fileLocList.push_back(fileLoc);
	} while (!_findnext(handle,&fileInfo));
	_findclose(handle);

	return 0;
}

void example_hierarchical(int nrows, int ncols, double** data, int** mask,double** distmatrix){
	int i;
	const int nnodes = nrows-1;
	double* weight = (double*)malloc(ncols*sizeof(double));
	int* clusterid;
	Node* tree;
	for (i = 0; i < ncols; i++) weight[i] = 1.0;
	printf("\n");
	printf("================ Pairwise single linkage clustering ============\n");
	/* Since we have the distance matrix here, we may as well use it. */
	tree = treecluster(nrows, ncols, 0, 0, 0, 0, 'e', 's', distmatrix);
	/* The distance matrix was modified by treecluster, so we cannot use it any
	* more. But we still need to deallocate it here.
	* The first row of distmatrix is a single null pointer; no need to free it.
	*/
	for (i = 1; i < nrows; i++) free(distmatrix[i]);
	free(distmatrix);
	if (!tree)
	{ /* Indication that the treecluster routine failed */
		printf ("treecluster routine failed due to insufficient memory\n");
		free(weight);
		return;
	}
	printf("Node     Item 1   Item 2    Distance\n");
	for(i=0; i<nnodes; i++)
		printf("%3d:%9d%9d      %g\n",
		-i-1, tree[i].left, tree[i].right, tree[i].distance);
	printf("\n");
	free(tree);
	printf("================ Pairwise maximum linkage clustering ============\n");
	tree = treecluster(nrows, ncols, data, mask, weight, 0, 'e', 'm', 0);
	/* Here, we let treecluster calculate the distance matrix for us. In that
	* case, the treecluster routine may fail due to insufficient memory to store
	* the distance matrix. For the small data sets in this example, that is
	* unlikely to occur though. Let's check for it anyway:
	*/
	if (!tree)
	{ /* Indication that the treecluster routine failed */
		printf ("treecluster routine failed due to insufficient memory\n");
		free(weight);
		return;
	}
	printf("Node     Item 1   Item 2    Distance\n");
	for(i=0; i<nnodes; i++)
		printf("%3d:%9d%9d      %g\n",
		-i-1, tree[i].left, tree[i].right, tree[i].distance);
	printf("\n");
	free(tree);
}