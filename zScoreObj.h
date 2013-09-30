#ifndef Z_SCORE_OBJ_H
#define Z_SCORE_OBJ_H

#define DIM 132	//zScore dimentions
#define LOC_LEN 64	//64 charactors reserved to save image location

#include <iostream>
#include <sstream>
#include <cstring>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

using namespace std;

class zScoreObj{
private:
	char imageLoc[LOC_LEN];
	double zScoreData[DIM];

public:
	zScoreObj(const string& loc, const Mat& data){
		strcpy(imageLoc,loc.c_str());
		for ( int i=0; i<DIM; i++ ){
			zScoreData[i] = data.at<double>(0,i);
		}
	}
	
	zScoreObj(){
		strcpy(imageLoc,"");
		memset(zScoreData,0,DIM*sizeof(double));
	}

	string getImageLoc(){
		string loc(imageLoc);
		return loc;
	}

	double getData(int i) const{
		return zScoreData[i];
	}

	friend ostream& operator << (ostream& os, const zScoreObj& obj);
};

#endif