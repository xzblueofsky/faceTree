#include "stdafx.h"

#include "zScoreObj.h"

ostream& operator << (ostream& os, const zScoreObj& obj){
	os<<"loc:"<<obj.imageLoc<<"	";
	for ( int i=0; i<DIM; i++ ){
		os<<obj.zScoreData[i]<<"	";
	}
	os<<endl;
	return os;
}