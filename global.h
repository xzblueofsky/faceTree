#ifndef GLOBAL_H
#define GLOBSL_H
#include <iostream>

using namespace std;

namespace global{
	namespace gabor{
		namespace tag{
			const static string fpGroupMatrix = "gaborFpGroupMatrix";	//train set gabor fingerprint xml tag
			const static string mean = "gfgmMean";	//train set gabor fingerprint mean xml tag
			const static string covar = "gfgmCovar";	//train set gabor fingerprint matrix coviarance matrix
			const static string normGfgm = "normGfgm";	//normalized gfgm
		}

		namespace loc{
			const static string fgmLoc = "H:/360/result/gaborFpGroupMatrix.xml";	//train set gabor fingerprint matrix file location
			const static string meanLoc = "H:/360/result/gfgmMean.xml";	//train set gabor fingerprint matrix mean file location
			const static string covarLoc = "H:/360/result/gfgmCovar.xml";	//train set gabor fingerprint matrix coviarance file location
			const static string normGfgm = "H:/360/result/normGfgm.xml";	//normalized gfgm xml file location
		}
	}

	namespace lbp{
		namespace tag{
			const static string fpGroupMatrix = "lbpFpGroupMatrix";	//train set lbp fingerprint matrix xml tag
			const static string mean = "lfgmMean";	//train set lbp fingerprint matrix mean tag
			const static string covar = "lfgmCovar";	//train set lbp fingerprint matrix coviarance tag
			const static string normLfgm = "normLfgm";	//normalized lfgm
		}
		
		namespace loc{
			const static string fgmLoc = "H:/360/result/lbpFpGroupMatrix.xml";	//train set lbp fingerprint matrix file location
			const static string meanLoc = "H:/360/result/lfgmMean.xml";	//train set lbp fingerprint matrix mean file location
			const static string covarLoc = "H:/360/result/lfgmCovar.xml";	//train set lbp fingerprint matrix coviarance file location
			const static string normLfgm = "H:/360/result/normLfgm.xml";	//normalized lfgm xml file location
		}
	}
	namespace test{
		namespace tag{
			const static string tag = "test";
		}
		namespace loc{
			const static string loc = "H:/360/result/test.xml";
		}
	}

	//const int sampleNumber = 11626;
}
#endif