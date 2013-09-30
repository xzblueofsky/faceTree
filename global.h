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
			const static string fgmLoc = "E:/360/face/result/gaborFpGroupMatrix.xml";	//train set gabor fingerprint matrix file location
			const static string meanLoc = "E:/360/face/result/gfgmMean.xml";	//train set gabor fingerprint matrix mean file location
			const static string covarLoc = "E:/360/face/result/gfgmCovar.xml";	//train set gabor fingerprint matrix coviarance file location
			const static string normGfgm = "E:/360/face/result/normGfgm.xml";	//normalized gfgm xml file location
		}

		namespace pca{
			namespace loc{
				const static string eigenValues = "E:/360/face/result/gfgmEigenValues.xml";
				const static string eigenVectors = "E:/360/face/result/gfgmEigenVectors.xml";
				const static string projected = "E:/360/face/result/gfgmProjected.xml";
			}
			namespace tag{
				const static string eigenValues = "gfgmEigenValues";
				const static string eigenVectors = "gfgmEigenVectors";
				const static string projected = "gfgmProjected";
			}
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
			const static string fgmLoc = "E:/360/face/result/lbpFpGroupMatrix.xml";	//train set lbp fingerprint matrix file location
			const static string meanLoc = "E:/360/face/result/lfgmMean.xml";	//train set lbp fingerprint matrix mean file location
			const static string covarLoc = "E:/360/face/result/lfgmCovar.xml";	//train set lbp fingerprint matrix coviarance file location
			const static string normLfgm = "E:/360/face/result/normLfgm.xml";	//normalized lfgm xml file location
		}

		namespace pca{
			namespace loc{
				const static string eigenValues = "E:/360/face/result/lfgmEigenValues.xml";
				const static string eigenVectors = "E:/360/face/result/lfgmEigenVectors.xml";
				const static string projected = "E:/360/face/result/lfgmProjected.xml";
			}
			namespace tag{
				const static string eigenValues = "lfgmEigenValues";
				const static string eigenVectors = "lfgmEigenVectors";
				const static string projected = "lfgmProjected";
			}
		}
	}

	namespace z_score{
		namespace loc{
			const static string fileLoc = "E:/360/face/result/z_score.xml";
			const static string calcLoc = "E:/360/face/result/z_score.dat";	//data used to do clustering calculation
		}
		namespace tag{
			const static string tag = "z_score";
		}
	}

	namespace dataBase{
		const static string location = "E:/360/face/face_urls_com/";
	}

	namespace test{
		namespace tag{
			const static string tag = "test";
		}
		namespace loc{
			const static string loc = "E:/360/face/result/test.xml";
		}

		namespace gabor{
			namespace tag{
				const static string fpGroupMatrix = "gaborFpGroupMatrix";	//train set gabor fingerprint xml tag
				const static string mean = "gfgmMean";	//train set gabor fingerprint mean xml tag
				const static string covar = "gfgmCovar";	//train set gabor fingerprint matrix coviarance matrix
				const static string normGfgm = "normGfgm";	//normalized gfgm
			}

			namespace loc{
				const static string fgmLoc = "E:/360/face/test/result/gaborFpGroupMatrix.xml";	//train set gabor fingerprint matrix file location
				const static string meanLoc = "E:/360/face/test/result/gfgmMean.xml";	//train set gabor fingerprint matrix mean file location
				const static string covarLoc = "E:/360/face/test/result/gfgmCovar.xml";	//train set gabor fingerprint matrix coviarance file location
				const static string normGfgm = "E:/360/face/test/result/normGfgm.xml";	//normalized gfgm xml file location
			}

			namespace pca{
				namespace loc{
					const static string eigenValues = "E:/360/face/test/result/gfgmEigenValues.xml";
					const static string eigenVectors = "E:/360/face/test/result/gfgmEigenVectors.xml";
					const static string projected = "E:/360/face/test/result/gfgmProjected.xml";
				}
				namespace tag{
					const static string eigenValues = "gfgmEigenValues";
					const static string eigenVectors = "gfgmEigenVectors";
					const static string projected = "gfgmProjected";
				}
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
				const static string fgmLoc = "E:/360/face/test/result/lbpFpGroupMatrix.xml";	//train set lbp fingerprint matrix file location
				const static string meanLoc = "E:/360/face/test/result/lfgmMean.xml";	//train set lbp fingerprint matrix mean file location
				const static string covarLoc = "E:/360/face/test/result/lfgmCovar.xml";	//train set lbp fingerprint matrix coviarance file location
				const static string normLfgm = "E:/360/face/test/result/normLfgm.xml";	//normalized lfgm xml file location
			}

			namespace pca{
				namespace loc{
					const static string eigenValues = "E:/360/face/test/result/lfgmEigenValues.xml";
					const static string eigenVectors = "E:/360/face/test/result/lfgmEigenVectors.xml";
					const static string projected = "E:/360/face/test/result/lfgmProjected.xml";
				}
				namespace tag{
					const static string eigenValues = "lfgmEigenValues";
					const static string eigenVectors = "lfgmEigenVectors";
					const static string projected = "lfgmProjected";
				}
			}
		}

		namespace z_score{
			namespace loc{
				const static string fileLoc = "E:/360/face/test/result/z_score.xml";
				const static string calcLoc = "E:/360/face/test/result/z_score.dat";	//data used to do clustering calculation
			}
			namespace tag{
				const static string tag = "z_score";
			}
		}

		namespace dataBase{
			const static string location = "E:/360/face/test/face_urls_com/";
			const static int fpDimension = 132;
			const static int sampleNum = 16;
			const static string dataFile = "E:/360/face/test/data.txt";
			const static string cluster = "E:/360/face/test/result/cluster/";
		}
	}

	//const int sampleNumber = 11626;
}
#endif