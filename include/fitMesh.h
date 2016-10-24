#pragma once
#ifndef FITMESH_H
#define FITMESH_H

#include "main.h"
#include "cshapepose.h"
#include "tools.h"

#include "kdNNTree.h"

using arColIS = arma::Col<char>;

void printMat(const char *str, arma::mat m);
void printMatAll(const char * str, arma::mat m);

struct CScan
{
    arma::mat points;
    arma::mat faces;
    arma::mat pose_params;
    arma::mat shape_params;
    arma::mat normals;
    arma::mat normals_faces;
    arma::mat landmarks;
    arma::mat T;//rigid transform to the template
    std::vector<uint32_t> sample_point_idxes;
    uint32_t nPoints;
    cv::flann::Index *kdtree;
	miniBLAS::kdNNTree nntree;
    arma::mat points_orig;
    arma::mat normals_orig;
};

struct CTemplate
{
    arma::mat points;
	std::vector<uint32_t> points_idxes;
    arma::mat faces;
    arma::mat normals;
    arma::mat landmarks;
    arma::mat landmarksIdx;
    arma::mat meanShape;
    uint32_t nPoints;
};

struct ModelParam
{
	double pose[POSPARAM_NUM];
	double shape[SHAPEPARAM_NUM];
	ModelParam() 
	{ 
		memset(pose, 0, sizeof(double)*(POSPARAM_NUM + SHAPEPARAM_NUM));
	};
};

struct CParams
{
    int nPCA;
    int nPose;
    int nSamplePoints;
};

class SimpleLog
{
private:
	FILE *fp = nullptr;
	std::string fname;
	std::string cache;
	char tmp[2048];
public:
	SimpleLog();
	~SimpleLog();
	SimpleLog& log(const std::string& str, const bool isPrint = false);
	template<typename... Args>
	SimpleLog& log(const bool isPrint, const char *s, Args... args)
	{
		sprintf(tmp, s, args...);
		return log(tmp, isPrint);
	}
	void flush();
};

class fitMesh
{
public:
	static CTemplate tempbody;
	static CShapePose shapepose;
	static ctools tools;
	std::vector<uint32_t> tpFaceMap;
	std::vector<CScan> scanFrames;
	std::vector<ModelParam> modelParams;

	uint32_t curFrame = 0;
	SimpleLog logger;

	bool isFastCost = false;
	bool isAgLimNN = false;
	bool useFLANN = false;
    int angleLimit;
	double scale;
    arColIS isVisible;
	cv::Mat idxsNN_;
	arColIS isValidNN_;
	double tSPose = 0, tSShape = 0, tMatchNN = 0;
	uint32_t cSPose = 0, cSShape = 0, cMatchNN = 0;

    CParams params;
    arma::mat evectors;//the eigen vetors of the body shape model
    arma::mat evalues;
    std::string dataDir;
public:
	fitMesh(std::string dir);
	virtual ~fitMesh();
	void loadLandmarks();
	void init(const std::string& baseName, const bool isOnce);
	//The main process of the fitting procedue
	void mainProcess();
private:
	std::string baseFName;
	bool mode = true;//true: one scan, false:scan sequence
	arma::mat rotateMat;
	arma::rowvec totalShift, baseShift;
	double totalScale;
	ModelParam curMParam;

	void loadTemplate();
	void loadModel();
	bool loadScan(CParams& params, CScan& scan);
	void calculateNormals(const std::vector<uint32_t> &points, arma::mat &faces, arma::mat &normals, arma::mat &normals_faces);
	void calculateNormalsFaces(arma::mat &points, arma::mat &faces, arma::mat &normals_faces);
	std::vector<uint32_t> getVertexFacesIdx(int point_idx, arma::mat &faces);
	void rigidAlignTemplate2ScanPCA(CScan& scanbody);
	arma::mat rigidAlignFix(const arma::mat& input, const arma::mat& R, double& dDepth);
	static arma::vec searchShoulder(arma::mat model, const unsigned int lv, 
		std::vector<double> &widAvg, std::vector<double> &depAvg, std::vector<double> &depMax);
	void DirectRigidAlign(CScan& scan);
	void rigidAlignTemplate2ScanLandmarks();
	/** @brief fitShapePose
	 ** Fit the model to the scan by pose & shape
	 **/
	void fitShapePose(const CScan& scan, const bool solveP = true, const bool solveS = true, uint32_t iter = 10);
	/** @brief checkAngle
	 ** @param angle_thres max angle(in degree) between two norms
	 **/
	arColIS checkAngle(const arma::mat& normals_knn, const arma::mat& normals_tmp, const double angle_thres);
	void solvePose(const miniBLAS::VertexVec& scanCache, const arColIS& isValidNN, ModelParam &tpParam, double &scale);
	void solveShape(const miniBLAS::VertexVec& scanCache, const arColIS &isValidNN, ModelParam &tpParam, double &scale);
	uint32_t updatePoints(const CScan& scan, cv::Mat &idxsNN_rtn, arColIS &isValidNN_rtn, double &scale, double &err);
	/** @brief showResult
	 ** it must be called after updatepoints cause it skip the update precess
	 **/
	void showResult(const CScan& scan, const bool isNN = false);
};

#endif // FITMESH_H
