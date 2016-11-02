#pragma once
#ifndef FITMESH_H
#define FITMESH_H

#include "main.h"
#include "cshapepose.h"
#include "tools.h"

#include "kdNNTree.hpp"

using arColIS = arma::Col<char>;

void printMat(const char *str, arma::mat m);
void printMatAll(const char *str, arma::mat m);

ALIGN16 struct FastTriangle : public miniBLAS::AlignBase<32>
{
	miniBLAS::Vertex p0, axisu, axisv, norm;
	miniBLAS::VertexI pidx;
	
	bool intersect(const miniBLAS::Vertex& origin, const miniBLAS::Vertex& direction, const miniBLAS::VertexI& idx, const float dist) const;
};

struct CBaseModel
{
	miniBLAS::VertexVec vPts;
	miniBLAS::VertexVec vNorms;
	uint32_t nPoints;
};

struct CScan : public CBaseModel
{
	arma::mat points_orig;
	arma::mat normals_orig;
    arma::mat points;
    arma::mat faces;
    arma::mat pose_params;
    arma::mat shape_params;
    arma::mat normals;
    arma::mat normals_faces;
    arma::mat landmarks;
    arma::mat T;//rigid transform to the template
    std::vector<uint32_t> sample_point_idxes;
    cv::flann::Index *kdtree;
	miniBLAS::h3NNTree nntree;

	void prepare();
};

struct CTemplate : public CBaseModel
{
    arma::mat points;
    arma::mat faces;
    arma::mat landmarks;
    arma::mat landmarksIdx;
	uint32_t nFaces;

	std::vector<uint32_t> faceMap;
	std::vector<FastTriangle, miniBLAS::AlignAllocator<FastTriangle>> vFaces;

	void init(const arma::mat& p, const arma::mat& f);
	void updPoints();
	void updPoints(miniBLAS::VertexVec&& pts);
	void calcFaces();
	void calcNormals();
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
	miniBLAS::Vertex camPos;
	pcl::ModelCoefficients coef;
	arma::mat rotateMat;
	arma::rowvec totalShift, baseShift;
	double totalScale;
	ModelParam curMParam;

	void loadTemplate();
	void loadModel();
	bool loadScan(CScan& scan);
	void rigidAlignTemplate2ScanPCA(CScan& scanbody);
	arma::mat rigidAlignFix(const arma::mat& input, const arma::mat& R, double& dDepth);
	static arma::vec searchShoulder(const arma::mat& model, const uint32_t level,
		std::vector<double>& widAvg, std::vector<double>& depAvg, std::vector<double>& depMax);
	void DirectRigidAlign(CScan& scan);
	void rigidAlignTemplate2ScanLandmarks();
	/** @brief fitShapePose
	 ** Fit the model to the scan by pose & shape
	 **/
	void fitShapePose(const CScan& scan, const bool solveP = true, const bool solveS = true, const uint32_t iter = 10);
	void solvePose(const miniBLAS::VertexVec& scanCache, const arColIS& isValidNN, ModelParam &tpParam, const double lastErr);
	void solveShape(const miniBLAS::VertexVec& scanCache, const arColIS &isValidNN, ModelParam &tpParam);
	/** @brief checkAngle
	 ** @param angle_thres max angle(in degree) between two norms
	 **/
	arColIS checkAngle(const arma::mat& normals_knn, const arma::mat& normals_tmp, const double angle_thres);
	void nnFilter(const miniBLAS::NNResult& res, arColIS& result, const miniBLAS::VertexVec& scNorms, const double angLim);
	void raytraceCut(miniBLAS::NNResult& res) const;
	uint32_t updatePoints(const CScan& scan, const double angLim, cv::Mat &idxsNN_rtn, arColIS &isValidNN_rtn, double &err);
	/** @brief showResult
	 ** it must be called after updatepoints cause it skip the update precess
	 **/
	void showResult(const CScan& scan, const bool isNN = false);
};

#endif // FITMESH_H
