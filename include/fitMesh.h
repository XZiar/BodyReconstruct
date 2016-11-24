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
    arma::mat normals;
    arma::mat normals_faces;
    arma::mat landmarks;
    arma::mat T;//rigid transform to the template
    std::vector<uint32_t> sample_point_idxes;
    //cv::flann::Index *kdtree;
	miniBLAS::VertexIVec vColors;
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
	void setPose(const double(&pose_)[POSPARAM_NUM])
	{
		memmove(pose, pose_, sizeof(double)*POSPARAM_NUM);
	};
	void setShape(const double(&shape_)[SHAPEPARAM_NUM])
	{
		memmove(shape, shape_, sizeof(double)*SHAPEPARAM_NUM);
	};
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
	std::atomic_bool isEnd, isAnimate, isShowScan = true;
	SimpleLog logger;

	bool isFastCost = false;
	bool isRayTrace = false;
	bool isAngWgt = false;
	bool isShFix = true;
	//bool useFLANN = false;
    int angleLimit;
	double scale;
	arColIS isValidNN_;
	std::vector<float> weights;
	double tSPose = 0, tSShape = 0, tMatchNN = 0;
	uint32_t cSPose = 0, cSShape = 0, cMatchNN = 0;

	uint32_t nSamplePoints;
    arma::mat evectors;//the eigen vetors of the body shape model
    arma::mat evalues;
    std::string dataDir;
public:
	fitMesh(std::string dir);
	~fitMesh();
	void loadLandmarks();
	void init(const std::string& baseName, const bool isOnce);
	//The main process of the fitting procedue
	void mainProcess();
	//The function of watch the result
	void watch();
	void watch(const uint32_t frameCount);
private:
	std::string baseFName;
	bool mode = true;//true: one scan, false:scan sequence
	miniBLAS::Vertex camPos;
	arma::mat rotateMat;
	arma::rowvec totalShift, baseShift;
	double totalScale;
	ModelParam curMParam, bakMParam, tmpMParam;
	PtrModSmooth msmooth;

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
	 ** @param -paramer  a function to determine solve params by iteration
	 **					function<tuple<double, bool, bool>(const uint32_t, const uint32_t, const double)>
	 **					cur_iter, all_iter, angle_limit ==> angle_limit, isSolvePose, isSolveShape
	 **/
	void fitShapePose(const CScan& scan, const uint32_t iter, 
		std::function<std::tuple<double, bool, bool>(const uint32_t, const uint32_t, const double)> paramer);
	void fitFinalShape(const uint32_t iter);
	void solvePose(const miniBLAS::VertexVec& scanCache, const arColIS& isValidNN, const double lastErr, const uint32_t curiter);
	void solveShape(const miniBLAS::VertexVec& scanCache, const arColIS& isValidNN, const double lastErr);
	void solveAllShape(const double angLim, const ceres::Solver::Options& options);
	
	/** @brief nnFilter
	 ** @param -angLim  max angle(in degree) between two norms
	 **/
	std::vector<float> nnFilter(const miniBLAS::NNResult& res, arColIS& result, const miniBLAS::VertexVec& scNorms, const double angLim);
	void raytraceCut(miniBLAS::NNResult& res) const;
	uint32_t updatePoints(const CScan& scan, const ModelParam& mPar, const double angLim, std::vector<uint32_t>& idxs, arColIS &isValidNN_rtn, double &err);
	/** @brief showResult
	 ** it must be called after updatepoints cause it skip the update precess
	 **/
	void showResult(const CScan& scan, const bool showScan = true, const std::vector<uint32_t>* const idxs = nullptr);
	void showColorResult(const CScan& scan, const bool showScan = true);

	std::string buildName(const uint32_t frame);
	void saveMParam(const std::string& fname);
	void showFrame(const uint32_t frame);

	void printFrame(const uint32_t frame);
	void setTitle(const std::string& title);
};

#endif // FITMESH_H
