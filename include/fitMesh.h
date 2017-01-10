#pragma once
#ifndef FITMESH_H
#define FITMESH_H

#include "main.h"
#include "cshapepose.h"
#include "tools.h"

#include "kdNNTree.hpp"

using arColIS = std::vector<int8_t>;

void printMat(const char *str, const arma::mat& m);
void printMatAll(const char *str, const arma::mat& m);

/*used for store information of triangles for ray-tracing*/
ALIGN16 struct FastTriangle : public miniBLAS::AlignBase<32>
{
	miniBLAS::Vertex p0, axisu, axisv, norm;
	miniBLAS::VertexI pidx;
	
	bool intersect(const miniBLAS::Vertex& origin, const miniBLAS::Vertex& direction, const miniBLAS::VertexI& idx, const float dist) const;
};

/*Models all contain points,normals,colors*/
struct ModelBase
{
	miniBLAS::VertexVec vPts;
	miniBLAS::VertexVec vNorms;
	miniBLAS::VertexVec vColors; 
	uint32_t nPoints;
	miniBLAS::h3NNTree nntree;
	void ShowPC(pcl::PointCloud<pcl::PointXYZRGB>& cloud) const;
	void ShowPC(pcl::PointCloud<pcl::PointXYZRGB>& cloud, pcl::PointXYZRGB color, const bool isCalcNorm = true) const;
};

struct CScan : public ModelBase
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

	void prepare();
};

struct CTemplate : public ModelBase
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
	/*calculate normals simply(use face normal)*/
	void calcNormals();
	/*calculate normals softly(use blend of face normals)*/
	void calcNormalsEx();
	std::vector<pcl::Vertices> ShowMesh(pcl::PointCloud<pcl::PointXYZRGB>& cloud);
};

/*Templete Model's params, controlling body shape and pose*/
struct ModelParam
{
	using PoseParam = std::array<double, POSPARAM_NUM>;
	using ShapeParam = std::array<double, SHAPEPARAM_NUM>;
	PoseParam pose;
	ShapeParam shape;
	ModelParam() 
	{
		pose.fill(0); shape.fill(0);
	};
	ModelParam(const PoseParam& pose_, const ShapeParam& shape_) : pose(pose_), shape(shape_)
	{
	};
	auto Pshape() { return shape.data(); };
	auto Ppose() { return pose.data(); };
};

/*simple utility to create log*/
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
	/*log string*/
	SimpleLog& log(const std::string& str, const bool isPrint = false);
	/*log formatted string, similar synax with printf*/
	template<typename... Args>
	SimpleLog& log(const bool isPrint, const char *s, Args... args)
	{
		sprintf(tmp, s, args...);
		return log(tmp, isPrint);
	}
	/*the log file will not be created before the first calling flush*/
	void flush();
};

class fitMesh
{
public:
	static pcl::visualization::CloudViewer viewer;
	static ctools tools;
	CShapePose shapepose;
	CTemplate tempbody;
	std::vector<CScan> scanFrames;
	std::vector<ModelParam> modelParams;

	volatile uint32_t curFrame = 0;
	volatile std::atomic_bool isEnd, isAnimate, isShowScan = true, isRefresh = true, isShowOrigin = false, isBlockPose = false;
	SimpleLog logger;

	bool isFastCost = true;
	bool isRayTrace = false;
	bool isAngWgt = true;
	bool isRE = false;
	bool isReShift = false;
	bool isP2S = false;
	bool isShFix = true;

	uint32_t nSamplePoints;
    int angleLimit;
    std::string dataDir;
public:
	fitMesh(const std::string& dir);
	~fitMesh();
	
	/*basename to identify scan, filename is the real file name(with subdir), isOnce means it's only a scan or a scan sequence*/
	void init(const std::string& baseName, const std::string& fileName, const bool isOnce);
	//The main process of the fitting procedue
	void mainProcess();
	//The function of watch the result
	void watch();
	//log pre-saved model-params to watch
	void watch(const uint32_t frameCount);
private:
	double npp[POSPARAM_NUM][6];
	double scale;
	arColIS isValidNN_;
	std::vector<float> weights;
	miniBLAS::VertexVec normalCache;
	double tSPose = 0, tSShape = 0, tMatchNN = 0;
	uint32_t cSPose = 0, cSShape = 0, cMatchNN = 0;

	std::string baseFName, baseEleName;
	bool mode = true;//true: one scan, false:scan sequence
	enum ShowMode : uint8_t { None, PointCloud, ColorCloud, Mesh };
	ShowMode showTMode = ColorCloud;
	miniBLAS::Vertex camPos;
	//store the align params of the first frame
	arma::mat rotateMat;
	arma::rowvec totalShift, baseShift;
	double totalScale;
	ModelParam curMParam, bakMParam, predMParam;
	PtrModSmooth msmooth;

	void loadTemplate();
	void loadLandmarks();
	void loadModel();
	bool loadScan(CScan& scan);
	void rigidAlignTemplate2ScanPCA(CScan& scanbody);
	arma::mat rigidAlignFix(const arma::mat& input, const arma::mat& R, double& dDepth);
	static arma::vec searchShoulder(const arma::mat& model, const uint32_t level,
		std::vector<double>& widAvg, std::vector<double>& depAvg, std::vector<double>& depMax);
	void DirectRigidAlign(CScan& scan);
	void rigidAlignTemplate2ScanLandmarks();

	struct FitParam
	{
		//whether solve pose and shape
		bool isSPose, isSShape;
		//additional param
		uint32_t param;
		//angle limit, which may differ among different frames
		double anglim;
		//solve option to use
		ceres::Solver::Options option;
	};
	/** @brief fitShapePose
	 ** Fit the model to the scan by pose & shape
	 ** @param -fitparams  a vector contains params needed for fit process
	 **/
	void fitShapePose(const CScan& scan, const std::vector<FitParam>& fitparams);
	void fitShapePoseRe(const CScan& scan, const std::vector<FitParam>& fitparams);
	/** @brief fitFinal
	 ** Fit the model to the whole scan series by pose & shape
	 ** @param -fitparams  a vector contains params needed for fit process
	 **/
	void fitFinal(const std::vector<FitParam>& fitparams);
	void solvePose(const ceres::Solver::Options& options, const miniBLAS::VertexVec& scanCache, const arColIS& isValidNN, const double lastErr, const uint32_t curiter);
	void solvePoseRe(const ceres::Solver::Options& options, const miniBLAS::VertexVec& scanCache, const std::vector<uint32_t>& idxs, const double lastErr, const uint32_t curiter);
	void solveShape(const ceres::Solver::Options& options, const miniBLAS::VertexVec& scanCache, const arColIS& isValidNN, const double lastErr);
	void solveShapeRe(const ceres::Solver::Options& options, const miniBLAS::VertexVec& scanCache, const std::vector<uint32_t>& idxs, const double lastErr);
	void solveAllPose(const ceres::Solver::Options& options, const double angLim, const bool dopred);
	void solveAllShape(const ceres::Solver::Options& options, const double angLim);
	void solveAllShapeRe(const ceres::Solver::Options& options, const double angLim);
	
	/*perform prediction, result stores in predParam.*/
	void predictPose();
	/*perform prediction for soften movement in the final solve stage.solveGlobal means whether predict first 6 params(global movement)*/
	void predSoftPose(const bool solveGlobal);

	void buildModelColor();

	/** @brief nnFilter
	 ** @param -angLim  max angle(in degree) between two norms
	 **/
	std::vector<float> nnFilter(const miniBLAS::NNResult& res, arColIS& isValid, const miniBLAS::VertexVec& scNorms, const double angLim);
	void raytraceCut(miniBLAS::NNResult& res) const;

	/*perform search and update templete. errors and matched points are also calculated.*/
	uint32_t updatePoints(const CScan& scan, const ModelParam& mPar, const double angLim, std::vector<uint32_t>& idxs, arColIS &isValidNN_rtn, double &err);
	/*reverse version of updatepoints, point cache is immediately filled here.*/
	uint32_t updatePointsRe(const CScan& scan, const ModelParam& mPar, const double angLim, std::vector<uint32_t>& idxs, miniBLAS::VertexVec& ptCache, double &err);
	/** @brief showResult
	 ** it must be called after updatepoints cause it skip the update precess
	 **/
	void showResult(const CScan& scan, const bool showScan = true, const std::vector<uint32_t>* const idxs = nullptr) const;
	
	/*save current model-param sequences to file*/
	bool saveMParam(const std::string& fname);
	/*read model-param sequences from file, and load scans if needed*/
	bool readMParamScan(const std::string& fname);
	void showFrame(const uint32_t frame);

	std::string buildName(const uint32_t frame);
	void printFrame(const uint32_t frame);
	void setTitle(const std::string& title);
};

#endif // FITMESH_H
