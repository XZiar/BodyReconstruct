#pragma once
#ifndef FITMESH_H
#define FITMESH_H

#include "main.h"
#include "cshapepose.h"
#include "tools.h"

#include "kdNNTree.h"

#include <dlib/optimization.h>

#define POSPARAM_NUM 31
#define SHAPEPARAM_NUM 20
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
    arma::mat pose_params;
    arma::mat shape_params;
    arma::mat normals;
    arma::mat landmarks;
    arma::mat landmarksIdx;
    arma::mat meanShape;
    uint32_t nPoints;
};

struct CParams
{
    int nPCA;
    int nPose;
    int nSamplePoints;
};

class fitMesh
{
public:
	static CTemplate tempbody;
	static CShapePose shapepose;

	CScan scanbody;

	bool isFastCost = false;
	bool useFLANN = false;
    int angleLimit;
	double scale;
    arColIS isVisible;
	cv::Mat idxsNN_;
	arColIS isValidNN_;

    CParams params;
    arma::mat evectors;//the eigen vetors of the body shape model
    arma::mat evalues;
    std::string dataDir;
    ctools tools;
public:
	fitMesh();
	virtual ~fitMesh();

private:
	void calculateNormals(const std::vector<uint32_t> &points, arma::mat &faces, arma::mat &normals, arma::mat &normals_faces);
	void calculateNormalsFaces(arma::mat &points, arma::mat &faces, arma::mat &normals_faces);
	std::vector<uint32_t> getVertexFacesIdx(int point_idx, arma::mat &faces);
	static int32_t getVertexFacesIdxEx(const int point_idx, const arma::mat& faces);
	void rigidAlignTemplate2ScanPCA();
	arma::mat rigidAlignFix(const arma::mat& input, const arma::mat& R, double& dDepth);
	void rigidAlignTemplate2ScanLandmarks();
	void fitModel();
	void fitShapePose();
	arColIS checkAngle(const arma::mat& normals_knn, const arma::mat& normals_tmp, const double angle_thres);
	//基于ceres求解
	void solvePose(const cv::Mat& idxNN, const arColIS& isValidNN, arma::mat &poseParam, const arma::mat &shapeParam, double &scale);
	//基于ceres求解
	void solveShape(const cv::Mat &idxNN, const arColIS &isValidNN, const arma::mat &poseParam, arma::mat &shapeParam, double &scale);
	void updatePoints(cv::Mat &idxsNN_rtn, arColIS &isValidNN_rtn, double &scale, double &err);
	void solvePose_dlib();
	void solveShape_dlib();

public:
	void loadLandmarks();
	void loadScan();
	void loadTemplate();
	void loadModel();
	void mainProcess();
	arma::mat test();
	void showResult(bool isNN);
	static double posecost_dlib(dlib::matrix<double, POSPARAM_NUM, 1> pose);
	static double shapecost_dlib(dlib::matrix<double, SHAPEPARAM_NUM, 1> shape);
};

#endif // FITMESH_H
