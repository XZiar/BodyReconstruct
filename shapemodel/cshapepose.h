#pragma once
#ifndef CSHAPEPOSE_H
#define CSHAPEPOSE_H

#include "main.h"
#include "CTMesh.h"

class CShapePose
{
private:
	CMesh initMesh_bk;
	arma::mat evectors;
	miniBLAS::VertexVec evecCache;
	miniBLAS::Vertex evalue[5];
	int getpose(const double* motionParamsIn, const double* shapeParamsIn, const double *eigenVectorsIn,
		const uint32_t numEigenVectors, double* pointsOut, double* jointsOut);

public:
	bool isFastFitShape = false;
	CShapePose();
	void preCompute(const char *__restrict validMask) { initMesh_bk.preCompute(validMask); };
	miniBLAS::VertexVec getBaseModel(const double *__restrict shapeParamsIn);
	CMesh getBaseModel2(const double *__restrict shapeParamsIn, const char *__restrict validMask);
	miniBLAS::VertexVec getModelByPose(const miniBLAS::VertexVec& basePoints, const double *__restrict poseParamsIn);
	miniBLAS::VertexVec getModelByPose2(const CMesh& baseMesh, const double *__restrict poseParamsIn,
		const char *__restrict validMask);
	miniBLAS::VertexVec getModelFast(const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn);
	miniBLAS::VertexVec getModelFast2(const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn,
		const char *__restrict validMask);
	void getModel(const double *shapeParamsIn, const double *poseParamsIn, arma::mat &points, arma::mat &joints);
	void getModel(const arma::mat &shapeParam, const arma::mat &poseParam, arma::mat &points, arma::mat &joints);
	void setEvectors(arma::mat &evectorsIn);
	void setEvalues(const arma::mat& evelues);
};

#endif // CSHAPEPOSE_H
