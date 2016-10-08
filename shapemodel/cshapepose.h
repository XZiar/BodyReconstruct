#pragma once
#ifndef CSHAPEPOSE_H
#define CSHAPEPOSE_H

#include "main.h"
#include "CTMesh.h"

#define SMOOTHMODEL true

class CShapePose
{
private:
	CMesh initMesh_bk;
	arma::mat evectors;
	std::vector<miniBLAS::Vertex> evecCache;
	miniBLAS::Vertex evalue[5];
	int getpose(const double* motionParamsIn, const double* shapeParamsIn, const double *eigenVectorsIn,
		const uint32_t numEigenVectors, double* pointsOut, double* jointsOut);

public:
	CShapePose();
	std::vector<miniBLAS::Vertex> getBaseModel(const double *__restrict shapeParamsIn);
	std::vector<miniBLAS::Vertex> getModelByPose(const std::vector<miniBLAS::Vertex>& basePoints, const double *__restrict poseParamsIn);
	std::vector<miniBLAS::Vertex> getModelFast(const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn);
	void getModel(const double *shapeParamsIn, const double *poseParamsIn, arma::mat &points, arma::mat &joints);
	void getModel(const arma::mat &shapeParam, const arma::mat &poseParam, arma::mat &points, arma::mat &joints);
	void setEvectors(arma::mat &evectorsIn);
	void setEvalues(const arma::mat& evelues);
};

#endif // CSHAPEPOSE_H
