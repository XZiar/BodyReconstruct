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
	int getpose(const double* motionParamsIn, const double* shapeParamsIn, double *eigenVectorsIn,
		uint32_t numEigenVectors, double* pointsOut, double* jointsOut);

public:
	CShapePose();

	void getModel(const arma::mat &shapeParam, const arma::mat &poseParam, arma::mat &points, arma::mat &joints);
	void setEvectors(arma::mat &evectorsIn);
};

#endif // CSHAPEPOSE_H
