#pragma once
#ifndef CSHAPEPOSE_H
#define CSHAPEPOSE_H

#include "main.h"
#include "CTMesh.h"

class CShapePose
{
private:
	CMesh initMesh_bk;
	miniBLAS::Vertex evalue[5];
	int getpose(const double* motionParamsIn, const double* shapeParamsIn, const double *eigenVectorsIn,
		const uint32_t numEigenVectors, double* pointsOut, double* jointsOut);

public:
	CShapePose(const std::string& modelFileName);
	PtrModSmooth preCompute(const int8_t *__restrict validMask) { return initMesh_bk.preCompute(validMask); };
	miniBLAS::VertexVec getBaseModel(const double *__restrict shapeParamsIn) const;
	CMesh getBaseModel2(const double *__restrict shapeParamsIn, const int8_t *__restrict validMask) const;
	miniBLAS::VertexVec getModelByPose(const miniBLAS::VertexVec& basePoints, const double *__restrict poseParamsIn) const;
	miniBLAS::VertexVec getModelByPose2(const PtrModSmooth mSmooth, const CMesh& baseMesh, const double *__restrict poseParamsIn,
		const int8_t *__restrict validMask) const;
	miniBLAS::VertexVec getModelFast(const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn) const;
	miniBLAS::VertexVec getModelFast2(const PtrModSmooth mSmooth, const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn,
		const int8_t *__restrict validMask) const;
	void setEvectors(const arma::mat &evectorsIn);
	void setEvalues(const arma::mat& evelues);
};

#endif // CSHAPEPOSE_H
