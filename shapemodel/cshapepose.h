#pragma once
#ifndef CSHAPEPOSE_H
#define CSHAPEPOSE_H

#include "main.h"
#include "CTMesh.h"

class CShapePose
{
private:
	CMesh initMesh_bk;
	miniBLAS::Vertex bodysize, paramscale;
	/*For shape param, each param may be different in its scale, hence use evalue to make them in a same scale, which is better for solving*/
	miniBLAS::Vertex evalue[5];
	int getpose(const double* motionParamsIn, const double* shapeParamsIn, const double *eigenVectorsIn,
		const uint32_t numEigenVectors, double* pointsOut, double* jointsOut);
	// read motion params from the precomputed 3D poses
	CVector<double> getPoseVec(const double *poseParamsIn) const;
public:
	CShapePose(const std::string& modelFileName);
	/*pre-compute mesh data based on validmask*/
	PtrModSmooth preCompute(const int8_t *__restrict validMask) const { return initMesh_bk.preCompute(validMask); };
	/*For pose-solving, shape param remains unchanged, hence fitShapeChangesToMesh and updJoint can be calc only once
	 *Function return a CMesh, which can be passed to getModelByPose
	 **/
	CMesh getBaseModel(const double *__restrict shapeParamsIn) const;
	/*use validMask to precompute validPts, returning CMesh's smoothparam is also set to precompute value*/
	CMesh getBaseModel2(const double *__restrict shapeParamsIn, const int8_t *__restrict validMask) const;
	miniBLAS::VertexVec getModelByPose(const CMesh& baseMesh, const double *__restrict poseParamsIn) const;
	miniBLAS::VertexVec getModelByPose2(const CMesh& baseMesh, const double *__restrict poseParamsIn) const;
	miniBLAS::VertexVec getModelFast(const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn) const;
	miniBLAS::VertexVec getModelFast2(const PtrModSmooth mSmooth, const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn,
		const int8_t *__restrict validMask) const;
	void setEvectors(const arma::mat &evectorsIn);
	void setEvalues(const arma::mat& evelues);
};

#endif // CSHAPEPOSE_H
