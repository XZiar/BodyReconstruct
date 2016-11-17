#include "cshapepose.h"

#include "NRBM.h"
#include "onlyDefines.h"
#include "paramMap.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using miniBLAS::Vertex;
using miniBLAS::VertexVec;
using miniBLAS::SQMat4x4;

CShapePose::CShapePose()
{
	auto aModelFile = "../BodyReconstruct/data/model.dat";
	initMesh_bk.readModel(aModelFile, true);
	initMesh_bk.updateJntPos();
	initMesh_bk.centerModel();
	initMesh_bk.prepareData();
}

int CShapePose::getpose(const double* motionParamsIn, const double* shapeParamsIn, const double *eigenVectorsIn,
	const uint32_t numEigenVectors, double* pointsOut, double* jointsOut)
{
	// Read object model
	CMesh initMesh = initMesh_bk;

	initMesh.fastShapeChangesToMesh(shapeParamsIn, numEigenVectors, eigenVectorsIn);

	// update joints
	initMesh.updateJntPos();

	// read motion params from the precomputed 3D poses
	CVector<double> mParams(POSPARAM_NUM);
	for (uint32_t i = 0; i < POSPARAM_NUM; i++)
	{
		mParams(i) = motionParamsIn[i];
	}
	CMatrix<float> mRBM(4, 4);
	NRBM::RVT2RBM(&mParams, mRBM);
#ifndef EXEC_FAST
	cout << "3D pose parameters:\n";
	cout << mParams << endl;
#endif
	CVector<CMatrix<float> > M(CMesh::mJointNumber + 1);
	CVector<float> TW(POSPARAM_NUM);
	for (int j = 6; j < mParams.size(); ++j)
	{
		TW(j) = (float)mParams(j);
	}
	initMesh.angleToMatrix(mRBM, TW, M);

	// rotate joints
	initMesh.rigidMotion(M, TW, true, true);

	// Fill in resulting joints array
	const uint32_t nJoints = CMesh::mJointNumber;
	for (uint32_t i = 0; i < nJoints; i++)
	{
		CJoint& joint = initMesh.joint(i + 1);
		const auto& direct = joint.getDirection();
		const auto& point = joint.getPoint();
		uint32_t idx = i; jointsOut[idx] = i + 1;
		idx += nJoints; jointsOut[idx] = direct[0];
		idx += nJoints; jointsOut[idx] = direct[1];
		idx += nJoints; jointsOut[idx] = direct[2];
		idx += nJoints; jointsOut[idx] = point[0];
		idx += nJoints; jointsOut[idx] = point[1];
		idx += nJoints; jointsOut[idx] = point[2];
		idx += nJoints; jointsOut[idx] = joint.mParent;
	}
	// Fill in resulting points array
	int nPoints = initMesh.GetPointSize();
	for (int i = 0; i < nPoints; i++)
		initMesh.GetPoint(i, pointsOut[i], pointsOut[i + nPoints], pointsOut[i + 2 * nPoints]);
	return 0;
}
VertexVec CShapePose::getBaseModel(const double *__restrict shapeParamsIn) const
{
	// Read object model
	CMesh mesh(initMesh_bk, nullptr);
	Vertex vShape[5];
	{
		float *__restrict pShape = vShape[0];
		const float *pEV = evalue[0];
		for (uint32_t a = 0; a < SHAPEPARAM_NUM; ++a)
		{
			*pShape++ = shapeParamsIn[a] * (*pEV++);
		}
	}
	mesh.fastShapeChangesToMesh_AVX(vShape);
	return std::move(mesh.vPoints);
}
CMesh CShapePose::getBaseModel2(const double *__restrict shapeParamsIn, const char *__restrict validMask) const
{
	// Read object model
	CMesh initMesh(initMesh_bk, nullptr);
	Vertex vShape[5];
	{
		float *__restrict pShape = vShape[0];
		const float *pEV = evalue[0];
		for (uint32_t a = 0; a < SHAPEPARAM_NUM; ++a)
		{
			*pShape++ = shapeParamsIn[a] * (*pEV++);
		}
	}
	initMesh.fastShapeChangesToMesh_AVX(vShape, validMask);
	return initMesh;
}
VertexVec CShapePose::getModelByPose(const VertexVec& basePoints, const double *__restrict poseParamsIn) const
{
	// Read object model
	CMesh mesh(initMesh_bk, &basePoints);

	// update joints
	mesh.updateJntPosEx();

	// read motion params from the precomputed 3D poses
	CVector<double> mParams(POSPARAM_NUM);
	for (uint32_t i = 0; i < POSPARAM_NUM; i++)
	{
		mParams(i) = poseParamsIn[i];
	}
	CMatrix<float> mRBM(4, 4);
	NRBM::RVT2RBM(&mParams, mRBM);

	const auto M = mesh.angleToMatrixEx(mRBM, poseParamsIn);
	// rotate joints
	mesh.rigidMotionSim_AVX(M, true);

	return std::move(mesh.vPoints);
}
VertexVec CShapePose::getModelByPose2(const PtrModSmooth mSmooth, const CMesh& baseMesh,
	const double *__restrict poseParamsIn, const char *__restrict validMask) const
{
	// Read object model
	CMesh mesh(initMesh_bk, baseMesh, mSmooth);
	// update joints
	mesh.updateJntPosEx();

	// read motion params from the precomputed 3D poses
	CVector<double> mParams(POSPARAM_NUM);
	for (uint32_t i = 0; i < POSPARAM_NUM; i++)
	{
		mParams(i) = poseParamsIn[i];
	}
	CMatrix<float> mRBM(4, 4);
	NRBM::RVT2RBM(&mParams, mRBM);

	const auto M = mesh.angleToMatrixEx(mRBM, poseParamsIn);
	// rotate joints
	mesh.rigidMotionSim2_AVX(M, true);

	return std::move(mesh.validPts);
}
VertexVec CShapePose::getModelFast(const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn) const
{
	// Read object model
	CMesh mesh(initMesh_bk, nullptr);

	Vertex vShape[5];
	{
		float *__restrict pShape = vShape[0];
		const float *pEV = evalue[0];
		for (uint32_t a = 0; a < SHAPEPARAM_NUM; ++a)
		{
			*pShape++ = shapeParamsIn[a] * (*pEV++);
		}
	}
	mesh.fastShapeChangesToMesh_AVX(vShape);

	// update joints
	mesh.updateJntPosEx();

	// read motion params from the precomputed 3D poses
	CVector<double> mParams(POSPARAM_NUM);
	for (uint32_t i = 0; i < POSPARAM_NUM; i++)
	{
		mParams(i) = poseParamsIn[i];
	}
	CMatrix<float> mRBM(4, 4);
	NRBM::RVT2RBM(&mParams, mRBM);

	const auto M = mesh.angleToMatrixEx(mRBM, poseParamsIn);
	// rotate joints
	mesh.rigidMotionSim_AVX(M, true);

	return std::move(mesh.vPoints);
}
VertexVec CShapePose::getModelFast2(const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn, const char *__restrict validMask) const
{
	// Read object model
	CMesh mesh(initMesh_bk, nullptr);

	Vertex vShape[5];
	{
		float *__restrict pShape = vShape[0];
		const float *pEV = evalue[0];
		for (uint32_t a = 0; a < SHAPEPARAM_NUM; ++a)
		{
			*pShape++ = shapeParamsIn[a] * (*pEV++);
		}
	}
	mesh.fastShapeChangesToMesh_AVX(vShape, validMask);

	// update joints
	mesh.updateJntPosEx();

	// read motion params from the precomputed 3D poses
	CVector<double> mParams(POSPARAM_NUM);
	for (uint32_t i = 0; i < POSPARAM_NUM; i++)
	{
		mParams(i) = poseParamsIn[i];
	}
	CMatrix<float> mRBM(4, 4);
	NRBM::RVT2RBM(&mParams, mRBM);

	const auto M = mesh.angleToMatrixEx(mRBM, poseParamsIn);
	// rotate joints
	mesh.rigidMotionSim2_AVX(M, true);

	return std::move(mesh.validPts);
}

void CShapePose::setEvectors(const arma::mat &evectorsIn)
{
	initMesh_bk.setShapeSpaceEigens(evectorsIn);
}

void CShapePose::setEvalues(const arma::mat & inEV)
{
	if (inEV.n_elem != SHAPEPARAM_NUM || SHAPEPARAM_NUM != 20)
	{
		printf("evelues should be 20 for optimization.\n\n");
		getchar();
		exit(-1);
	}
	const auto *pIn = inEV.memptr();
	float *pOut = evalue[0];
	for (uint32_t a = SHAPEPARAM_NUM; a--;)
		*pOut++ = *pIn++;
	
	evalue[0].do_sqrt();
	evalue[1].do_sqrt();
	evalue[2].do_sqrt();
	evalue[3].do_sqrt();
	evalue[4].do_sqrt();
}
