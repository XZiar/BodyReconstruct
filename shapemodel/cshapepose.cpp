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

CShapePose::CShapePose(const string& modelFileName)
{
	initMesh_bk.readModel(modelFileName.c_str(), true);
	initMesh_bk.updateJntPos();
	initMesh_bk.centerModel();
	initMesh_bk.prepareData();
}

int CShapePose::getpose(const double* motionParamsIn, const double* shapeParamsIn, const double *eigenVectorsIn,
	const uint32_t numEigenVectors, double* pointsOut, double* jointsOut)
{
	// Read object model
	CMesh initMesh(initMesh_bk, false);

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
CMesh CShapePose::getBaseModel(const double *__restrict shapeParamsIn) const
{
	// Read object model
	CMesh baseMesh(initMesh_bk, true);
	Vertex vShape[5];
	{
		float *__restrict pShape = vShape[0];
		const float *pEV = evalue[0];
		for (uint32_t a = 0; a < SHAPEPARAM_NUM; ++a)
		{
			*pShape++ = shapeParamsIn[a] * (*pEV++);
		}
	}
	baseMesh.fastShapeChangesToMesh(vShape);
	// update joints
	baseMesh.updateJntPosEx();
	/*IMPORTANT!!!!
	*This function relys on copy elision since operator= of CMesh is alway the old version of copy.
	*However, since there is no default copy constructor(only CMesh(CMesh,bool)), there is no default move constructor, neither.
	*That means, when using the return value to initializing an CMesh, compiler will tends to use copy elision.
	*If copy elision failed and compiler use operator=, some data will not be copied right, and later function may throw error.
	*The best way to solve this is to remove old version codes in CMesh
	**/
	return baseMesh;
}
CMesh CShapePose::getBaseModel2(const double *__restrict shapeParamsIn, const int8_t *__restrict validMask) const
{
	// Read object model
	CMesh baseMesh(initMesh_bk, true);
	Vertex vShape[5];
	{
		float *__restrict pShape = vShape[0];
		const float *pEV = evalue[0];
		for (uint32_t a = 0; a < SHAPEPARAM_NUM; ++a)
		{
			*pShape++ = shapeParamsIn[a] * (*pEV++);
		}
	}
	baseMesh.fastShapeChangesToMesh(vShape, validMask);
	// update joints
	baseMesh.updateJntPosEx();
	baseMesh.modsmooth = baseMesh.preCompute(validMask);
	/*IMPORTANT!!!!
	 *This function relys on copy elision since operator= of CMesh is alway the old version of copy.
	 *However, since there is no default copy constructor(only CMesh(CMesh,bool)), there is no default move constructor, neither.
	 *That means, when using the return value to initializing an CMesh, compiler will tends to use copy elision.
	 *If copy elision failed and compiler use operator=, some data will not be copied right, and later function may throw error.
	 *The best way to solve this is to remove old version codes in CMesh
	 **/
	return baseMesh;
}
VertexVec CShapePose::getModelByPose(const CMesh& baseMesh, const double *__restrict poseParamsIn) const
{
	// Read object model
	CMesh mesh(baseMesh, true);

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
	mesh.rigidMotionSim_AVX(M);

	return std::move(mesh.vPoints);
}
VertexVec CShapePose::getModelByPose2(const CMesh& baseMesh, const double *__restrict poseParamsIn) const
{
	// Read object model
	CMesh mesh(baseMesh, true);

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
	mesh.rigidMotionSim2_AVX(M);
	return std::move(mesh.validPts);
}
VertexVec CShapePose::getModelFast(const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn) const
{
	// Read object model
	CMesh mesh(initMesh_bk, true);

	Vertex vShape[5];
	{
		float *__restrict pShape = vShape[0];
		const float *pEV = evalue[0];
		for (uint32_t a = 0; a < SHAPEPARAM_NUM; ++a)
		{
			*pShape++ = shapeParamsIn[a] * (*pEV++);
		}
	}
	mesh.fastShapeChangesToMesh(vShape);

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
	mesh.rigidMotionSim_AVX(M);

	return std::move(mesh.vPoints);
}
VertexVec CShapePose::getModelFast2(const PtrModSmooth mSmooth, 
	const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn, const int8_t *__restrict validMask) const
{
	// Read object model
	CMesh mesh(initMesh_bk, true);
	mesh.modsmooth = mSmooth;

	Vertex vShape[5];
	{
		float *__restrict pShape = vShape[0];
		const float *pEV = evalue[0];
		for (uint32_t a = 0; a < SHAPEPARAM_NUM; ++a)
		{
			*pShape++ = shapeParamsIn[a] * (*pEV++);
		}
	}
	mesh.fastShapeChangesToMesh(vShape, validMask);

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
	mesh.rigidMotionSim2_AVX(M);

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
