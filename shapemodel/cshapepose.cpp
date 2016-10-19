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

int CShapePose::getpose(/*const string inputDir, */const double* motionParamsIn, const double* shapeParamsIn, const double *eigenVectorsIn,
	const uint32_t numEigenVectors, double* pointsOut, double* jointsOut)
{
	// ************** Read Input ************** //
	//string s = inputDir;
	//if( (char)(*--s.end()) != '/' )
	//s.append("/");
	//NShow::mInputDir = s;

#ifndef EXEC_FAST
	cout << "Input dir: " << NShow::mInputDir << endl;
#endif

#ifndef EXEC_FAST
	cout << "LOAD MESH...\n";
#endif 

#ifndef EXEC_FAST
	cout << "semantic shape parameters: ";
#endif

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
	CVector<CMatrix<float> > M(initMesh.joints() + 1);
	CVector<float> TW(initMesh.joints() + 6);
	for (int j = 6; j < mParams.size(); ++j)
	{
		TW(j) = (float)mParams(j);
	}
	initMesh.angleToMatrix(mRBM, TW, M);

	// rotate joints
	initMesh.rigidMotion(M, TW, true, true);

	// Fill in resulting joints array
	const uint32_t nJoints = initMesh.joints();
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
	{
		float x, y, z;
		initMesh.GetPoint(i, x, y, z);
		pointsOut[i] = x;
		pointsOut[i + nPoints] = y;
		pointsOut[i + 2 * nPoints] = z;
	}
	return 0;
}
VertexVec CShapePose::getBaseModel(const double *__restrict shapeParamsIn)
{
	const double *eigenVectorsIn = evectors.memptr();/* nEigenVec x EVALUATE_POINTS_NUM(6449) x 3*/
	const uint32_t numEigenVectors = evectors.n_rows;
	// Read object model
	CMesh initMesh(initMesh_bk, nullptr);
	Vertex vShape[5];
	{
		float *__restrict pShape = vShape[0];
		const float *pEV = evalue[0];
		for (uint32_t a = 0; a < 20; ++a)
		{
			*pShape++ = shapeParamsIn[a] * (*pEV++);
		}
	}
	initMesh.fastShapeChangesToMesh_AVX(vShape, &evecCache[0]);
	return std::move(initMesh.vPoints);
}
CMesh CShapePose::getBaseModel2(const double *__restrict shapeParamsIn, const char *__restrict validMask)
{
	const double *eigenVectorsIn = evectors.memptr();/* nEigenVec x EVALUATE_POINTS_NUM(6449) x 3*/
	const uint32_t numEigenVectors = evectors.n_rows;
	// Read object model
	CMesh initMesh(initMesh_bk, nullptr);
	Vertex vShape[5];
	{
		float *__restrict pShape = vShape[0];
		const float *pEV = evalue[0];
		for (uint32_t a = 0; a < 20; ++a)
		{
			*pShape++ = shapeParamsIn[a] * (*pEV++);
		}
	}
	initMesh.fastShapeChangesToMesh_AVX(vShape, &evecCache[0], validMask);
	return initMesh;
}
VertexVec CShapePose::getModelByPose(const VertexVec& basePoints, const double *__restrict poseParamsIn)
{
	// Read object model
	CMesh initMesh(initMesh_bk, &basePoints);

	// update joints
	initMesh.updateJntPosEx();

	// read motion params from the precomputed 3D poses
	CVector<double> mParams(POSPARAM_NUM);
	for (uint32_t i = 0; i < POSPARAM_NUM; i++)
	{
		mParams(i) = poseParamsIn[i];
	}
	CMatrix<float> mRBM(4, 4);
	NRBM::RVT2RBM(&mParams, mRBM);

	CVector<float> TW(initMesh.joints() + 6);
	for (int j = 6; j < mParams.size(); ++j)
	{
		TW(j) = (float)mParams(j);
	}

	SQMat4x4 newM[26];
	initMesh.angleToMatrixEx(mRBM, TW, newM);

	// rotate joints
	initMesh.rigidMotionSim_AVX(newM, true);

	return std::move(initMesh.vPoints);
}
VertexVec CShapePose::getModelByPose2(const CMesh& baseMesh, const double *__restrict poseParamsIn, const char *__restrict validMask)
{
	// Read object model
	CMesh initMesh(initMesh_bk, &baseMesh.vPoints, &baseMesh.validPts);
	// update joints
	initMesh.updateJntPosEx();

	// read motion params from the precomputed 3D poses
	CVector<double> mParams(POSPARAM_NUM);
	for (uint32_t i = 0; i < POSPARAM_NUM; i++)
	{
		mParams(i) = poseParamsIn[i];
	}
	CMatrix<float> mRBM(4, 4);
	NRBM::RVT2RBM(&mParams, mRBM);
	CVector<float> TW(initMesh.joints() + 6);
	for (int j = 6; j < mParams.size(); ++j)
	{
		TW(j) = (float)mParams(j);
	}
	SQMat4x4 newM[26];
	initMesh.angleToMatrixEx(mRBM, TW, newM);
	// rotate joints
	initMesh.rigidMotionSim2_AVX(newM, true);

	return std::move(initMesh.validPts);
}
VertexVec CShapePose::getModelFast(const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn)
{
	const double *eigenVectorsIn = evectors.memptr();/* nEigenVec x EVALUATE_POINTS_NUM(6449) x 3*/
	const uint32_t numEigenVectors = evectors.n_rows;
	
	// Read object model
	CMesh initMesh(initMesh_bk, nullptr);

	Vertex vShape[5];
	{
		float *__restrict pShape = vShape[0];
		const float *pEV = evalue[0];
		for (uint32_t a = 0; a < 20; ++a)
		{
			*pShape++ = shapeParamsIn[a] * (*pEV++);
		}
	}
	initMesh.fastShapeChangesToMesh_AVX(vShape, &evecCache[0]);

	// update joints
	initMesh.updateJntPosEx();

	// read motion params from the precomputed 3D poses
	CVector<double> mParams(POSPARAM_NUM);
	for (uint32_t i = 0; i < POSPARAM_NUM; i++)
	{
		mParams(i) = poseParamsIn[i];
	}
	CMatrix<float> mRBM(4, 4);
	NRBM::RVT2RBM(&mParams, mRBM);

	CVector<float> TW(initMesh.joints() + 6);
	for (int j = 6; j < mParams.size(); ++j)
	{
		TW(j) = (float)mParams(j);
	}

	SQMat4x4 newM[26];
	initMesh.angleToMatrixEx(mRBM, TW, newM);
	// rotate joints
	initMesh.rigidMotionSim_AVX(newM, true);

	return std::move(initMesh.vPoints);
}
VertexVec CShapePose::getModelFast2(const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn, const char *__restrict validMask)
{
	const double *eigenVectorsIn = evectors.memptr();/* nEigenVec x EVALUATE_POINTS_NUM(6449) x 3*/
	const uint32_t numEigenVectors = evectors.n_rows;

	// Read object model
	CMesh initMesh(initMesh_bk, nullptr);

	Vertex vShape[5];
	{
		float *__restrict pShape = vShape[0];
		const float *pEV = evalue[0];
		for (uint32_t a = 0; a < 20; ++a)
		{
			*pShape++ = shapeParamsIn[a] * (*pEV++);
		}
	}
	initMesh.fastShapeChangesToMesh_AVX(vShape, &evecCache[0], validMask);

	// update joints
	initMesh.updateJntPosEx();

	// read motion params from the precomputed 3D poses
	CVector<double> mParams(POSPARAM_NUM);
	for (uint32_t i = 0; i < POSPARAM_NUM; i++)
	{
		mParams(i) = poseParamsIn[i];
	}
	CMatrix<float> mRBM(4, 4);
	NRBM::RVT2RBM(&mParams, mRBM);

	CVector<float> TW(initMesh.joints() + 6);
	for (int j = 6; j < mParams.size(); ++j)
	{
		TW(j) = (float)mParams(j);
	}

	SQMat4x4 newM[26];
	initMesh.angleToMatrixEx(mRBM, TW, newM);
	// rotate joints
	initMesh.rigidMotionSim2_AVX(newM, true);

	return std::move(initMesh.validPts);
}

void CShapePose::getModel(const double *shapeParamsIn, const double *poseParamsIn, arma::mat &points, arma::mat &joints)
{
	const double *eigenVectorsIn = evectors.memptr();/* nEigenVec x EVALUATE_POINTS_NUM(6449) x 3*/
	const uint32_t numEigenVectors = evectors.n_rows;

	const float *pEV = evalue[0];
	double shapePara[20];
	for (uint32_t a = 0; a < 20; ++a)
		shapePara[a] = shapeParamsIn[a] * pEV[a];

	//For return data
	points.zeros(EVALUATE_POINTS_NUM, 3);
	double *pointsOut = points.memptr();/* EVALUATE_POINTS_NUM(6449) x 3*/
	joints.zeros(25, 8);
	double *jointsOut = joints.memptr();/* 24x8: jid directions_XYZ positions_XYZ jparent_id*/
	getpose(poseParamsIn, shapePara, eigenVectorsIn, numEigenVectors, pointsOut, jointsOut);
}

void CShapePose::getModel(const arma::mat &shapeParam, const arma::mat &poseParam, arma::mat &points, arma::mat &joints)
{
	const double *poseParamsIn = poseParam.memptr();/* 31x1*/
	const double *shapeParamsIn = shapeParam.memptr();/* 20x1*/
	getModel(shapeParamsIn, poseParamsIn, points, joints);
}

void CShapePose::setEvectors(arma::mat &evectorsIn)
{
	evectors = evectorsIn;

	const uint32_t numEigenVectors = evectorsIn.n_rows;
	if (numEigenVectors != 20)
	{
		printf("evecctors should be 20 for optimization.\n\n");
		getchar();
		exit(-1);
	}
	const uint32_t rows = evectorsIn.n_cols / 3, gap = evectorsIn.n_elem / 3;
	evecCache.resize(rows * 64);//15+1 vertex for a row
	float *pVert = evecCache[0];
	//20 * x(rows*3)
	const double *px = evectorsIn.memptr(), *py = px + gap, *pz = py + gap;
	for (uint32_t a = 0; a < rows; ++a)
	{
		for (uint32_t b = 0; b < 20; ++b)
			*pVert++ = *px++;
		for (uint32_t b = 0; b < 20; ++b)
			*pVert++ = *py++;
		for (uint32_t b = 0; b < 20; ++b)
			*pVert++ = *pz++;
		pVert += 4;
	}
}

void CShapePose::setEvalues(const arma::mat & inEV)
{
	const uint32_t num = inEV.n_elem;
	if (num != 20)
	{
		printf("evelues should be 20 for optimization.\n\n");
		getchar();
		exit(-1);
	}
	const auto *pIn = inEV.memptr();
	float *pOut = evalue[0];
	for (uint32_t a = num; a--;)
		*pOut++ = *pIn++;
	
	evalue[0].do_sqrt();
	evalue[1].do_sqrt();
	evalue[2].do_sqrt();
	evalue[3].do_sqrt();
	evalue[4].do_sqrt();
}
