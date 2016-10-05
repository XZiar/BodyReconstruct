#include "cshapepose.h"

#include "NRBM.h"
#include "onlyDefines.h"
#include "paramMap.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

using EigenVec = vector<CMatrix<double> >;

CShapePose::CShapePose()
{
	auto aModelFile = "../BodyReconstruct/data/model.dat";
	initMesh_bk.readModel(aModelFile, SMOOTHMODEL);
	initMesh_bk.updateJntPos();
	initMesh_bk.centerModel();
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

	if (isFastChange)
		initMesh.fastShapeChangesToMesh(shapeParamsIn, numEigenVectors, eigenVectorsIn);
	else
	{
		// use eingevectors passed as an argument
		EigenVec ev = CMesh::readShapeSpaceEigens(eigenVectorsIn, numEigenVectors, initMesh.GetPointSize());
		CVector<float> shapeParamsFloat(numEigenVectors);
		for (uint32_t i = 0; i < numEigenVectors; i++)
			shapeParamsFloat(i) = float(shapeParamsIn[i]);
		// reshape the model
		initMesh.shapeChangesToMesh(shapeParamsFloat, ev);
	}	

	// update joints
	initMesh.updateJntPos();

	// read motion params from the precomputed 3D poses
	const uint32_t numMotionParams = 31;
	CVector<double> mParams(numMotionParams);
	for (uint32_t i = 0; i < numMotionParams; i++)
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

void CShapePose::getModelFast(const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn, float *__restrict pointsOut)
{
	const double *eigenVectorsIn = evectors.memptr();/* nEigenVec x 6449 x 3*/
	const uint32_t numEigenVectors = evectors.n_rows;
	
	// Read object model
	CMesh initMesh = initMesh_bk;
	//initMesh.fastShapeChangesToMesh(shapeParamsIn, numEigenVectors, eigenVectorsIn);
	miniBLAS::Vertex vShape[5];
	{
		float *__restrict pShape = vShape[0];
		for (uint32_t a = 0; a < 20; ++a)
			*pShape++ = shapeParamsIn[a];
	}
	initMesh.fastShapeChangesToMesh(vShape, evecCache);

	// update joints
	initMesh.updateJntPos();

	// read motion params from the precomputed 3D poses
	const uint32_t numMotionParams = 31;
	CVector<double> mParams(numMotionParams);
	for (uint32_t i = 0; i < numMotionParams; i++)
	{
		mParams(i) = poseParamsIn[i];
	}
	CMatrix<float> mRBM(4, 4);
	NRBM::RVT2RBM(&mParams, mRBM);

	CVector<CMatrix<float> > M(initMesh.joints() + 1);
	CVector<float> TW(initMesh.joints() + 6);
	for (int j = 6; j < mParams.size(); ++j)
	{
		TW(j) = (float)mParams(j);
	}
	initMesh.angleToMatrix(mRBM, TW, M);

	// rotate joints
	initMesh.rigidMotion(M, TW, true, true);

	// Fill in resulting points array
	const int nPoints = initMesh.GetPointSize();
	for (int i = 0; i < nPoints; pointsOut += 4)
	{
		initMesh.GetPoint3(i++, pointsOut);
	}
}

void CShapePose::getModelFastEx(const double *__restrict shapeParamsIn, const double *__restrict poseParamsIn, float *__restrict pointsOut)
{
	const double *eigenVectorsIn = evectors.memptr();/* nEigenVec x 6449 x 3*/
	const uint32_t numEigenVectors = evectors.n_rows;

	// Read object model
	CMesh initMesh = initMesh_bk;
	miniBLAS::Vertex vShape[5];
	{
		float *pShape = vShape[0];
		for (uint32_t a = 0; a < 20; ++a)
			*pShape++ = shapeParamsIn[a];
	}
	uint64_t t1, t2, t3;
	t1 = getCurTimeNS();
	initMesh.fastShapeChangesToMesh(shapeParamsIn, numEigenVectors, eigenVectorsIn);
	t2 = getCurTimeNS();
	initMesh.fastShapeChangesToMesh(vShape, evecCache);
	t3 = getCurTimeNS();
	printf("#$#$#$#$fastShapeChangesToMesh  COST\nOLD %lld ns, NEW %lld ns\n", t2 - t1, t3 - t2);
	getchar();

	// update joints
	initMesh.updateJntPos();

	// read motion params from the precomputed 3D poses
	const uint32_t numMotionParams = 31;
	CVector<double> mParams(numMotionParams);
	for (uint32_t i = 0; i < numMotionParams; i++)
	{
		mParams(i) = poseParamsIn[i];
	}
	CMatrix<float> mRBM(4, 4);
	NRBM::RVT2RBM(&mParams, mRBM);

	CVector<CMatrix<float> > M(initMesh.joints() + 1);
	CVector<float> TW(initMesh.joints() + 6);
	for (int j = 6; j < mParams.size(); ++j)
	{
		TW(j) = (float)mParams(j);
	}
	initMesh.angleToMatrix(mRBM, TW, M);

	// rotate joints
	initMesh.rigidMotion(M, TW, true, true);

	// Fill in resulting points array
	const int nPoints = initMesh.GetPointSize();
	for (int i = 0; i < nPoints; pointsOut += 4)
	{
		initMesh.GetPoint3(i++, pointsOut);
	}
}

void CShapePose::getModel(const double *shapeParamsIn, const double *poseParamsIn, arma::mat &points, arma::mat &joints)
{
	const double *eigenVectorsIn = evectors.memptr();/* nEigenVec x 6449 x 3*/
	const uint32_t numEigenVectors = evectors.n_rows;
	//
	//  cout<<"in shape pose====================="<<endl;
	//  cout<<poseParam<<endl;
	//  cout<<evectors.n_rows<<","<<evectors.n_cols<<endl;
	//  cout<<shapeParam<<endl;
	//  cin.ignore();

	//For return data
	points.zeros(6449, 3);
	double *pointsOut = points.memptr();/* 6449x3*/
	joints.zeros(25, 8);
	double *jointsOut = joints.memptr();/* 24x8: jid directions_XYZ positions_XYZ jparent_id*/
	getpose(poseParamsIn, shapeParamsIn, eigenVectorsIn, numEigenVectors, pointsOut, jointsOut);
}

void CShapePose::getModel(const arma::mat &shapeParam, const arma::mat &poseParam, arma::mat &points, arma::mat &joints)
{
	const double *motionParamsIn = poseParam.memptr();/* 31x1*/
	const double *shapeParamsIn = shapeParam.memptr();/* 20x1*/
	const double *eigenVectorsIn = evectors.memptr();/* nEigenVec x 6449 x 3*/
	const uint32_t numEigenVectors = evectors.n_rows;
	//
	//  cout<<"in shape pose====================="<<endl;
	//  cout<<poseParam<<endl;
	//  cout<<evectors.n_rows<<","<<evectors.n_cols<<endl;
	//  cout<<shapeParam<<endl;
	//  cin.ignore();

	//For return data
	points.zeros(6449, 3);
	double *pointsOut = points.memptr();/* 6449x3*/
	joints.zeros(25, 8);
	double *jointsOut = joints.memptr();/* 24x8: jid directions_XYZ positions_XYZ jparent_id*/
	getpose(motionParamsIn, shapeParamsIn, eigenVectorsIn, numEigenVectors, pointsOut, jointsOut);
}

void CShapePose::setEvectors(arma::mat &evectorsIn)
{
	using miniBLAS::Vertex;
	evectors = evectorsIn;
	if (evecCache != nullptr)
	{
		delete[] evecCache;
	}
	const uint32_t numEigenVectors = evectorsIn.n_rows;
	if (numEigenVectors != 20)
	{
		printf("evecctors should be 20 for optimization.\n\n");
		getchar();
		exit(-1);
	}
	evecCache = new Vertex[evectorsIn.n_elem / 4];
	float *pVert = evecCache[0];
	//20 * x(rows*3)
	const uint32_t rows = evectorsIn.n_cols / 3, gap = evectorsIn.n_elem / 3;
	const double *px = evectorsIn.memptr(), *py = px + gap, *pz = py + gap;
	for (uint32_t a = 0; a < rows; ++a)
	{
		for (uint32_t b = 0; b < 20; ++b)
			*pVert++ = *px++;
		for (uint32_t b = 0; b < 20; ++b)
			*pVert++ = *py++;
		for (uint32_t b = 0; b < 20; ++b)
			*pVert++ = *pz++;
	}

}
