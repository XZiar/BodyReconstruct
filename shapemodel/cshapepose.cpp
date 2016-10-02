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

int CShapePose::getpose(/*const string inputDir, */const double* motionParamsIn, const double* shapeParamsIn, double *eigenVectorsIn,
	uint32_t numEigenVectors, double* pointsOut, double* jointsOut)
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
	// read motion params from the precomputed 3D poses
	const uint32_t numMotionParams = 31;
	CVector<double> mParams(numMotionParams);
	for (uint32_t i = 0; i < numMotionParams; i++)
	{
		mParams(i) = motionParamsIn[i];
	}
#ifndef EXEC_FAST
	cout << "3D pose parameters:" << endl;
	cout << mParams << endl;
#endif

#ifndef EXEC_FAST
	cout << "semantic shape parameters: ";
#endif
	/*
	CVector<double> shapeParams(numEigenVectors);
	for (uint32_t i0 = 0; i0 < numEigenVectors; i0++)
	{
		shapeParams[i0] = shapeParamsIn[i0];
	}
	*/
	CVector<float> shapeParamsFloat(numEigenVectors);
	for (uint32_t i = 0; i < numEigenVectors; i++)
		shapeParamsFloat(i) = float(shapeParamsIn[i]);

	// Read object model
	CMatrix<float> mRBM(4, 4);
	NRBM::RVT2RBM(&mParams, mRBM);

	CMesh initMesh = initMesh_bk;
	/*
	auto aModelFile = "../BodyReconstruct/data/model.dat";
	initMesh.readModel(aModelFile, SMOOTHMODEL);
	initMesh.updateJntPos();
	initMesh.centerModel();
	*/
	// use eingevectors passed as an argument
	EigenVec ev = CMesh::readShapeSpaceEigens(eigenVectorsIn, numEigenVectors, initMesh.GetPointSize());

	// reshape the model
	initMesh.shapeChangesToMesh(shapeParamsFloat, ev);

	// update joints
	initMesh.updateJntPos();

	CVector<CMatrix<float> > M(initMesh.joints() + 1);
	CVector<float> TW(initMesh.joints() + 6);
	float tmp[100];

	for (int j = 6; j < mParams.size(); ++j)
	{
		tmp[j] = (float)mParams(j);
		TW(j) = (float)mParams(j);
	}

	initMesh.angleToMatrix(mRBM, TW, M);

	// rotate joints
	initMesh.rigidMotion(M, TW, true, true);

	// Fill in resulting joints array
	int nJoints = initMesh.joints();
	for (int i = 0; i < nJoints; i++)
	{
		CJoint joint = initMesh.joint(i + 1);
		jointsOut[i] = i + 1;
		jointsOut[i + nJoints] = joint.getDirection()[0];
		jointsOut[i + nJoints * 2] = joint.getDirection()[1];
		jointsOut[i + nJoints * 3] = joint.getDirection()[2];
		jointsOut[i + nJoints * 4] = joint.getPoint()[0];
		jointsOut[i + nJoints * 5] = joint.getPoint()[1];
		jointsOut[i + nJoints * 6] = joint.getPoint()[2];
		jointsOut[i + nJoints * 7] = double(joint.mParent);
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

void CShapePose::getModel(const arma::mat &shapeParam, const arma::mat &poseParam, arma::mat &points, arma::mat &joints)
{
	uint64_t t1, t2;
	t1 = getCurTime();

	int m, n;
	double *pointsOut; /* 6449x3*/
	double *jointsOut; /* 24x8: jid directions_XYZ positions_XYZ jparent_id*/
	const double *motionParamsIn;  /* 31x1*/
	const double *shapeParamsIn;  /* 20x1*/
	double *eigenVectorsIn; /* nEigenVec x 6449 x 3*/
	//For return data
	m = 6449; n = 3;
	points = points.zeros(m, n);
	m = 25; n = 8;
	joints = joints.zeros(m, n);
	///
	motionParamsIn = poseParam.memptr();
	shapeParamsIn = shapeParam.memptr();
	eigenVectorsIn = evectors.memptr();
	uint32_t numEigenVectors = evectors.n_rows;
	//
	//  cout<<"in shape pose====================="<<endl;
	//  cout<<poseParam<<endl;
	//  cout<<evectors.n_rows<<","<<evectors.n_cols<<endl;
	//  cout<<shapeParam<<endl;
	//  cin.ignore();
	///
	pointsOut = points.memptr();
	jointsOut = joints.memptr();
	getpose(motionParamsIn, shapeParamsIn, eigenVectorsIn, numEigenVectors, pointsOut, jointsOut);

	t2 = getCurTime();
	//printf("getModel uses %lld ms.\n", t2 - t1);
}

void CShapePose::setEvectors(arma::mat &evectorsIn)
{
	evectors = evectorsIn;
}
