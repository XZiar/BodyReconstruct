/**
	This file is part of the implementation of the 3D human shape model as described in the paper:

	Leonid Pishchulin, Stefanie Wuhrer, Thomas Helten, Christian Theobalt and Bernt Schiele
	Building Statistical Shape Spaces for 3D Human Modeling
	ArXiv, March 2015

	Please cite the paper if you are using this code in your work.

	Contributors: Arjun Jain, Juergen Gall, Leonid Pishchulin.
	This code also uses open source functionality for matrix operations by Thomas Brox.

	The code may be used free of charge for non-commercial and
	educational purposes, the only requirement is that this text is
	preserved within the derivative work. For any other purpose you
	must contact the authors for permission. This code may not be
	redistributed without permission from the authors.
*/

#ifndef CMeshH
#define CMeshH

#include "main.h"

#include "CMatrix.h"
#include "CTensor.h"
#include "Show.h"

#include "miniBLAS.hpp"

#define NO_OF_EIGENVECTORS 20
#define SKEL_SCALE_FACTOR 1.0

#define EXEC_FAST 1
//#include "RenderWidget.h"

class CJoint
{
public:
	// constructor
	inline CJoint() { mParent = 0; mDirection.setSize(0); mPoint.setSize(0); mMoment.setSize(0); };
	~CJoint() { };
	inline CJoint(CJoint& aCopyFrom) { *this = aCopyFrom; };
	CJoint(CVector<float>& aDirection, CVector<float>& aPoint, int aParent);
	// Performs a rigid motion M of the joint
	void rigidMotion(CMatrix<float>& M);
	// Constructs the motion matrix from the joint axis and the rotation angle
	void angleToMatrix(float aAngle, CMatrix<float>& M);
	void angleToMatrixEx(const float aAngle, CMatrix<float>& M);
	miniBLAS::SQMat4x4 angleToMatrixEx(const float aAngle);
	// Access to joint's position and axis
	inline void set(const CVector<float>& aDirection, const CVector<float>& aPoint)
	{
		mDirection = aDirection; vDir.assign(aDirection.data());
		mPoint = aPoint; vPoint.assign(aPoint.data());
		mMoment = aPoint / aDirection; vMom = vPoint * vDir; vDM = vDir *vMom;
	};
	inline void setDirection(const CVector<float>& aDirection)
	{ 
		mDirection = aDirection; vDir.assign(aDirection.data());
		mMoment = mPoint / aDirection; vMom = vPoint * vDir; vDM = vDir *vMom;
	};
	inline void setPoint(const CVector<float>& aPoint) 
	{ 
		mPoint = aPoint; vPoint.assign(aPoint.data());
		mMoment = aPoint / mDirection; vMom = vPoint * vDir; vDM = vDir *vMom;
	};
	inline CVector<float>& getDirection() { return mDirection; };
	inline CVector<float>& getPoint() { return mPoint; };
	inline CVector<float>& getMoment() { return mMoment; };
	// Copy operator
	CJoint& operator=(CJoint& aCopyFrom);
	// Parent joint
	int mParent;
protected:
	// Defines joint's position and axis
	CVector<float> mDirection;
	CVector<float> mPoint;
	CVector<float> mMoment;
	miniBLAS::Vertex vDir, vPoint, vMom, vDM;
};

class CMeshMotion
{
public:
	// constructor
	inline CMeshMotion() { };
	// Resets the model to zero motion
	void reset(int aJointNumber);
	// Shows current configuration
	void print();
	// Writes current configuration to file
	void writeToFile();
	// Copy operator
	CMeshMotion& operator=(const CMeshMotion& aCopyFrom);
	// Access to pose parameters
	inline float& operator()(int aIndex) const { return mPoseParameters(aIndex); };

	// Main body motion
	CMatrix<float> mRBM;
	// Vector with all pose parameters (including joint angles)
	CVector<float> mPoseParameters;
};

/* class CAppearanceModel { */
/* public: */
/*   inline CAppearanceModel() {}; */
/*   CVector<int> mLastFrameVisible; */
/*   CTensor<float> mHistogram; */
/* }; */


struct ModelSmooth
{
	struct SmoothParam
	{
		uint32_t idx;
		float weight;
	};
	std::vector<uint32_t> smtCnt;
	std::vector<SmoothParam> ptSmooth;
};
using PtrModSmooth = std::shared_ptr<ModelSmooth>;

class CMesh
{
public:
	static atomic_uint64_t functime[8];
	static atomic_uint32_t funccount[8];
	constexpr static uint32_t mJointNumber = POSPARAM_NUM - 6;
	constexpr static uint32_t MotionMatCnt = mJointNumber + 1;
	using MotionMat = std::array<miniBLAS::SQMat4x4, MotionMatCnt>;

	PtrModSmooth modsmooth;

	CMesh()
	{
		evecCache.reset(new miniBLAS::VertexVec());
		evecCache2.reset(new miniBLAS::VertexVec());
		modsmooth.reset(new ModelSmooth());
		sh2jnt.reset(new std::array<ShapeJointParam, 14>());
	}
	CMesh(const CMesh& aMesh) { *this = aMesh; };
	CMesh(const CMesh& from, const miniBLAS::VertexVec *pointsIn);
	CMesh(const CMesh& from, const CMesh& baseMesh, const PtrModSmooth msmooth);
	~CMesh() = default;

	void setShapeSpaceEigens(const arma::mat &evectorsIn);
	void prepareData();
	PtrModSmooth preCompute(const int8_t *__restrict validMask) const;
	void writeMeshDat(std::string fname);
	void printPoints(std::string fname);
	int shapeChangesToMesh(CVector<float> shapeParams, const std::vector<CMatrix<double> >& eigenVectors);
	void fastShapeChangesToMesh(const double *shapeParamsIn, const uint32_t numEigenVectors, const double *eigenVectorsIn);
	void fastShapeChangesToMesh(const miniBLAS::Vertex *shapeParamsIn);
	//void NEWfastShapeChangesToMesh_AVX(const miniBLAS::Vertex *shapeParamsIn);
	//void NEWfastShapeChangesToMesh_AVX(const miniBLAS::Vertex *shapeParamsIn, const int8_t *__restrict validMask);
	void fastShapeChangesToMesh_AVX(const miniBLAS::Vertex *shapeParamsIn);
	void fastShapeChangesToMesh_AVX(const miniBLAS::Vertex *shapeParamsIn, const int8_t *__restrict validMask);
	int updateJntPos();
	void updateJntPosEx();
	
	// Reads the mesh from a file
	// liuyebin collect the readmodel functions together. The first is for initISA, and the second is for trackHybOpt
	bool readModel(const char* aFilename, bool smooth = false);
	bool adaptOFF(const char* aFilename, float lambda);
	bool readOFF(const char* aFilename);
	void centerModel();
	// Writes the mesh to a file
	void writeModel(const char* aFilename);
	void writeAdaptModel(const char* aFilename, const CMesh* adM);
	void writeSkel(const char* aFilename);

	//void draw();

	// Performs rigid body motions M of the mesh points in the kinematic chain
	void rigidMotion(CVector<CMatrix<float> >& M, CVector<float>& X, bool smooth = false, bool force = false);
	void rigidMotionSim_AVX(const MotionMat& M, const bool smooth = false);
	void rigidMotionSim2_AVX(const MotionMat& M, const bool smooth = false);
	void smoothMotionDQ(CVector<CMatrix<float> >& M, CVector<float>& X);
	// Reuses InitMesh to set up Smooth Pose: Global transformation
	void makeSmooth(CMesh* initMesh, bool dual = false);

	void angleToMatrix(const CMatrix<float>& aRBM, CVector<float>& aJAngles, CVector<CMatrix<float> >& M);
	MotionMat angleToMatrixEx(const CMatrix<float>& aRBM, const double * const aJAngles);
	void invAngleToMatrix(const CMatrix<float>& aRBM, CVector<float>& aJAngles, CVector<CMatrix<float> >& M);
	void twistToMatrix(CVector<float>& aTwist, CVector<CMatrix<float> >& M);

	// Fast projection of a 3-D point to the image plane
	template<typename T>
	void projectPoint(const CMatrix<float>& P, const float X, const float Y, const float Z, T& x, T& y);

	// Copies aMesh
	void operator=(const CMesh& aMesh);

	void setJointDir(int aJointID, CVector<float> dir)
	{
		mJoint(aJointID).setDirection(dir);
	};

	// Returns a joint
	CJoint& joint(int aJointID) { return mJoint(aJointID); };
	// turns whether a point is influenced by a certain joint
	bool influencedBy(int aJointIDOfPoint, int aJointID) { return mInfluencedBy(aJointIDOfPoint, aJointID); };
	bool isNeighbor(int i, int j) { return mNeighbor(mJointMap(i), mJointMap(j)); };
	bool isEndJoint(int aJointID) { return mEndJoint[aJointID]; }

	template<typename T>
	void GetPoint(const int i, T& x, T& y, T& z);
	template<typename T, typename T2>
	void GetPoint(const int i, T& x, T& y, T& z, T2& w);
	inline void GetPatch(int i, int& x, int& y, int& z);
	inline void GetBounds(int J, int i, float& x, float& y, float& z);
	int GetBoundJID(int J) { return (int)mBounds(J, 8, 0); };
	float GetCenter(int i) { return mCenter[i]; };
	int GetBodyPart(int jID) { return mJointMap[jID]; };

	int GetMeshSize() { return mNumPatch; };
	int GetPointSize() { return mNumPoints; };
	int GetBoundSize() { return mBounds.xSize(); };

	int GetJointID(int i) { return (int)mPoints[i](3); };

	bool IsCovered(int i) { return mCovered[i]; };
	bool IsExtremity(int i) { return mExtremity[i]; };

	void projectSurface(CMatrix<float>& aImage, CMatrix<float>& P, int xoffset = 0, int yoffset = 0);


	// Projects the mesh to the image plane given the projection matrix P
	void projectToImage(CMatrix<float>& aImage, CMatrix<float>& P, int aLineVal = 255);
	void projectToImage(CMatrix<float>& aImage, CMatrix<float>& P, int aLineVal, int aNodeVal);
	void projectToImage(CTensor<float>& aImage, CMatrix<float>& P, int aLineR = 255, int aLineG = 255, int aLineB = 255);
	void projectToImage(CTensor<float>& aImage, CMatrix<float>& P, int aLineR, int aLineG, int aLineB, int aNodeR, int aNodeG, int aNodeB);
	// Just project a joint index + neighbor ...
	void projectToImageJ(CMatrix<float>& aImage, CMatrix<float>& P, int aLineValue, int aJoint, int xoffset = 0, int yoffset = 0);

	// Projects all mesh points to the image plane
	void projectPointsToImage(CMatrix<float>& aImage, CMatrix<float>& P, int aNodeVal = 128);
	// Projects all mesh points to the image plane and returns the 3-D coordinates of these points
	void projectPointsToImage(CTensor<float>& a3DCoords, CMatrix<float>& P);

	// Initialization of appearance model
	//void initializeAppearanceModel(int aFeatureSize, int aViewCount);
	// Update of appearance model and occlusion detection
	//void updateAppearance(CTensor<float>& aData, CMatrix<float>& aOccluded, CMatrix<float>& P, int aViewNo, int aFrameNo, bool aOcclusionCheckOnly); 

	// Writes accumulated motion to standard output
	//inline void printAccumulatedMotion() {mAccumulatedMotion.print();};
	// Writes accumulated motion to file
	//inline void writeAccumulatedMotion() {mAccumulatedMotion.writeToFile();};
	// Resets accumulated motion to 0
	inline void resetAccumulation() { mAccumulatedMotion.reset(mJointNumber); };
	inline void resetCurrentMotion() { mCurrentMotion.reset(mJointNumber); };
	// Gives access to current motion
	inline CMeshMotion& currentMotion() { return mCurrentMotion; };

	miniBLAS::VertexVec vPoints;
	miniBLAS::VertexVec validPts;
protected:
	bool isCopy = false;
	void CheckCopy(const bool allow)
	{
		if (isCopy != allow)
		{
			printf("func called only allowed in %s!\n", allow ? "copy" : "non-copy");
			getchar();
			exit(-1);
		}
	}
	void CheckCut(const bool allow)
	{
		const bool isCut = (validPts.size() > 0);
		if (isCut != allow)
		{
			printf("func called only allowed in %s!\n", allow ? "cut" : "non-cut");
			getchar();
			exit(-1);
		}
	}
	static const uint8_t idxmap[14][3];
	arma::mat evectors;
	std::shared_ptr<miniBLAS::VertexVec> evecCache;
	std::shared_ptr<miniBLAS::VertexVec> evecCache2;

	//miniBLAS::VertexVec wgtMat;
	uint32_t wMatGap;
	struct ShapeJointParam
	{
		miniBLAS::VertexVec influence;
		std::vector<uint32_t> idxs;
	};
	std::shared_ptr<std::array<ShapeJointParam, 14>> sh2jnt;

	std::vector<CVector<float> >  mPoints;
	std::vector<CVector<int> >  mPatch;
	CTensor<float> mBounds;
	CVector<float> mCenter;
	CVector<int> mJointMap;
	CMatrix<bool> mNeighbor;
	CVector<bool> mEndJoint;

	CVector<bool> mCovered;
	CVector<bool> mExtremity;

	int mNumPoints;
	int mNumPatch;
	int mNumSmooth; // how many joints can influence any given point
	int mNoOfBodyParts;
	CVector<bool> mBoundJoints;

	CMatrix<bool> mInfluencedBy;
	// CMatrix<float> mRBM; // Overall Rigid Body Motion;

	//CMatrix<CAppearanceModel>* mAppearanceField;
	// CTensor<bool>* mOccluded;
	CMeshMotion mAccumulatedMotion;
	CMeshMotion mCurrentMotion;
	//CVector<CMeshMotion> mHistory;
	CVector<CJoint>  mJoint;
	//static std::vector<CMatrix<double> >  eigenVectors;
	CMatrix<float> weightMatrix;

	// Sure Fields before the Draping ...
	// CVector<float>* mPointsS;

	// true if aParentJoint is an ancestor of aJoint
	bool isParentOf(int aParentJointID, int aJointID);
	//~aj - small functions, can move elsewhere (think)
	bool minMLab(CMatrix<float> weightMatrix, int i0, int i1, CVector<float> &minEle);
	void componet_wise_mul_with_pnts(CVector<float> minEle, CMatrix<float> &tmp_wtMatrix);
	void componet_wise_mul_with_pntsEx(const float *minEle, CMatrix<float> &tmp_wtMatrix);
	double sumTheVector(CVector<float> minEle);
	void sumTheMatrix(CMatrix<float> tmpMatrix, CVector<float> & t1);
	bool copyMatColToVector(CMatrix<float> weightMatrix, int i0, CVector<float> &singleWts);
	void copyJointPos(int to, int from, CMatrix<float> &newJntPos);
	void findNewAxesOrient(CVector<double> &axisLeftArm, CVector<double> &axisRightArm, CMatrix<float> newJntPos);
	void getWeightsinBuffer(char *buffer, int ptNo);
};

template<typename T>
void CMesh::projectPoint(const CMatrix<float>& P, const float X, const float Y, const float Z, T& x, T& y)
{
	const float hx = P.data()[0] * X + P.data()[1] * Y + P.data()[2] * Z + P.data()[3];
	const float hy = P.data()[4] * X + P.data()[5] * Y + P.data()[6] * Z + P.data()[7];
	const float hz = P.data()[8] * X + P.data()[9] * Y + P.data()[10] * Z + P.data()[11];
	const float invhz = 1.0f / hz;
	x = (T)(hx*invhz + 0.5f);
	y = (T)(hy*invhz + 0.5f);
}
template<typename T>
void CMesh::GetPoint(const int i, T& x, T& y, T& z)
{
	const auto obj = mPoints[i];
	x = T(obj[0]);
	y = T(obj[1]);
	z = T(obj[2]);
}
template<typename T, typename T2>
void CMesh::GetPoint(const int i, T& x, T& y, T& z, T2& w)
{
	const auto obj = mPoints[i];
	x = T(obj[0]);
	y = T(obj[1]);
	z = T(obj[2]);
	w = T(obj[3]);
}

inline void CMesh::GetPatch(int i, int& x, int& y, int& z)
{
	x = mPatch[i](0);
	y = mPatch[i](1);
	z = mPatch[i](2);
}

inline void CMesh::GetBounds(int J, int i, float& x, float& y, float& z)
{
	x = mBounds(J, i, 0);
	y = mBounds(J, i, 1);
	z = mBounds(J, i, 2);
}

#endif

