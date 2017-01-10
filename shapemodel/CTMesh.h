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
	miniBLAS::Vertex vDir, vPoint, vMom, vDM;
protected:
	// Defines joint's position and axis
	CVector<float> mDirection;
	CVector<float> mPoint;
	CVector<float> mMoment;
};
/*an enhanced class of CJoint
 *Cjoint is based on CVector, which will cost lots of time when copying(allocate&free heap wastefully)
 *This version use Vertex instead(data stores on stack)
 */
ALIGN16 struct CJointEx
{
public:
	// constructor
	CJointEx() { };
	CJointEx& operator=(const CJoint& from)
	{
		vDir = from.vDir; vPoint = from.vPoint; vMom = from.vMom; vDM = from.vDM; mParent = from.mParent;
		return *this;
	};
	// Constructs the motion matrix from the joint axis and the rotation angle
	miniBLAS::SQMat4x4 angleToMatrixEx(const float aAngle) const;
	// Access to joint's position and axis
	void set(const CVector<float>& aDirection, const CVector<float>& aPoint)
	{
		vDir.assign(aDirection.data());
		vPoint.assign(aPoint.data());
		vMom = vPoint * vDir; vDM = vDir * vMom;
	};
	void setDirection(const CVector<float>& aDirection)
	{
		vDir.assign(aDirection.data());
		vMom = vPoint * vDir; vDM = vDir * vMom;
	};
	void setPoint(const miniBLAS::Vertex& pt)
	{
		vPoint = pt;
		vMom = vPoint * vDir; vDM = vDir * vMom;
	}
	void setPoint(const CVector<float>& aPoint)
	{
		vPoint.assign(aPoint.data());
		vMom = vPoint * vDir; vDM = vDir * vMom;
	};
	// Defines joint's position and axis, vDM is a pre-compute value
	miniBLAS::Vertex vDir, vPoint, vMom, vDM;
	// Parent joint
	int mParent = 0;
};
constexpr uint32_t kksize_CJEX = sizeof(CJointEx);

/*smooth-params for model
 *one point could be influenced by many joints(hence it's smooth), each influence is stored in a SmoothParam
 *put records with 1 joint influenced into ptSmooth first, then 2,3,4joints(a point could be influenced at most 4 joints)
 *and save the number of records of each type into smtCnt
 *in fact the data structure could be vector<SmoothParam>[4] for easier understanding
 *however putting them in a continuous vector may be more friendly to cache
 */
struct ModelSmooth
{
	struct SmoothParam
	{
		//point index
		uint16_t pid;
		//joint index
		uint16_t jid;
		float weight;
	};
	std::vector<SmoothParam> ptSmooth;
	uint32_t smtCnt[4] = { 0 };
};
/*ModelSmooth could be shared in fastcopy of mesh, or other condition, so use shared_ptr to handle it*/
using PtrModSmooth = std::shared_ptr<ModelSmooth>;

class CMesh
{
public:
	static atomic_uint64_t functime[8];
	static atomic_uint32_t funccount[8];
	constexpr static uint32_t mJointNumber = POSPARAM_NUM - 6;
	//joint index is based on 1, so need to add 1
	constexpr static uint32_t MotionMatCnt = mJointNumber + 1;
	using MotionMat = std::array<miniBLAS::SQMat4x4, MotionMatCnt>;

	PtrModSmooth modsmooth;

	CMesh()
	{
		evecCache.reset(new miniBLAS::VertexVec());
		modsmooth.reset(new ModelSmooth());
		sh2jnt.reset(new std::array<ShapeJointParam, 14>());
	}
	CMesh(const CMesh& from, const bool isFastCopy);// = false);
	~CMesh() = default;

	void setShapeSpaceEigens(const arma::mat &evectorsIn);
	void prepareData();
	uint32_t getPointsCount() const { return mNumPoints; }
	/*pre-compute mesh data based on validmask
	 *when we can determine some points invalid inmatch(it will not be used in final cost calculation),
	 *we can avoid calculating these points in rigidMotion step and just output useful points data
	 *This function compute a new ModelSmooth based on the valid-mask
	 **/
	PtrModSmooth preCompute(const int8_t *__restrict validMask) const;
	void writeMeshDat(std::string fname);
	void printPoints(std::string fname);
	//apply shape-params' influences on body's points' pos
	int shapeChangesToMesh(CVector<float> shapeParams, const std::vector<CMatrix<double> >& eigenVectors);
	//apply shape-params' influences on body's points' pos
	void fastShapeChangesToMesh(const double *shapeParamsIn, const uint32_t numEigenVectors, const double *eigenVectorsIn);
	//apply shape-params' influences on body's points' pos
	void fastShapeChangesToMesh(const miniBLAS::Vertex *shapeParamsIn);
	//apply shape-params' influences on body's points' pos
	//This function additionally calculate useful points and store them to validPts
	void fastShapeChangesToMesh(const miniBLAS::Vertex *shapeParamsIn, const int8_t *__restrict validMask);
	int updateJntPos();
	//after shape-params influence mesh points, the joints' actual position will be influenced too, hence need to update their pos
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
	// Apply motion to points
	void rigidMotion(CVector<CMatrix<float> >& M, CVector<float>& X, bool smooth = false, bool force = false);
	void rigidMotionSim_AVX(const MotionMat& M);
	//this function only calculte useful points(validPts) and stores in validPts
	void rigidMotionSim2_AVX(const MotionMat& M);
	void smoothMotionDQ(CVector<CMatrix<float> >& M, CVector<float>& X);
	// Reuses InitMesh to set up Smooth Pose: Global transformation
	void makeSmooth(CMesh* initMesh, bool dual = false);

	void angleToMatrix(const CMatrix<float>& aRBM, CVector<float>& aJAngles, CVector<CMatrix<float> >& M);
	MotionMat angleToMatrixEx(const CMatrix<float>& aRBM, const double * const aJAngles) const;
	void invAngleToMatrix(const CMatrix<float>& aRBM, CVector<float>& aJAngles, CVector<CMatrix<float> >& M);
	void twistToMatrix(CVector<float>& aTwist, CVector<CMatrix<float> >& M);

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

	miniBLAS::VertexVec vPoints;
	miniBLAS::VertexVec validPts;
	miniBLAS::Vertex bodysize;
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
	//idxmap is used when update joint pos
	static const uint8_t idxmap[14][3];
	arma::mat evectors;
	std::shared_ptr<miniBLAS::VertexVec> evecCache;

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

	CVector<CJoint>  mJoint;
	std::array<CJointEx, MotionMatCnt> vJoint;

	CMatrix<float> weightMatrix;

	// true if aParentJoint is an ancestor of aJoint
	bool isParentOf(int aParentJointID, int aJointID);
	bool isParentOfEx(int aParentJointID, int aJointID);
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
void CMesh::GetPoint(const int i, T& x, T& y, T& z)
{
	const auto obj = mPoints[i];
	x = T(obj[0]);
	y = T(obj[1]);
	z = T(obj[2]);
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

