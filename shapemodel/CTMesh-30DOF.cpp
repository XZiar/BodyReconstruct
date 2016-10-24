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

#include "NMath.h"
#include "NRBM.h"
#include "CTMesh.h"
#include <fstream>
#include <assert.h>
#include <set>
#include <map>


using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::ifstream;
using std::make_pair;
using std::vector;


static void showfloat4(const char *name, const __m128 *data)
{
	float* ptr = (float*)data;
	printf("%s = %f,%f,%f,%f\n", name, ptr[0], ptr[1], ptr[2], ptr[3]);
}

// C J O I N T -----------------------------------------------------------------
// constructor
CJoint::CJoint(CVector<float>& aDirection, CVector<float>& aPoint, int aParent)
{
	mDirection = aDirection;
	mPoint = aPoint;
	mMoment = aPoint / aDirection;
	mParent = aParent;
}

// rigidMotion
void CJoint::rigidMotion(CMatrix<float>& M)
{
	CMatrix<float> R(3, 3);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			R(i, j) = M(i, j);
	mDirection = R*mDirection;
	mPoint = R*mPoint;
	mPoint(0) += M(3, 0); mPoint(1) += M(3, 1); mPoint(2) += M(3, 2);
	mMoment = mPoint / mDirection;
}

// angleToMatrix
void CJoint::angleToMatrix(float aAngle, CMatrix<float>& M)
{
	CMatrix<float> omegaHat(3, 3);
	omegaHat.data()[0] = 0.0;            omegaHat.data()[1] = -mDirection(2); omegaHat.data()[2] = mDirection(1);
	omegaHat.data()[3] = mDirection(2);  omegaHat.data()[4] = 0.0;            omegaHat.data()[5] = -mDirection(0);
	omegaHat.data()[6] = -mDirection(1); omegaHat.data()[7] = mDirection(0);  omegaHat.data()[8] = 0.0;
	CMatrix<float> omegaT(3, 3);
	for (int j = 0; j < 3; j++)
		for (int i = 0; i < 3; i++)
			omegaT(i, j) = mDirection(i)*mDirection(j);
	CMatrix<float> R(3, 3);
	R = (omegaHat*(float)sin(aAngle)) + ((omegaHat*omegaHat)*(float)(1.0 - cos(aAngle)));

	CVector<float> t = ((omegaT*mMoment)*aAngle) - R*(mDirection / mMoment);
	//printf("t{3} = %f,%f,%f\n", t(0), t(1), t(2));
	R(0, 0) += 1.0; R(1, 1) += 1.0; R(2, 2) += 1.0;
	/*
	CMatrix<float> T(R); T *= -1.0;
	T(0, 0) += 1.0; T(1, 1) += 1.0; T(2, 2) += 1.0;
	CVector<float> oldt = T*(mDirection / mMoment) + ((omegaT*mMoment)*aAngle);
	*/
	M.data()[0] = R.data()[0]; M.data()[1] = R.data()[1]; M.data()[2] = R.data()[2]; M.data()[3] = t(0);
	M.data()[4] = R.data()[3]; M.data()[5] = R.data()[4]; M.data()[6] = R.data()[5]; M.data()[7] = t(1);
	M.data()[8] = R.data()[6]; M.data()[9] = R.data()[7]; M.data()[10] = R.data()[8]; M.data()[11] = t(2);
	M.data()[12] = 0.0;         M.data()[13] = 0.0;         M.data()[14] = 0.0;         M.data()[15] = 1.0;
}
void CJoint::angleToMatrixEx(const float aAngle, CMatrix<float>& M)
{
	using miniBLAS::Vertex;
	const __m128 dir = _mm_loadu_ps(mDirection.data()), mmt = _mm_loadu_ps(mMoment.data());
	const float dirX = mDirection(0), dirY = mDirection(1), dirZ = mDirection(2);
	__m128
		omegaHat0 = _mm_set_ps(0, dirY, -dirZ, 0),
		omegaHat1 = _mm_set_ps(0, -dirX, 0, dirZ),
		omegaHat2 = _mm_set_ps(0, 0, dirX, -dirY),
		/*omegaT(i, j) = mDirection(i)*mDirection(j)*/
		omegaT0 = _mm_mul_ps(dir, _mm_permute_ps(dir, _MM_SHUFFLE(0, 0, 0, 0))),
		omegaT1 = _mm_mul_ps(dir, _mm_permute_ps(dir, _MM_SHUFFLE(1, 1, 1, 1))),
		omegaT2 = _mm_mul_ps(dir, _mm_permute_ps(dir, _MM_SHUFFLE(2, 2, 2, 2)));

	const __m128 oh00 = _mm_dp_ps(omegaHat0, omegaHat0, 0x77), oh01 = _mm_dp_ps(omegaHat0, omegaHat1, 0x77), oh02 = _mm_dp_ps(omegaHat0, omegaHat2, 0x77),
		oh11 = _mm_dp_ps(omegaHat1, omegaHat1, 0x77), oh12 = _mm_dp_ps(omegaHat1, omegaHat2, 0x77), oh22 = _mm_dp_ps(omegaHat2, omegaHat2, 0x77);
	const __m128 RB0 = _mm_blend_ps(_mm_blend_ps(oh00, oh01, 0b010), oh02, 0b100),
		RB1 = _mm_blend_ps(_mm_blend_ps(oh01, oh11, 0b010), oh12, 0b100),
		RB2 = _mm_blend_ps(_mm_blend_ps(oh02, oh12, 0b010), oh22, 0b100);

	/*R = (omegaHat*(float)sin(aAngle)) + ((omegaHat*omegaHat)*(float)(1.0 - cos(aAngle)));*/
	const __m128 sin4 = _mm_set_ps1(std::sin(aAngle)), cos4 = _mm_set_ps1(1.0f - std::cos(aAngle));
	__m128 R0 = _mm_sub_ps(_mm_mul_ps(omegaHat0, sin4), _mm_mul_ps(RB0, cos4));
	__m128 R1 = _mm_sub_ps(_mm_mul_ps(omegaHat1, sin4), _mm_mul_ps(RB1, cos4));
	__m128 R2 = _mm_sub_ps(_mm_mul_ps(omegaHat2, sin4), _mm_mul_ps(RB2, cos4));

	/*t = T*(mDirection / mMoment) + ((omegaT*mMoment)*aAngle);*/
	const __m128 dm = miniBLAS::cross_product(dir, mmt);
	__m128 t = miniBLAS::Mat3x3_Mul_Vec3(R0, R1, R2, dm),
		tB = miniBLAS::Mat3x3_Mul_Vec3(omegaT0, omegaT1, omegaT2, mmt);
	t = _mm_sub_ps(_mm_mul_ps(tB, _mm_set_ps1(aAngle)), t);
	//showfloat4("t{3}", &t);

	const static __m128 ones = _mm_set_ps1(1);
	R0 = _mm_add_ps(R0, _mm_insert_ps(ones, t, 0b00110110)/*1,0,0,tx*/);
	R1 = _mm_add_ps(R1, _mm_insert_ps(ones, t, 0b01110101)/*0,1,0,tx*/);
	R2 = _mm_add_ps(R2, _mm_insert_ps(ones, t, 0b10110011)/*0,0,1,tx*/);
	_mm_storeu_ps(M.data(), R0);
	_mm_storeu_ps(M.data() + 4, R1);
	_mm_storeu_ps(M.data() + 8, R2);
	_mm_storeu_ps(M.data() + 12, _mm_set_ps(1, 0, 0, 0));
}
miniBLAS::SQMat4x4 CJoint::angleToMatrixEx(const float aAngle)
{
	using miniBLAS::Vertex;
	using miniBLAS::SQMat3x3;
	using miniBLAS::SQMat4x4;
	const float dirX = mDirection(0), dirY = mDirection(1), dirZ = mDirection(2);
	__m128 /*omegaT(i, j) = mDirection(i)*mDirection(j)*/
		omegaT0 = _mm_mul_ps(vDir, _mm_permute_ps(vDir, _MM_SHUFFLE(0, 0, 0, 0))),
		omegaT1 = _mm_mul_ps(vDir, _mm_permute_ps(vDir, _MM_SHUFFLE(1, 1, 1, 1))),
		omegaT2 = _mm_mul_ps(vDir, _mm_permute_ps(vDir, _MM_SHUFFLE(2, 2, 2, 2)));
	const SQMat3x3 omegaHat(_mm_set_ps(0, dirY, -dirZ, 0), _mm_set_ps(0, -dirX, 0, dirZ), _mm_set_ps(0, 0, dirX, -dirY));


	const __m128 oh00 = _mm_dp_ps(omegaHat[0], omegaHat[0], 0x77), oh01 = _mm_dp_ps(omegaHat[0], omegaHat[1], 0x77),
		oh02 = _mm_dp_ps(omegaHat[0], omegaHat[2], 0x77), oh11 = _mm_dp_ps(omegaHat[1], omegaHat[1], 0x77),
		oh12 = _mm_dp_ps(omegaHat[1], omegaHat[2], 0x77), oh22 = _mm_dp_ps(omegaHat[2], omegaHat[2], 0x77);
	SQMat3x3 RB(_mm_blend_ps(_mm_blend_ps(oh00, oh01, 0b010), oh02, 0b100),
		_mm_blend_ps(_mm_blend_ps(oh01, oh11, 0b010), oh12, 0b100),
		_mm_blend_ps(_mm_blend_ps(oh02, oh12, 0b010), oh22, 0b100));

	/*R = (omegaHat*(float)sin(aAngle)) + ((omegaHat*omegaHat)*(float)(1.0 - cos(aAngle)));*/
	const SQMat3x3 R = omegaHat*std::sin(aAngle) - RB*(1.0f - std::cos(aAngle));
	
	/*t = T*(mDirection / mMoment) + ((omegaT*mMoment)*aAngle);*/
	//const __m256 t2 = miniBLAS::Mat3x3_Mul2_Vec3(omegaT0, omegaT1, omegaT2, vMom, R[0], R[1], R[2], vDM)/*om;R*/;
	//__m128 t = _mm_sub_ps(
	//	_mm_mul_ps(_mm256_castps256_ps128(t2)/*om*/, _mm_set_ps1(aAngle)),
	//	_mm256_extractf128_ps(t2, 1));
	__m128 t = miniBLAS::Mat3x3_Mul_Vec3(R[0], R[1], R[2], vDM),
		tB = miniBLAS::Mat3x3_Mul_Vec3(omegaT0, omegaT1, omegaT2, vMom);
	t = _mm_sub_ps(_mm_mul_ps(tB, _mm_set_ps1(aAngle)), t);
	/*
	{
		printf("R---compare-R\n");
		const float *oR = (float*)&R0, *nR = (float*)&R[0];
		printf("oldR  %e,%e,%e \t newR  %e,%e,%e\n", oR[0], oR[1], oR[2], nR[0], nR[1], nR[2]);
		oR = (float*)&R1, nR = (float*)&R[1];
		printf("oldR  %e,%e,%e \t newR  %e,%e,%e\n", oR[0], oR[1], oR[2], nR[0], nR[1], nR[2]);
		oR = (float*)&R2, nR = (float*)&R[2];
		printf("oldR  %e,%e,%e \t newR  %e,%e,%e\n", oR[0], oR[1], oR[2], nR[0], nR[1], nR[2]);
	}*/

	const static __m128 ones = _mm_set_ps1(1);
	SQMat4x4 M = R + SQMat4x4(true);
	M[0] = _mm_insert_ps(M[0], t, 0b00110000)/*x,y,z,tx*/;
	M[1] = _mm_insert_ps(M[1], t, 0b01110000)/*x,y,z,ty*/;
	M[2] = _mm_insert_ps(M[2], t, 0b10110000)/*x,y,z,tz*/;
	M[3] = _mm_set_ps(1, 0, 0, 0);
	return M;
}

// operator =
CJoint& CJoint::operator=(CJoint& aCopyFrom)
{
	mDirection = aCopyFrom.mDirection; vDir = aCopyFrom.vDir;
	mPoint = aCopyFrom.mPoint; vPoint = aCopyFrom.vPoint;
	mMoment = aCopyFrom.mMoment; vMom = aCopyFrom.vMom;
	mParent = aCopyFrom.mParent;
	vDM = vDM;
	return *this;
}

// C M E S H M O T I O N -------------------------------------------------------

// reset
void CMeshMotion::reset(int aJointNumber)
{
	mRBM.setSize(4, 4);
	mRBM = 0.0;
	mRBM(0, 0) = 1.0; mRBM(1, 1) = 1.0; mRBM(2, 2) = 1.0; mRBM(3, 3) = 1.0;
	mPoseParameters.setSize(6 + aJointNumber);
	mPoseParameters = 0.0;
}

// print
void CMeshMotion::print()
{
	std::cout << mRBM << std::endl;
	std::cout << mPoseParameters;
}

// writeToFile
void CMeshMotion::writeToFile()
{
	//char buffer[200];
	//sprintf(buffer,"%s%03d.txt",(NShow::mResultDir+"RBM").c_str(),NShow::mImageNo);
	//mRBM.writeToTXT(buffer);
	//sprintf(buffer,"%s%03d.txt",(NShow::mResultDir+"Params").c_str(),NShow::mImageNo);
	//mPoseParameters.writeToTXT(buffer);
}

// operator =
CMeshMotion& CMeshMotion::operator=(const CMeshMotion& aCopyFrom)
{
	mRBM = aCopyFrom.mRBM;
	mPoseParameters = aCopyFrom.mPoseParameters;
	return *this;
}

// C M E S H -------------------------------------------------------------------

// constructor
CMesh::CMesh(int aPatches, int aPoints)
{
	mPoints.resize(aPoints);
	mPatch.resize(aPatches);
	mNumPoints = aPoints;
	mNumPatch = aPatches;

	for (int i = 0; i < aPoints; i++)
	{
		mPoints[i].setSize(4);
	}
	for (int i = 0; i < aPatches; i++)
	{
		mPatch[i].setSize(3);
	}
}

// destructor
CMesh::~CMesh()
{
	//cout << "DEL-CMESH...";
	//if(mPointsS !=0) delete[] mPointsS;
	//if(mPoints !=0) delete[] mPoints;
	//if(mPatch !=0) delete[] mPatch;
	//if(mAppearanceField !=0) delete[] mAppearanceField;
	//cout << "...ok\n"; 
}

// readModel
bool CMesh::readModelHybOpt(const char* aFilename, bool smooth)
{
#if 0
#ifndef EXEC_FAST
	cout << "Read Model... ";
#endif

	std::ifstream aStream(aFilename);
	if (!aStream.is_open())
	{
		cerr << "Could not open " << aFilename << endl;
		return false;
	}

	int aXSize, aYSize, size = 4;

	CJoint Joint;

	aStream >> mNumPoints;
	aStream >> mNumPatch;
	aStream >> mJointNumber;
	if (smooth)
	{
		aStream >> mNumSmooth;
		if (mNumSmooth == 1)
			smooth = false;
		else
			size = 3 + mNumSmooth * 2;
	}
	else
		mNumSmooth = 1;

	weightMatrix.setSize(mNumPoints, mJointNumber);
	weightMatrix = 0;

#ifndef EXEC_FAST
	cout << mNumPoints << " " << mNumPatch << " " << mJointNumber << " " << mNumSmooth << endl;
#endif

	CVector<bool> BoundJoints(mJointNumber + 1, false);

	mJoint.setSize(mJointNumber + 1);

	// Read mesh components
	mPoints.resize(mNumPoints);
	mPatch.resize(mNumPatch);

	for (int i = 0; i < mNumPoints; i++)
	{
		mPoints[i].setSize(size);
	}

	for (int i = 0; i < mNumPatch; i++)
	{
		mPatch[i].setSize(3);
	}

	for (int i = 0; i < mNumPoints; i++)
	{
		aStream >> mPoints[i][0];
		aStream >> mPoints[i][1];
		aStream >> mPoints[i][2];
		if (smooth)
		{
			for (int n = 0; n < mNumSmooth * 2; n++)
				aStream >> mPoints[i][3 + n];
		}
		else
			aStream >> mPoints[i][3];
		BoundJoints((int)mPoints[i][3]) = true;
	}

	for (int i0 = 0; i0 < mNumPoints; i0++)
	{
		for (int i1 = 3; i1 < mNumSmooth * 2 + 3; i1 += 2)
		{
			int jointID = int(mPoints[i0][i1]);
			float weight = mPoints[i0][i1 + 1];
			weightMatrix(i0, jointID - 1) += weight;
		}
	}

	for (int i = 0; i < mNumPatch; i++)
	{
		aStream >> mPatch[i][0];
		aStream >> mPatch[i][1];
		aStream >> mPatch[i][2];
	}

	// set bounds
	int count = 0;
	mJointMap.setSize(mJointNumber + 1);
	mJointMap = -1;
	for (int j = 0; j <= mJointNumber; ++j)
		if (BoundJoints(j))
			mJointMap(j) = count++;

	mBounds.setSize(count, 9, 3);
	mNeighbor.setSize(count, count);
	CMatrix<float> minV(count, 3, 100000);
	CMatrix<float> maxV(count, 3, -100000);

	for (int i = 0; i < mNumPoints; i++)
	{
		int index = mJointMap((int)mPoints[i][3]);

		if (mPoints[i][0] < minV(index, 0)) minV(index, 0) = mPoints[i][0];
		if (mPoints[i][1] < minV(index, 1)) minV(index, 1) = mPoints[i][1];
		if (mPoints[i][2] < minV(index, 2)) minV(index, 2) = mPoints[i][2];
		if (mPoints[i][0] > maxV(index, 0)) maxV(index, 0) = mPoints[i][0];
		if (mPoints[i][1] > maxV(index, 1)) maxV(index, 1) = mPoints[i][1];
		if (mPoints[i][2] > maxV(index, 2)) maxV(index, 2) = mPoints[i][2];
	}

	for (int i = 0; i < count; ++i)
	{
		mBounds(i, 0, 0) = mBounds(i, 1, 0) = mBounds(i, 2, 0) = mBounds(i, 3, 0) = minV(i, 0);
		mBounds(i, 4, 0) = mBounds(i, 5, 0) = mBounds(i, 6, 0) = mBounds(i, 7, 0) = maxV(i, 0);
		mBounds(i, 0, 1) = mBounds(i, 1, 1) = mBounds(i, 4, 1) = mBounds(i, 5, 1) = minV(i, 1);
		mBounds(i, 2, 1) = mBounds(i, 3, 1) = mBounds(i, 6, 1) = mBounds(i, 7, 1) = maxV(i, 1);
		mBounds(i, 0, 2) = mBounds(i, 2, 2) = mBounds(i, 4, 2) = mBounds(i, 6, 2) = minV(i, 2);
		mBounds(i, 1, 2) = mBounds(i, 3, 2) = mBounds(i, 5, 2) = mBounds(i, 7, 2) = maxV(i, 2);
	}

	mCenter.setSize(4);
	mCenter[0] = (maxV(0, 0) + minV(0, 0)) / 2.0f;
	mCenter[1] = (maxV(0, 1) + minV(0, 1)) / 2.0f;
	mCenter[2] = (maxV(0, 2) + minV(0, 2)) / 2.0f;
	mCenter[3] = 1.0f;

	for (int j = 0; j < mJointMap.size(); ++j)
		if (mJointMap(j) >= 0)
			mBounds(mJointMap(j), 8, 0) = j;

	// Read joints
	int dummy;
	CVector<float> aDirection(3);
	CVector<float> aPoint(3);
	for (int aJointID = 1; aJointID <= mJointNumber; aJointID++)
	{
		aStream >> dummy; // ID
		aStream >> aDirection(0) >> aDirection(1) >> aDirection(2);
		aStream >> aPoint(0) >> aPoint(1) >> aPoint(2);
		mJoint(aJointID).set(aDirection, aPoint);
		aStream >> mJoint(aJointID).mParent;
	}
	// Determine which joint motion is influenced by parent joints
	mInfluencedBy.setSize(mJointNumber + 1, mJointNumber + 1);
	mInfluencedBy = false;
	for (int j = 0; j <= mJointNumber; j++)
		for (int i = 0; i <= mJointNumber; i++)
		{
			if (i == j) mInfluencedBy(i, j) = true;
			if (isParentOf(j, i)) mInfluencedBy(i, j) = true;
		}

	mNeighbor = false;
	for (int i = 0; i < mNeighbor.xSize(); i++)
	{
		mNeighbor(i, i) = true;
		int jID = (int)mBounds(i, 8, 0);
		for (int j = jID - 1; j >= 0; --j)
		{
			if (mJointMap(j) >= 0 && mInfluencedBy(jID, j))
			{
				mNeighbor(i, mJointMap(j)) = true;
				mNeighbor(mJointMap(j), i) = true;
				break;
			}
		}
	}

	mEndJoint.setSize(mJointNumber + 1);
	mEndJoint.fill(true);
	for (int i = 1; i <= mJointNumber; ++i)
		mEndJoint[mJoint(i).mParent] = false;

	for (int i = 1; i <= mJointNumber; ++i)
		if (mEndJoint[i] == true)
		{
			int j = i;
			while (mJointMap[--j] == -1)
			{
				mEndJoint[j] = true;
			}
		}

	cout << "End Joints:" << endl;
	cout << mEndJoint << endl;

	mAccumulatedMotion.reset(mJointNumber);
	mCurrentMotion.reset(mJointNumber);

#ifndef EXEC_FAST
	cout << mNeighbor << endl;
	cout << "ok" << endl;
#endif
#endif
	return true;
}

// readModel
bool CMesh::readModel(const char* aFilename, bool smooth)
{
#ifndef EXEC_FAST
	cout << "Read Model... ";
#endif

	std::ifstream aStream(aFilename);
	if (!aStream.is_open())
	{
		cerr << "Could not open " << aFilename << endl;
		return false;
	}

	int aXSize, aYSize, size = 4;

	CJoint Joint;

	aStream >> mNumPoints;
	aStream >> mNumPatch;
	aStream >> mJointNumber;
	if (smooth)
	{
		aStream >> mNumSmooth;
		if (mNumSmooth == 1)
			smooth = false;
		else
			size = 3 + mNumSmooth * 2;
	}
	else
		mNumSmooth = 1;

	weightMatrix.setSize(mNumPoints, mJointNumber);
	weightMatrix = 0;

#ifndef EXEC_FAST
	cout << mNumPoints << " " << mNumPatch << " " << mJointNumber << " " << mNumSmooth << endl;
#endif

	CVector<bool> BoundJoints(mJointNumber + 1, false);
	mBoundJoints.setSize(mJointNumber + 1);

	//mHistory.setSize(5);
	mJoint.setSize(mJointNumber + 1);
	// Read mesh components
	mPoints.resize(mNumPoints);
	mPatch.resize(mNumPatch);

	for (int i = 0; i < mNumPoints; i++)
	{
		mPoints[i].setSize(size);
	}

	for (int i = 0; i < mNumPatch; i++)
	{
		mPatch[i].setSize(3);
	}

	//   mPointsS=new CVector<float>[mNumPoints];
	//   for(int i=0;i<mNumPoints;i++)
	//     {
	//       mPointsS[i].setSize(4);
	//     }
	for (int i = 0; i < mNumPoints; i++)
	{
		aStream >> mPoints[i][0];
		aStream >> mPoints[i][1];
		aStream >> mPoints[i][2];
		//~aj: FIX FIX
		for (int m0 = 0; m0 < 3; m0++)
			mPoints[i][m0] *= SKEL_SCALE_FACTOR;

		if (smooth)
		{
			for (int n = 0; n < mNumSmooth * 2; n++)
				aStream >> mPoints[i][3 + n];
		}
		else
			aStream >> mPoints[i][3];
		BoundJoints((int)mPoints[i][3]) = true;
	}
	for (int i0 = 0; i0 < mNumPoints; i0++)
	{
		for (int i1 = 3; i1 < mNumSmooth * 2 + 3; i1 += 2)
		{
			int jointID = int(mPoints[i0][i1]);
			float weight = mPoints[i0][i1 + 1];
			weightMatrix(i0, jointID - 1) += weight;
		}
	}
	for (int i = 0; i < mNumPatch; i++)
	{
		aStream >> mPatch[i][0];
		aStream >> mPatch[i][1];
		aStream >> mPatch[i][2];
	}

	// set bounds
	int count = 0;
	mJointMap.setSize(mJointNumber + 1);
	mJointMap = -1;
	for (int j = 0; j <= mJointNumber; ++j)
		if (BoundJoints(j))
			mJointMap(j) = count++;

	mNoOfBodyParts = count;

#ifndef EXEC_FAST
	cout << "Bodyparts: " << count << endl;
	cout << "\nmJointMap: " << mJointMap;
	cout << "\nBoundJoints: " << BoundJoints;
#endif	
	mBoundJoints = BoundJoints;


	mBounds.setSize(count, 9, 3);
	mNeighbor.setSize(count, count);
	CMatrix<float> minV(count, 3, 100000);
	CMatrix<float> maxV(count, 3, -100000);

	mCenter.setSize(4);
	mCenter[0] = 0;
	mCenter[1] = 0;
	mCenter[2] = 0;
	mCenter[3] = 1.0f;

	for (int i = 0; i < mNumPoints; i++)
	{
		int index = mJointMap((int)mPoints[i][3]);

		if (mPoints[i][0] < minV(index, 0)) minV(index, 0) = mPoints[i][0];
		if (mPoints[i][1] < minV(index, 1)) minV(index, 1) = mPoints[i][1];
		if (mPoints[i][2] < minV(index, 2)) minV(index, 2) = mPoints[i][2];
		if (mPoints[i][0] > maxV(index, 0)) maxV(index, 0) = mPoints[i][0];
		if (mPoints[i][1] > maxV(index, 1)) maxV(index, 1) = mPoints[i][1];
		if (mPoints[i][2] > maxV(index, 2)) maxV(index, 2) = mPoints[i][2];

		mCenter[0] += mPoints[i][0];
		mCenter[1] += mPoints[i][1];
		mCenter[2] += mPoints[i][2];

	}

	mCenter[0] /= (float)mNumPoints;
	mCenter[1] /= (float)mNumPoints;
	mCenter[2] /= (float)mNumPoints;

#ifndef EXEC_FAST
	cout << "mCenter=" << mCenter << endl;
#endif
	for (int i = 0; i < count; ++i)
	{
		mBounds(i, 0, 0) = mBounds(i, 1, 0) = mBounds(i, 2, 0) = mBounds(i, 3, 0) = minV(i, 0);
		mBounds(i, 4, 0) = mBounds(i, 5, 0) = mBounds(i, 6, 0) = mBounds(i, 7, 0) = maxV(i, 0);
		mBounds(i, 0, 1) = mBounds(i, 1, 1) = mBounds(i, 4, 1) = mBounds(i, 5, 1) = minV(i, 1);
		mBounds(i, 2, 1) = mBounds(i, 3, 1) = mBounds(i, 6, 1) = mBounds(i, 7, 1) = maxV(i, 1);
		mBounds(i, 0, 2) = mBounds(i, 2, 2) = mBounds(i, 4, 2) = mBounds(i, 6, 2) = minV(i, 2);
		mBounds(i, 1, 2) = mBounds(i, 3, 2) = mBounds(i, 5, 2) = mBounds(i, 7, 2) = maxV(i, 2);
	}

	for (int j = 0; j < mJointMap.size(); ++j)
		if (mJointMap(j) >= 0)
			mBounds(mJointMap(j), 8, 0) = j;

	// Read joints
	int dummy;
	CVector<float> aDirection(3);
	CVector<float> aPoint(3);

	for (int aJointID = 1; aJointID <= mJointNumber; aJointID++)
	{
		aStream >> dummy; // ID
		aStream >> aDirection(0) >> aDirection(1) >> aDirection(2);
		aStream >> aPoint(0) >> aPoint(1) >> aPoint(2);

		for (int m0 = 0; m0 < 3; m0++)
		{
			aPoint(m0) *= SKEL_SCALE_FACTOR;

		}
		mJoint(aJointID).set(aDirection, aPoint);
		aStream >> mJoint(aJointID).mParent;
	}
	// Determine which joint motion is influenced by parent joints
	mInfluencedBy.setSize(mJointNumber + 1, mJointNumber + 1);
	mInfluencedBy = false;
	for (int j = 0; j <= mJointNumber; j++)
		for (int i = 0; i <= mJointNumber; i++)
		{
			if (i == j) mInfluencedBy(i, j) = true;
			if (isParentOf(j, i)) mInfluencedBy(i, j) = true;
		}

	mNeighbor = false;
	for (int i = 0; i < mNeighbor.xSize(); i++)
	{
		mNeighbor(i, i) = true;
		int jID = (int)mBounds(i, 8, 0);
		for (int j = jID - 1; j >= 0; --j)
		{
			if (mJointMap(j) >= 0 && mInfluencedBy(jID, j))
			{
				mNeighbor(i, mJointMap(j)) = true;
				mNeighbor(mJointMap(j), i) = true;
				break;
			}
		}
	}

	mEndJoint.setSize(mJointNumber + 1);
	mEndJoint.fill(true);
	for (int i = 1; i <= mJointNumber; ++i)
		mEndJoint[mJoint(i).mParent] = false;

	for (int i = 1; i <= mJointNumber; ++i)
		if (mEndJoint[i] == true)
		{
			int j = i;
			while (mJointMap[--j] == -1 && j > 0)
			{
				mEndJoint[j] = true;
			}
		}
#ifndef EXEC_FAST
	cout << "End Joints:" << endl;
	cout << mEndJoint << endl;
#endif
	mAccumulatedMotion.reset(mJointNumber);
	mCurrentMotion.reset(mJointNumber);

	mCovered.setSize(mBounds.xSize());
	mExtremity.setSize(mBounds.xSize());
	for (int i = 0; i < mExtremity.size(); ++i)
		aStream >> mExtremity[i];
	for (int i = 0; i < mCovered.size(); ++i)
		aStream >> mCovered[i];

#ifndef EXEC_FAST
	cout << mNeighbor << endl;
	cout << "ok" << endl;
#endif
	return true;
}

void CMesh::prepareData()
{
	using miniBLAS::Vertex;
	//prepare weightMatrix
	{
		wMatGap = ((mNumPoints + 7) / 4) & 0xfffffffe;
		wgtMat.resize(wMatGap*mJointNumber);
		memset(&wgtMat[0], 0x0, wMatGap*mJointNumber * sizeof(Vertex));
		const float *pWM = weightMatrix.data();
		const uint32_t ept = 4 - (mNumPoints & 0x3);
		for (uint32_t a = 0; a < mJointNumber; ++a)
			memcpy(&wgtMat[a*wMatGap], &pWM[a*mNumPoints], mNumPoints * sizeof(float));
	}
	{
		static const uint8_t idxmap[][2] = //i0,i1
		{ { 2,0 },{ 3,2 },{ 4,3 },{ 6,0 },{ 7,6 },{ 8,7 },{ 10,0 },{ 11,10 }, { 14,10 },{ 15,14 },{ 16,15 },{ 19,10 },{ 20,19 },{ 21,20 } };
		minWgtMat.resize(wMatGap * 14);
		theMinWgtMat = &minWgtMat[0];
		memset(&minWgtMat[0], 0x0, wMatGap * 14 * sizeof(Vertex));
		Vertex *__restrict vo = &minWgtMat[0];
		const __m256 helper = _mm256_set1_ps(1);
		uint32_t idx = 0;
		for (const uint8_t(&item)[2] : idxmap)
		{
			//calc min
			const Vertex *__restrict va = &wgtMat[0] + item[0] * wMatGap, *__restrict vb = &wgtMat[0] + item[1] * wMatGap;
			__m256 sumvec = _mm256_setzero_ps();
			for (uint32_t a = wMatGap / 2; a--; vo += 2, va += 2, vb += 2)
			{
				const __m256 minM = _mm256_min_ps(_mm256_load_ps((float*)va), _mm256_load_ps((float*)vb))/*01234567*/;
				sumvec = _mm256_add_ps(sumvec, minM);
				//pre-compute interlanced matrix
				__m256 mA = _mm256_permute_ps(minM, 0b10001101);
				mA = _mm256_permute2f128_ps(mA, mA, 0b00000001);
				__m256 mB = _mm256_permute_ps(minM, 0b11011000);
				_mm256_stream_ps(vo[0], _mm256_blend_ps(mA, mB, 0b11000011)/*02461357*/);
			}
			sumvec = _mm256_hadd_ps(sumvec, sumvec);
			sumvec = _mm256_hadd_ps(sumvec, sumvec);
			sumvec = _mm256_add_ps(sumvec, _mm256_permute2f128_ps(sumvec, sumvec, 0b00000001));
			minEleSum[idx++].assign(_mm256_castps256_ps128(sumvec));
		}
	}
	//prepare vPoints & smooth
	vPoints.resize(mNumPoints + 7);//at lest multiply of 8
	memset(&vPoints[mNumPoints], 0x0, 7 * sizeof(Vertex));
	ptSmooth.clear();
	smtCnt.resize(mNumPoints);
	for (uint32_t a = 0; a < mNumPoints; ++a)
	{
		uint8_t scnt = 0;
		const float *ptr = mPoints[a].data();
		vPoints[a].assign(ptr);
		vPoints[a].w = 1;
		for (uint32_t c = 0; c < mNumSmooth; c++)
		{
			const float weight = ptr[4 + 2 * c];
			if (weight > 10e-6)//take it
			{
				scnt++;
				ptSmooth.push_back({ 4 * uint32_t(ptr[3 + 2 * c]), weight });//pre compute idx
			}
		}
		if (scnt == 0)
		{
			smtCnt[a] = 1;
			ptSmooth.push_back({ 4, 0 });// this 0 weight will be ignore and considered weight 1 though
		}
		else
			smtCnt[a] = scnt;
	}
	theSmtCnt = &smtCnt[0];
	thePtSmooth = &ptSmooth[0];
	theMinEleSum = &minEleSum[0];
}

void CMesh::preCompute(const char *__restrict validMask)
{
	validPtSmooth.clear();
	validSmtCnt.clear();
	for (uint32_t row = 0, smtIdx = 0; row < mNumPoints; ++row)
	{
		const uint32_t sc = theSmtCnt[row];
		if (validMask[row] != 0)
		{
			validSmtCnt.push_back(sc);
			validPtSmooth.insert(validPtSmooth.end(), &thePtSmooth[smtIdx], &thePtSmooth[smtIdx + sc]);
		}
		smtIdx += sc;
	}
	validPts.resize(validSmtCnt.size());
}

void CMesh::centerModel()
{

#ifndef EXEC_FAST
	cout << "RESET CENTER!\n";
#endif

	CVector<float> trans(4);
	trans(0) = -mCenter(0); trans(1) = -mCenter(1); trans(2) = -mCenter(2); trans(3) = 0;

#ifndef EXEC_FAST
	cout << "trans:" << trans;
#endif

	for (int i = 0; i < mNumPoints; i++)
	{
		mPoints[i](0) += trans(0);
		mPoints[i](1) += trans(1);
		mPoints[i](2) += trans(2);
	}

	for (int i = 1; i <= mJointNumber; i++)
	{
		CVector<float> jPos(mJoint[i].getPoint());
		for (int j = 0; j < 3; ++j)
			jPos(j) += trans(j);
		mJoint[i].setPoint(jPos);
	}

	for (int i = 0; i < mBounds.xSize(); ++i)
		for (int j = 0; j < mBounds.ySize() - 1; ++j)
		{
			mBounds(i, j, 0) += trans(0);
			mBounds(i, j, 1) += trans(1);
			mBounds(i, j, 2) += trans(2);
		}
	mCenter += trans;
}

// writeModel
void CMesh::writeModel(const char* aFilename)
{
	cout << "Write Model... ";
	std::ofstream aStream(aFilename);
	aStream << "OFF" << std::endl;
	aStream << mNumPoints << " " << mNumPatch << " " << 0 << std::endl;
	// Write vertices
	for (int i = 0; i < mNumPoints; i++)
	{
		aStream << mPoints[i][0] << " ";
		aStream << mPoints[i][1] << " ";
		aStream << mPoints[i][2] << std::endl;
	}
	// Write patches
	for (int i = 0; i < mNumPatch; i++)
	{
		aStream << "3 " << mPatch[i][0] << " ";
		aStream << mPatch[i][1] << " ";
		aStream << mPatch[i][2] << std::endl;
	}
	cout << "ok" << endl;
}

void CMesh::writeAdaptModel(const char* aFilename, const CMesh* adM)
{
	cout << "Write Adapt Model... ";
	std::ofstream aStream(aFilename);
	aStream << "OFF" << std::endl;
	aStream << mNumPoints << " " << mNumPatch << " " << 0 << std::endl;
	// Write vertices
	for (int i = 0; i < mNumPoints; i++)
	{

	#if 0
		if (!IsCovered(GetBodyPart(int(mPoints[i][3]))))
		{
			aStream << mPoints[i][0] << " ";
			aStream << mPoints[i][1] << " ";
			aStream << mPoints[i][2] << std::endl;
		}
		else
		{
			float tmp = mPoints[i][4];
			int j = 5;
			for (; j < mCovered.size() * 2 + 3;)
			{
				if (IsCovered(GetBodyPart(int(mPoints[i][j]))))
				{
					tmp += mPoints[i][++j];
					++j;
				}
				else j += 2;
			}
			aStream << mPoints[i][0] * (1.0 - tmp) + tmp*adM->mPoints[i][0] << " ";
			aStream << mPoints[i][1] * (1.0 - tmp) + tmp*adM->mPoints[i][1] << " ";
			aStream << mPoints[i][2] * (1.0 - tmp) + tmp*adM->mPoints[i][2] << std::endl;
		}
	#endif    


		float tmp = 0;
		for (int j = 3; j < mCovered.size() * 2 + 3;)
		{
			if (IsCovered(GetBodyPart(int(mPoints[i][j]))))
			{
				tmp += mPoints[i][++j];
				++j;
			}
			else j += 2;
		}

		aStream << mPoints[i][0] * (1.0 - tmp) + tmp*adM->mPoints[i][0] << " ";
		aStream << mPoints[i][1] * (1.0 - tmp) + tmp*adM->mPoints[i][1] << " ";
		aStream << mPoints[i][2] * (1.0 - tmp) + tmp*adM->mPoints[i][2] << std::endl;
	}
	// Write patches
	for (int i = 0; i < mNumPatch; i++)
	{
		aStream << "3 " << mPatch[i][0] << " ";
		aStream << mPatch[i][1] << " ";
		aStream << mPatch[i][2] << std::endl;
	}
	cout << "ok" << endl;
}

// writeModel
bool CMesh::adaptOFF(const char* aFilename, float lambda)
{
	cout << "Read OFF... ";
	std::ifstream aStream(aFilename);
	if (aStream.is_open())
	{
		char buffer[200];
		aStream.getline(buffer, 200);
		cout << buffer << endl;
		aStream.getline(buffer, 200);
		cout << buffer << endl;
		// Write vertices
		for (int i = 0; i < mNumPoints; i++)
		{
			for (int j = 0; j < 3; ++j)
			{
				float tmp;
				aStream >> tmp;
				mPoints[i][j] *= (1 - lambda);
				mPoints[i][j] += lambda*tmp;
			}
		}
		cout << "ok" << endl;
		return true;
	}
	else return false;
}

// readOFF
bool CMesh::readOFF(const char* aFilename)
{
	cout << "Read OFF... ";
	std::ifstream aStream(aFilename);
	if (aStream.is_open())
	{

		char buffer[200];
		aStream.getline(buffer, 200);
		cout << buffer << endl;
		aStream >> mNumPoints;
		aStream >> mNumPatch;
		aStream >> mNumSmooth;
		mPoints.resize(mNumPoints);
		mPatch.resize(mNumPatch);

		mCenter.setSize(4);
		mCenter[0] = 0;
		mCenter[1] = 0;
		mCenter[2] = 0;
		mCenter[3] = 1.0f;

		mBounds.setSize(1, 9, 3);
		CVector<float> minV(3, 100000);
		CVector<float> maxV(3, -100000);

		// Read vertices
		for (int i = 0; i < mNumPoints; i++)
		{
			for (int j = 0; j < 3; ++j)
			{
				aStream >> mPoints[i][j];
				mCenter[j] += mPoints[i][j];
				if (mPoints[i][j] < minV(j)) minV(j) = mPoints[i][j];
				if (mPoints[i][j] > maxV(j)) maxV(j) = mPoints[i][j];
			}
		}

		mCenter[0] /= (float)mNumPoints;
		mCenter[1] /= (float)mNumPoints;
		mCenter[2] /= (float)mNumPoints;

		mBounds(0, 0, 0) = mBounds(0, 1, 0) = mBounds(0, 2, 0) = mBounds(0, 3, 0) = minV(0);
		mBounds(0, 4, 0) = mBounds(0, 5, 0) = mBounds(0, 6, 0) = mBounds(0, 7, 0) = maxV(0);
		mBounds(0, 0, 1) = mBounds(0, 1, 1) = mBounds(0, 4, 1) = mBounds(0, 5, 1) = minV(1);
		mBounds(0, 2, 1) = mBounds(0, 3, 1) = mBounds(0, 6, 1) = mBounds(0, 7, 1) = maxV(1);
		mBounds(0, 0, 2) = mBounds(0, 2, 2) = mBounds(0, 4, 2) = mBounds(0, 6, 2) = minV(2);
		mBounds(0, 1, 2) = mBounds(0, 3, 2) = mBounds(0, 5, 2) = mBounds(0, 7, 2) = maxV(2);

		// Read triangles
		for (int i = 0; i < mNumPatch; i++)
		{
			int dummy;
			aStream >> dummy;
			for (int j = 0; j < 3; ++j)
			{
				aStream >> mPatch[i][j];
			}
		}

		mJointNumber = 0;
		cout << "ok" << endl;
		return true;
	}
	else return false;
}


void CMesh::writeSkel(const char* aFilename)
{
	cout << "Write Skel... ";
	std::ofstream aStream(aFilename);
	aStream << "Skeleton" << std::endl;
	aStream << mJointNumber << std::endl;

	// Write vertices
	for (int i = 1; i <= mJointNumber; i++)
	{
		aStream << i << " ";
		aStream << mJoint[i].getDirection() << " ";
		aStream << mJoint[i].getPoint() << " ";
		aStream << mJoint[i].mParent << std::endl;
	}

	cout << "ok" << endl;
}

// rigidMotion
void CMesh::rigidMotion(CVector<CMatrix<float> >& M, CVector<float>& X, bool smooth, bool force)
{
	CVector<float> a(4); a(3) = 1.0;
	CVector<float> b(4);
	// Apply motion to points
	if (!smooth || mNumSmooth == 1)
	{
		for (int i = 0; i < mNumPoints; i++)
		{
			a(0) = mPoints[i](0); a(1) = mPoints[i](1); a(2) = mPoints[i](2);
			b = M(int(mPoints[i](3)))*a;
			mPoints[i](0) = b(0); mPoints[i](1) = b(1); mPoints[i](2) = b(2);
		}
	}
	else
	{
		for (int i = 0; i < mNumPoints; i++)
		{
			a(0) = mPoints[i](0); a(1) = mPoints[i](1); a(2) = mPoints[i](2);
			b = 0;
			for (int n = 0; n < mNumSmooth; n++)
				b += M(int(mPoints[i](3 + n * 2))) * a * mPoints[i](4 + n * 2);
			mPoints[i](0) = b(0); mPoints[i](1) = b(1); mPoints[i](2) = b(2);
		}
	}
	// Apply motion to joints
	for (int i = mJointNumber; i > 0; i--)
	{
		mJoint(i).rigidMotion(M(mJoint(i).mParent));
	}
	if (!smooth || force)
	{
		for (int j = 0; j < mBounds.xSize(); ++j)
		{
			int jID = (int)mBounds(j, 8, 0);
			for (int i = 0; i < 8; ++i)
			{
				a(0) = mBounds(j, i, 0); a(1) = mBounds(j, i, 1); a(2) = mBounds(j, i, 2);
				b = M(jID)*a;
				mBounds(j, i, 0) = b(0); mBounds(j, i, 1) = b(1); mBounds(j, i, 2) = b(2);
			}
		}
		mCenter = M(0)*mCenter;
		// Sum up motion
		mAccumulatedMotion.mRBM = M(0)*mAccumulatedMotion.mRBM;
		mCurrentMotion.mRBM = M(0)*mCurrentMotion.mRBM;
		// Correct pose parameters:
		CVector<float> T1(6);
		NRBM::RBM2Twist(T1, mCurrentMotion.mRBM);
		for (int i = 0; i < 6; i++)
			X(i) = T1(i) - mAccumulatedMotion.mPoseParameters(i);
		mAccumulatedMotion.mPoseParameters += X;
		mCurrentMotion.mPoseParameters += X;
	}
}
void CMesh::rigidMotionSim_AVX(miniBLAS::SQMat4x4(&M)[26], const bool smooth)
{
	using miniBLAS::Vertex;
	// Apply motion to points
	const SmoothParam *__restrict pSP = thePtSmooth;
	const uint32_t *__restrict pSC = theSmtCnt;
	Vertex *__restrict pPt = &vPoints[0];
	__m128 *pMat = &M[0][0];
	for (int i = mNumPoints; i--; pPt++)
	{
		const uint32_t sc = *pSC++;
		const __m128 dat = *pPt; const __m256 adat = _mm256_set_m128(dat, dat);
		switch (sc)
		{
		case 1:
		{
			const __m128 *trans = &pMat[pSP[0].idx];
			pPt->assign(_mm_blend_ps
			(
				_mm_movelh_ps(_mm_dp_ps(dat, trans[0], 0b11110001)/*x,0,0,0*/, _mm_dp_ps(dat, trans[2], 0b11110001)/*z,0,0,0*/)/*x,0,z,0*/,
				_mm_dp_ps(dat, trans[1], 0b11110010)/*0,y,0,0*/, 0b1010
			)/*x,y,z,0*/);
			break;
		}
		case 2:
		{
			const __m128 *trans0 = &pMat[pSP[0].idx]; const __m128 *trans1 = &pMat[pSP[1].idx];
			const __m256 tmp = _mm256_mul_ps
			(
				_mm256_set_m128(_mm_set1_ps(pSP[1].weight), _mm_set1_ps(pSP[0].weight))/*weight for two*/,
				_mm256_blend_ps
				(
					_mm256_blend_ps(_mm256_dp_ps(adat, _mm256_set_m128(trans1[0], trans0[0]), 0b11110001)/*x0,0,0,0;x1,0,0,0*/,
						_mm256_dp_ps(adat, _mm256_set_m128(trans1[1], trans0[1]), 0b11110010)/*0,y0,0,0;0,y1,0,0*/, 0b10101010)/*x0,y0,0,0;x1,y1,0,0*/,
					_mm256_dp_ps(adat, _mm256_set_m128(trans1[2], trans0[2]), 0b11110100)/*0,0,z0,0;0,0,z1,0*/, 0b11001100
				)/*x0,y0,z0,0;x1,y1,z1,0*/
			);
			pPt->assign(_mm_add_ps(_mm256_castps256_ps128(tmp), _mm256_extractf128_ps(tmp, 1)));
			break;
		}
		case 3:
		{
			const __m128 *trans0 = &pMat[pSP[0].idx];
			const __m128 *trans1 = &pMat[pSP[1].idx]; const __m128 *trans2 = &pMat[pSP[2].idx];
			const __m256 wgt12 = _mm256_set_m128(_mm_set1_ps(pSP[2].weight), _mm_set1_ps(pSP[1].weight));
			const __m256 tmpA = _mm256_set_m128(_mm_setzero_ps(), _mm_mul_ps
			(
				_mm_set1_ps(pSP[0].weight)/*weight for 0*/,
				_mm_blend_ps
				(
					_mm_movelh_ps(_mm_dp_ps(dat, trans0[0], 0b11110001)/*x,0,0,0*/, _mm_dp_ps(dat, trans0[2], 0b11110001)/*z,0,0,0*/)/*x,0,z,0*/,
					_mm_dp_ps(dat, trans0[1], 0b11110010)/*0,y,0,0*/, 0b1010
				)/*x,y,z,0*/
			));
			const __m256 tmpB = _mm256_mul_ps
			(wgt12, _mm256_blend_ps
				(
					_mm256_blend_ps(_mm256_dp_ps(adat, _mm256_set_m128(trans2[0], trans1[0]), 0b11110001)/*x0,0,0,0;x1,0,0,0*/,
						_mm256_dp_ps(adat, _mm256_set_m128(trans2[1], trans1[1]), 0b11110010)/*0,y0,0,0;0,y1,0,0*/, 0b10101010)/*x0,y0,0,0;x1,y1,0,0*/,
					_mm256_dp_ps(adat, _mm256_set_m128(trans2[2], trans1[2]), 0b11110100)/*0,0,z0,0;0,0,z1,0*/, 0b11001100
				)/*x0,y0,z0,0;x1,y1,z1,0*/
			);
			const __m256 tmp = _mm256_add_ps(tmpA, tmpB);
			pPt->assign(_mm_add_ps(_mm256_castps256_ps128(tmp), _mm256_extractf128_ps(tmp, 1)));
			break;
		}
		default://just consider first 4
		{
			const __m128 *trans0 = &pMat[pSP[0].idx]; const __m128 *trans1 = &pMat[pSP[1].idx];
			const __m256 wgt01 = _mm256_set_m128(_mm_set1_ps(pSP[1].weight), _mm_set1_ps(pSP[0].weight));
			const __m128 *trans2 = &pMat[pSP[2].idx]; const __m128 *trans3 = &pMat[pSP[3].idx];
			const __m256 wgt23 = _mm256_set_m128(_mm_set1_ps(pSP[3].weight), _mm_set1_ps(pSP[2].weight));
			const __m256 tmpA = _mm256_mul_ps
			(wgt01, _mm256_blend_ps
				(
					_mm256_blend_ps(_mm256_dp_ps(adat, _mm256_set_m128(trans1[0], trans0[0]), 0b11110001)/*x0,0,0,0;x1,0,0,0*/,
						_mm256_dp_ps(adat, _mm256_set_m128(trans1[1], trans0[1]), 0b11110010)/*0,y0,0,0;0,y1,0,0*/, 0b10101010)/*x0,y0,0,0;x1,y1,0,0*/,
					_mm256_dp_ps(adat, _mm256_set_m128(trans1[2], trans0[2]), 0b11110100)/*0,0,z0,0;0,0,z1,0*/, 0b11001100
				)/*x0,y0,z0,0;x1,y1,z1,0*/
			);
			const __m256 tmpB = _mm256_mul_ps
			(wgt23, _mm256_blend_ps
				(
					_mm256_blend_ps(_mm256_dp_ps(adat, _mm256_set_m128(trans3[0], trans2[0]), 0b11110001)/*x2,0,0,0;x3,0,0,0*/,
						_mm256_dp_ps(adat, _mm256_set_m128(trans3[1], trans2[1]), 0b11110010)/*0,y2,0,0;0,y3,0,0*/, 0b10101010)/*x2,y2,0,0;x3,y3,0,0*/,
					_mm256_dp_ps(adat, _mm256_set_m128(trans3[2], trans2[2]), 0b11110100)/*0,0,z2,0;0,0,z3,0*/, 0b11001100
				)/*x2,y2,z2,0;x3,y3,z3,0*/
			);
			const __m256 tmp = _mm256_add_ps(tmpA, tmpB);
			pPt->assign(_mm_add_ps(_mm256_castps256_ps128(tmp), _mm256_extractf128_ps(tmp, 1)));
			break;
		}
		}
		pSP += sc;
	}
}
void CMesh::rigidMotionSim2_AVX(miniBLAS::SQMat4x4(&M)[26], const bool smooth)
{
	using miniBLAS::Vertex;
	// Apply motion to points
	const SmoothParam *__restrict pSP = theVPtSmooth;
	const uint32_t *__restrict pSC = theVSmtCnt;
	const uint32_t cnt = validPts.size();
	__m128 *pMat = &M[0][0];
	for (uint32_t i = 0; i < cnt; i++)
	{
		const uint32_t sc = pSC[i];
		const __m128 dat = validPts[i]; const __m256 adat = _mm256_set_m128(dat, dat);
		switch (sc)
		{
		case 1:
		{
			const __m128 *trans = &pMat[pSP[0].idx];
			validPts[i].assign(_mm_blend_ps
			(
				_mm_movelh_ps(_mm_dp_ps(dat, trans[0], 0b11110001)/*x,0,0,0*/, _mm_dp_ps(dat, trans[2], 0b11110001)/*z,0,0,0*/)/*x,0,z,0*/,
				_mm_dp_ps(dat, trans[1], 0b11110010)/*0,y,0,0*/, 0b1010
			)/*x,y,z,0*/);
			break;
		}
		case 2:
		{
			const __m128 *trans0 = &pMat[pSP[0].idx]; const __m128 *trans1 = &pMat[pSP[1].idx];
			const __m256 tmp = _mm256_mul_ps
			(
				_mm256_set_m128(_mm_set1_ps(pSP[1].weight), _mm_set1_ps(pSP[0].weight))/*weight for two*/,
				_mm256_blend_ps
				(
					_mm256_blend_ps(_mm256_dp_ps(adat, _mm256_set_m128(trans1[0], trans0[0]), 0b11110001)/*x0,0,0,0;x1,0,0,0*/,
						_mm256_dp_ps(adat, _mm256_set_m128(trans1[1], trans0[1]), 0b11110010)/*0,y0,0,0;0,y1,0,0*/, 0b10101010)/*x0,y0,0,0;x1,y1,0,0*/,
					_mm256_dp_ps(adat, _mm256_set_m128(trans1[2], trans0[2]), 0b11110100)/*0,0,z0,0;0,0,z1,0*/, 0b11001100
				)/*x0,y0,z0,0;x1,y1,z1,0*/
			);
			validPts[i].assign(_mm_add_ps(_mm256_castps256_ps128(tmp), _mm256_extractf128_ps(tmp, 1)));
			break;
		}
		case 3:
		{
			const __m128 *trans0 = &pMat[pSP[0].idx];
			const __m128 *trans1 = &pMat[pSP[1].idx]; const __m128 *trans2 = &pMat[pSP[2].idx];
			const __m256 wgt12 = _mm256_set_m128(_mm_set1_ps(pSP[2].weight), _mm_set1_ps(pSP[1].weight));
			const __m256 tmpA = _mm256_set_m128(_mm_setzero_ps(), _mm_mul_ps
			(
				_mm_set1_ps(pSP[0].weight)/*weight for 0*/,
				_mm_blend_ps
				(
					_mm_movelh_ps(_mm_dp_ps(dat, trans0[0], 0b11110001)/*x,0,0,0*/, _mm_dp_ps(dat, trans0[2], 0b11110001)/*z,0,0,0*/)/*x,0,z,0*/,
					_mm_dp_ps(dat, trans0[1], 0b11110010)/*0,y,0,0*/, 0b1010
				)/*x,y,z,0*/
			));
			const __m256 tmpB = _mm256_mul_ps
			(wgt12, _mm256_blend_ps
			(
				_mm256_blend_ps(_mm256_dp_ps(adat, _mm256_set_m128(trans2[0], trans1[0]), 0b11110001)/*x0,0,0,0;x1,0,0,0*/,
					_mm256_dp_ps(adat, _mm256_set_m128(trans2[1], trans1[1]), 0b11110010)/*0,y0,0,0;0,y1,0,0*/, 0b10101010)/*x0,y0,0,0;x1,y1,0,0*/,
				_mm256_dp_ps(adat, _mm256_set_m128(trans2[2], trans1[2]), 0b11110100)/*0,0,z0,0;0,0,z1,0*/, 0b11001100
			)/*x0,y0,z0,0;x1,y1,z1,0*/
			);
			const __m256 tmp = _mm256_add_ps(tmpA, tmpB);
			validPts[i].assign(_mm_add_ps(_mm256_castps256_ps128(tmp), _mm256_extractf128_ps(tmp, 1)));
			break;
		}
		default://just consider first 4
		{
			const __m128 *trans0 = &pMat[pSP[0].idx]; const __m128 *trans1 = &pMat[pSP[1].idx];
			const __m256 wgt01 = _mm256_set_m128(_mm_set1_ps(pSP[1].weight), _mm_set1_ps(pSP[0].weight));
			const __m128 *trans2 = &pMat[pSP[2].idx]; const __m128 *trans3 = &pMat[pSP[3].idx];
			const __m256 wgt23 = _mm256_set_m128(_mm_set1_ps(pSP[3].weight), _mm_set1_ps(pSP[2].weight));
			const __m256 tmpA = _mm256_mul_ps
			(wgt01, _mm256_blend_ps
			(
				_mm256_blend_ps(_mm256_dp_ps(adat, _mm256_set_m128(trans1[0], trans0[0]), 0b11110001)/*x0,0,0,0;x1,0,0,0*/,
					_mm256_dp_ps(adat, _mm256_set_m128(trans1[1], trans0[1]), 0b11110010)/*0,y0,0,0;0,y1,0,0*/, 0b10101010)/*x0,y0,0,0;x1,y1,0,0*/,
				_mm256_dp_ps(adat, _mm256_set_m128(trans1[2], trans0[2]), 0b11110100)/*0,0,z0,0;0,0,z1,0*/, 0b11001100
			)/*x0,y0,z0,0;x1,y1,z1,0*/
			);
			const __m256 tmpB = _mm256_mul_ps
			(wgt23, _mm256_blend_ps
			(
				_mm256_blend_ps(_mm256_dp_ps(adat, _mm256_set_m128(trans3[0], trans2[0]), 0b11110001)/*x2,0,0,0;x3,0,0,0*/,
					_mm256_dp_ps(adat, _mm256_set_m128(trans3[1], trans2[1]), 0b11110010)/*0,y2,0,0;0,y3,0,0*/, 0b10101010)/*x2,y2,0,0;x3,y3,0,0*/,
				_mm256_dp_ps(adat, _mm256_set_m128(trans3[2], trans2[2]), 0b11110100)/*0,0,z2,0;0,0,z3,0*/, 0b11001100
			)/*x2,y2,z2,0;x3,y3,z3,0*/
			);
			const __m256 tmp = _mm256_add_ps(tmpA, tmpB);
			validPts[i].assign(_mm_add_ps(_mm256_castps256_ps128(tmp), _mm256_extractf128_ps(tmp, 1)));
			break;
		}
		}
		pSP += sc;
	}
}

void CMesh::smoothMotionDQ(CVector<CMatrix<float> >& M, CVector<float>& X)
{
	std::vector<CVector<float> > vq0(M.size());
	std::vector<CVector<float> > vdq(M.size());

	//transform matrix to quaternion
	for (int i = 0; i < M.size(); ++i)
	{
		vq0[i].setSize(4);
		vdq[i].setSize(4);
		NRBM::RBM2QDQ(M[i], vq0[i], vdq[i]);
	}

	CVector<float> a(4); a(3) = 1.0;
	CVector<float> b(4);
	CVector<float> eq0(4);
	CVector<float> edq(4);
	CMatrix<float> eM(4, 4, 0);

	// Apply motion to points
	for (int i = 0; i < mNumPoints; i++)
	{
		a(0) = mPoints[i](0); a(1) = mPoints[i](1); a(2) = mPoints[i](2);
		eq0 = 0;
		edq = 0;

		CVector<float> q = vq0[int(mPoints[i](3))];
		eq0 = vq0[int(mPoints[i](3))] * mPoints[i](4);
		edq = vdq[int(mPoints[i](3))] * mPoints[i](4);
		for (int n = 1; n < mNumSmooth; n++)
		{
			if (q*vq0[int(mPoints[i](3 + n * 2))] < 0)
			{
				eq0 -= vq0[int(mPoints[i](3 + n * 2))] * mPoints[i](4 + n * 2);
				edq -= vdq[int(mPoints[i](3 + n * 2))] * mPoints[i](4 + n * 2);
			}
			else
			{
				eq0 += vq0[int(mPoints[i](3 + n * 2))] * mPoints[i](4 + n * 2);
				edq += vdq[int(mPoints[i](3 + n * 2))] * mPoints[i](4 + n * 2);
			}
		}

		float invl = 1.0 / eq0.norm();
		eq0 *= invl;
		edq *= invl;

		NRBM::QDQ2RBM(eq0, edq, eM);

		b = eM*a;

		mPoints[i](0) = b(0); mPoints[i](1) = b(1); mPoints[i](2) = b(2);
	}

	// Apply motion to joints
	for (int i = mJointNumber; i > 0; i--)
	{
		mJoint(i).rigidMotion(M(mJoint(i).mParent));
	}
}


void CMesh::makeSmooth(CMesh* initMesh, bool dual)
{
	if (mNumSmooth > 1)
	{

		mPoints = initMesh->mPoints;
		mJoint = initMesh->mJoint;

		for (int i = 6; i < mCurrentMotion.mPoseParameters.size(); ++i)
		{
			while (mCurrentMotion.mPoseParameters[i] < -3.1415926536)
				mCurrentMotion.mPoseParameters[i] += 2.0*3.1415926536;
			while (mCurrentMotion.mPoseParameters[i] > 3.1415926536)
				mCurrentMotion.mPoseParameters[i] -= 2.0*3.1415926536;
		}

		CVector<float> X(mCurrentMotion.mPoseParameters);
		CVector<CMatrix<float> >M(joints() + 1);

		M(0) = mCurrentMotion.mRBM;

		for (int i = 1; i <= joints(); i++)
		{
			M(i).setSize(4, 4); M(i) = 0;
			M(i)(0, 0) = 1.0; M(i)(1, 1) = 1.0; M(i)(2, 2) = 1.0; M(i)(3, 3) = 1.0;
		}

		for (int i = joints(); i > 0; i--)
		{
			CMatrix<float> Mi(4, 4);
			joint(i).angleToMatrix(X(5 + i), Mi);
			for (int j = 1; j <= joints(); j++)
			{
				if (influencedBy(j, i)) M(j) = Mi*M(j);
			}
		}

		for (int i = 1; i <= joints(); i++)
		{
			M(i) = M(0)*M(i);
		}

		if (dual == true)
			smoothMotionDQ(M, X);
		else
			rigidMotion(M, X, true);
	}
}

static void printM4x4(const char * name, const float *ptr)
{
	printf("%s : \n", name);
	for (uint32_t a = 0; a < 16;a += 4)
		printf("\t%f,%f,%f,%f\n", ptr[a], ptr[a + 1], ptr[a + 2], ptr[a + 3]);
}

// angleToMatrix
void CMesh::angleToMatrix(const CMatrix<float>& aRBM, CVector<float>& aJAngles, CVector<CMatrix<float> >& M)
{
	// Determine motion of parts behind joints
	for (int i = 1; i <= mJointNumber; i++)
	{
		M(i).setSize(4, 4); M(i) = 0;
		M(i)(0, 0) = 1.0; M(i)(1, 1) = 1.0; M(i)(2, 2) = 1.0; M(i)(3, 3) = 1.0;
	}
	unsigned int jIds[] = { 1,24,2,3,4,5,23,6,7,8,9,10,25,11,12,13,14,15,16,17,18,19,20,21,22 };

	// Leonid
	for (int ii = mJointNumber; ii > 0; ii--)
	{
		int i = jIds[ii - 1];
		CMatrix<float> Mi(4, 4);
		mJoint(i).angleToMatrixEx(aJAngles(i + 5), Mi); // i-1
		for (int jj = 1; jj <= mJointNumber; jj++)
		{
			int j = jIds[jj - 1];
			if (mInfluencedBy(j, i))
				M(j) = Mi*M(j);
		}
	}

	for (int i = 1; i <= mJointNumber; i++)
		M(i) = aRBM*M(i);
	M(0) = aRBM;
}
void CMesh::angleToMatrixEx(const CMatrix<float>& aRBM, const CVector<float>& aJAngles, miniBLAS::SQMat4x4(&M)[26])
{
	using miniBLAS::Vertex;
	using miniBLAS::SQMat4x4;

	for (auto& ele : M)
		ele = SQMat4x4(true);
	// Determine motion of parts behind joints
	uint32_t ids[] = { 1,24,2,3,4,5,23,6,7,8,9,10,25,11,12,13,14,15,16,17,18,19,20,21,22 };

	// Leonid
	for (int ii = mJointNumber; ii--;)
	{
		uint32_t id = ids[ii];
		auto Mi = mJoint(id).angleToMatrixEx(aJAngles(id + 5));
		/*
		CMatrix<float> oMi(4, 4);
		mJoint(id).angleToMatrixEx(aJAngles(id + 5), oMi); // i-1
		printf("joint matrix %d\n", id);
		for (uint32_t b = 0; b < 4; ++b)
		{
			Vertex nM(Mi[b]), oM;
			oM.assign(oMi.data() + b * 4);
			printf("new: %e,%e,%e,%e \t\told: %e,%e,%e,%e\n", nM.x, nM.y, nM.z, nM.w, oM.x, oM.y, oM.z, oM.w);
		}
		*/

		for (int jj = 0; jj < mJointNumber; jj++)
		{
			uint32_t jid = ids[jj];
			if (mInfluencedBy(jid, id))
				M[jid] = Mi * M[jid];
		}
	}

	M[0].assign(aRBM.data());
	for (int i = 1; i <= mJointNumber; i++)
		M[i] = M[0] * M[i];
}

void CMesh::invAngleToMatrix(const CMatrix<float>& aRBM, CVector<float>& aJAngles, CVector<CMatrix<float> >& M)
{
	// Determine motion of parts behind joints
	for (int i = 1; i <= mJointNumber; i++)
	{
		M(i).setSize(4, 4); M(i) = 0;
		M(i)(0, 0) = 1.0; M(i)(1, 1) = 1.0; M(i)(2, 2) = 1.0; M(i)(3, 3) = 1.0;
	}
	for (int i = mJointNumber; i > 0; i--)
	{
		CMatrix<float> Mi(4, 4);
		mJoint(i).angleToMatrix(aJAngles(i + 5), Mi); // i-1
		for (int j = 1; j <= mJointNumber; j++)
			if (mInfluencedBy(j, i)) M(j) = M(j)*Mi;
	}
	for (int i = 1; i <= mJointNumber; i++)
		M(i) = M(i)*aRBM;
	M(0) = aRBM;
}

void CMesh::twistToMatrix(CVector<float>& aTwist, CVector<CMatrix<float> >& M)
{
	NRBM::Twist2RBM(aTwist, M(0));
	angleToMatrix(M(0), aTwist, M);
}

// isParentOf
bool CMesh::isParentOf(int aParentJointID, int aJointID)
{
	if (aJointID == 0) return false;
	if (mJoint(aJointID).mParent == aParentJointID) return true;
	return isParentOf(aParentJointID, mJoint(aJointID).mParent);
}

// operator=
void CMesh::operator=(const CMesh& aMesh)
{
	mJointNumber = aMesh.mJointNumber;
	mNumPoints = aMesh.mNumPoints;
	mNumPatch = aMesh.mNumPatch;
	mNumSmooth = aMesh.mNumSmooth;

	mPoints = aMesh.mPoints;
	mPatch = aMesh.mPatch;

	mNoOfBodyParts = aMesh.mNoOfBodyParts;
	mBoundJoints = aMesh.mBoundJoints;

	mBounds = aMesh.mBounds;
	mCenter = aMesh.mCenter;
	mJointMap = aMesh.mJointMap;
	mNeighbor = aMesh.mNeighbor;
	mEndJoint = aMesh.mEndJoint;

	mCovered = aMesh.mCovered;
	mExtremity = aMesh.mExtremity;

	mJoint = aMesh.mJoint;
	mInfluencedBy = aMesh.mInfluencedBy;

	mAccumulatedMotion = aMesh.mAccumulatedMotion;
	mCurrentMotion = aMesh.mCurrentMotion;

	weightMatrix = aMesh.weightMatrix;

	wgtMat = aMesh.wgtMat;
	minWgtMat = aMesh.minWgtMat;
	theMinEleSum = aMesh.theMinEleSum;
	wMatGap = aMesh.wMatGap;
	theMinWgtMat = &minWgtMat[0];
	vPoints = aMesh.vPoints;
	ptSmooth = aMesh.ptSmooth;
	thePtSmooth = &ptSmooth[0];
}

CMesh::CMesh(const CMesh& from, const miniBLAS::VertexVec *pointsIn, const miniBLAS::VertexVec *vpIn)
{
	using miniBLAS::Vertex;
	isCopy = true;
	mJointNumber = from.mJointNumber;
	mNumPoints = from.mNumPoints;
	mNumPatch = 0;// from.mNumPatch;
	mNumSmooth = from.mNumSmooth;

	mNoOfBodyParts = from.mNoOfBodyParts;
	mBoundJoints = from.mBoundJoints;

	mBounds = from.mBounds;
	mCenter = from.mCenter;
	mJointMap = from.mJointMap;
	mNeighbor = from.mNeighbor;
	mEndJoint = from.mEndJoint;

	mCovered = from.mCovered;
	mExtremity = from.mExtremity;

	mJoint = from.mJoint;
	mInfluencedBy = from.mInfluencedBy;

	mAccumulatedMotion = from.mAccumulatedMotion;
	mCurrentMotion = from.mCurrentMotion;

	//avoid overhead of copying data unecessarily
	wMatGap = from.wMatGap;
	theMinEleSum = from.theMinEleSum;
	theMinWgtMat = from.theMinWgtMat;
	thePtSmooth = from.thePtSmooth;
	theSmtCnt = from.theSmtCnt;
	theVPtSmooth = &from.validPtSmooth[0];
	theVSmtCnt = &from.validSmtCnt[0];
	if (pointsIn == nullptr)
		vPoints = from.vPoints;
	else
		vPoints = *pointsIn;
	if (vpIn == nullptr)
		validPts.resize(from.validPts.size());
	else
	{
		validPts = *vpIn;
	}
}

void CMesh::projectToImage(CMatrix<float>& aImage, CMatrix<float>& P, int aLineValue)
{
	int I[4];
	CVector<float> aPoint(3);
	int ax, ay, bx, by, cx, cy;
	int Size = (*this).GetMeshSize();
	int j;

	int det;

	float X, Y, Z;

	//cout << Size << endl;

	for (int i = 0; i < Size; i++)
	{
		(*this).GetPatch(i, I[0], I[1], I[2]);
		j = 0;
		// Test if patch is visible ...
		X = mPoints[I[0]](0); Y = mPoints[I[0]](1); Z = mPoints[I[0]](2);
		(*this).projectPoint(P, X, Y, Z, ax, ay);
		X = mPoints[I[1]](0); Y = mPoints[I[1]](1); Z = mPoints[I[1]](2);
		(*this).projectPoint(P, X, Y, Z, bx, by);
		X = mPoints[I[2]](0); Y = mPoints[I[2]](1); Z = mPoints[I[2]](2);
		(*this).projectPoint(P, X, Y, Z, cx, cy);

		det = ax*(by - cy) - bx*(ay - cy) + cx*(ay - by);

		if (det < 0.001)
		{
			for (j = 0; j < 3; j++)
			{
				(*this).projectPoint(P, mPoints[I[j]](0), mPoints[I[j]](1), mPoints[I[j]](2), bx, by);

				if (ax >= 0 && ay >= 0 && ax < aImage.xSize() && ay < aImage.ySize())
					aImage.drawLine(ax, ay, bx, by, aLineValue);
				ax = bx;
				ay = by;
			}
			j = 0;
			(*this).projectPoint(P, mPoints[I[j]](0), mPoints[I[j]](1), mPoints[I[j]](2), bx, by);
			if (ax >= 0 && ay >= 0 && ax < aImage.xSize() && ay < aImage.ySize())
				aImage.drawLine(ax, ay, bx, by, aLineValue);
		}
	}
}

void CMesh::projectToImage(CMatrix<float>& aImage, CMatrix<float>& P, int aLineVal, int aNodeVal)
{
	projectToImage(aImage, P, aLineVal);
	projectPointsToImage(aImage, P, aNodeVal);
}

void CMesh::projectToImage(CTensor<float>& aImage, CMatrix<float>& P, int aLineR, int aLineG, int aLineB)
{
	CMatrix<float> aMesh(aImage.xSize(), aImage.ySize(), 0);
	projectToImage(aMesh, P, 255);
	int aSize = aMesh.size();
	int a2Size = 2 * aSize;
	for (int i = 0; i < aSize; i++)
		if (aMesh.data()[i] == 255)
		{
			aImage.data()[i] = aLineR;
			aImage.data()[i + aSize] = aLineG;
			aImage.data()[i + a2Size] = aLineB;
		}
#if 0
	int mx, my;
	projectPoint(P, mCenter[0], mCenter[1], mCenter[2], mx, my);
	int crossSize = 10;
	int x1 = NMath::max(mx - crossSize, 0); int x2 = NMath::min(mx + crossSize, aImage.xSize() - 1);
	int y1 = NMath::max(my - crossSize, 0); int y2 = NMath::min(my + crossSize, aImage.ySize() - 1);
	for (int x = x1; x <= x2; ++x)
	{
		aImage(x, my, 0) = 40;
		aImage(x, my, 1) = 240;
		aImage(x, my, 2) = 40;
	}
	for (int y = y1; y <= y2; ++y)
	{
		aImage(mx, y, 0) = 40;
		aImage(mx, y, 1) = 240;
		aImage(mx, y, 2) = 40;
	}
#endif
}

void CMesh::projectToImage(CTensor<float>& aImage, CMatrix<float>& P, int aLineR, int aLineG, int aLineB, int aNodeR, int aNodeG, int aNodeB)
{
	CMatrix<float> aMesh(aImage.xSize(), aImage.ySize(), 0);
	projectToImage(aMesh, P, 255, 128);
	int aSize = aMesh.size();
	int a2Size = 2 * aSize;
	for (int i = 0; i < aSize; i++)
		if (aMesh.data()[i] == 255)
		{
			aImage.data()[i] = aLineR;
			aImage.data()[i + aSize] = aLineG;
			aImage.data()[i + a2Size] = aLineB;
		}
		else if (aMesh.data()[i] == 128)
		{
			aImage.data()[i] = aNodeR;
			aImage.data()[i + aSize] = aNodeG;
			aImage.data()[i + a2Size] = aNodeB;
		}
}

// projectPointsToImage
void CMesh::projectPointsToImage(CMatrix<float>& aImage, CMatrix<float>& P, int aNodeVal)
{
	int ax, ay;
	CMatrix<float> aMesh(aImage.xSize(), aImage.ySize(), 0);
	projectToImage(aMesh, P, aNodeVal);

	int Size = (*this).GetPointSize();
	for (int i = 0; i < Size; i++)
	{
		projectPoint(P, mPoints[i](0), mPoints[i](1), mPoints[i](2), ax, ay);
		if (ax >= 0 && ay >= 0 && ax < aImage.xSize() && ay < aImage.ySize())
			if (aMesh(ax, ay) == aNodeVal)
				aImage(ax, ay) = aNodeVal;
	}
}


void CMesh::projectPointsToImage(CTensor<float>& a3DCoords, CMatrix<float>& P)
{
	int ax, ay;
	int aNodeVal = 39;

	CMatrix<float> aMesh(a3DCoords.xSize(), a3DCoords.ySize(), 0);
	projectToImage(aMesh, P, aNodeVal);

	int Size = GetPointSize();
	for (int i = 0; i < Size; i++)
	{
		projectPoint(P, mPoints[i](0), mPoints[i](1), mPoints[i](2), ax, ay);
		if (ax >= 0 && ay >= 0 && ax < a3DCoords.xSize() && ay < a3DCoords.ySize())
			if (aMesh(ax, ay) == aNodeVal)
			{
				a3DCoords(ax, ay, 0) = mPoints[i](0);
				a3DCoords(ax, ay, 1) = mPoints[i](1);
				a3DCoords(ax, ay, 2) = mPoints[i](2);
				a3DCoords(ax, ay, 3) = mPoints[i](3);
			}
	}

}

void CMesh::projectToImageJ(CMatrix<float>& aImage, CMatrix<float>& P, int aLineValue, int aJoint, int xoffset, int yoffset)
{

	int I[4];
	CVector<float> aPoint(3);
	int ax, ay, bx, by, cx, cy;
	int Size = (*this).GetMeshSize();
	int j;

	int det;

	float X, Y, Z;

	for (int i = 0; i < Size; i++)
	{
		(*this).GetPatch(i, I[0], I[1], I[2]);
		j = 0;
		// Test if patch is visible ...
		X = mPoints[I[0]](0); Y = mPoints[I[0]](1); Z = mPoints[I[0]](2);
		(*this).projectPoint(P, X, Y, Z, ax, ay);
		ax -= xoffset; ay -= yoffset;
		X = mPoints[I[1]](0); Y = mPoints[I[1]](1); Z = mPoints[I[1]](2);
		(*this).projectPoint(P, X, Y, Z, bx, by);
		bx -= xoffset; by -= yoffset;
		X = mPoints[I[2]](0); Y = mPoints[I[2]](1); Z = mPoints[I[2]](2);
		(*this).projectPoint(P, X, Y, Z, cx, cy);
		cx -= xoffset; cy -= yoffset;

		det = ax*(by - cy) - bx*(ay - cy) + cx*(ay - by);

		{
			if (((*this).GetJointID(I[0]) == aJoint) || ((*this).GetJointID(I[1]) == aJoint) || ((*this).GetJointID(I[2]) == aJoint))
			{
				for (j = 0; j < 3; j++)
				{
					(*this).projectPoint(P, mPoints[I[j]](0), mPoints[I[j]](1), mPoints[I[j]](2), bx, by);
					bx -= xoffset; by -= yoffset;

					if (ax >= 0 && ay >= 0 && ax < aImage.xSize() && ay < aImage.ySize())
						aImage.drawLine(ax, ay, bx, by, aLineValue);
					ax = bx;
					ay = by;
				}
				j = 0;
				(*this).projectPoint(P, mPoints[I[j]](0), mPoints[I[j]](1), mPoints[I[j]](2), bx, by);
				bx -= xoffset; by -= yoffset;
				if (ax >= 0 && ay >= 0 && ax < aImage.xSize() && ay < aImage.ySize())
					aImage.drawLine(ax, ay, bx, by, aLineValue);
			}
		}
	}
}

// projectSurface
void CMesh::projectSurface(CMatrix<float>& aImage, CMatrix<float>& P, int xoffset, int yoffset)
{
	//aImage = 0;

	CMatrix<float> ImageP = aImage;

	for (int k = 0; k <= (*this).joints(); k++)
	{
		ImageP = 0;
		projectToImageJ(ImageP, P, 1, k, xoffset, yoffset);

		for (int i = 0; i < ImageP.xSize(); i++)
			ImageP(i, 0) = 0;
		for (int i = 0; i < ImageP.ySize(); i++)
			ImageP(0, i) = 0;
		int aSize = ImageP.size();
		for (int i = 0; i < aSize; i++)
			if (ImageP.data()[i] == 0)
			{
				int y = i / ImageP.xSize();
				int x = i - ImageP.xSize()*y;
				ImageP.connectedComponent(x, y);
				break;
			}
		// Fill in ...
		for (int i = 0; i < aSize; i++)
		{
			if (ImageP.data()[i] == 0) aImage.data()[i] = 255;
		}
	}
}

std::vector<CMatrix<double> > CMesh::readShapeSpaceEigens(const double* eigenVectorsIn, int numEigenVectors, int nPoints)
{
	std::vector<CMatrix<double> > eigenVectors(numEigenVectors);
	int dimD = 3;

	for (unsigned int i0 = 0; i0 < numEigenVectors; i0++)
	{
		eigenVectors[i0].setSize(nPoints, dimD);
	}
	int n = 0;
	for (int col = 0; col < dimD; col++)
		for (int row = 0; row < nPoints; row++)
			for (unsigned int i0 = 0; i0 < numEigenVectors; i0++)
			{
				eigenVectors[i0](row, col) = eigenVectorsIn[n];
				n++;
			}
	return eigenVectors;
}

std::vector<CMatrix<double> > CMesh::readShapeSpaceEigens(std::string fileName, int numEigenVectors)
{
	string path = fileName;
	ifstream loadFile;
	loadFile.open(path.c_str());
	//reading only first four columns here
	char fullLine[10000];
	char(*c)[500] = new char[numEigenVectors][500];
	std::vector<CMatrix<double> > eigenVectors(numEigenVectors);
	for (unsigned int i0 = 0; i0 < numEigenVectors; i0++)
	{
		eigenVectors[i0].setSize(EVALUATE_POINTS_NUM, 3);
		eigenVectors[i0] = 0;
	}
	unsigned int row = 0, col = 0;
	unsigned totalLines = 1;
	int noPts = 0;
	while (!loadFile.getline(fullLine, 10000).eof())
	{
		if (fullLine[0] == '#')
		{	//first start line with comments
			int totalRws = 0, dimD = 0;
			sscanf(fullLine, "# %d ( %d x %d )", &totalRws, &noPts, &dimD);
			continue;
		}
		sscanf(fullLine, "%s%s%s%s%s%*s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s",
			&c[0], &c[1], &c[2], &c[3],
			&c[4], &c[5], &c[6], &c[7], &c[8], &c[9],
			&c[10], &c[11], &c[12], &c[13],
			&c[14], &c[15], &c[16], &c[17], &c[18], &c[19]);
		for (unsigned int i0 = 0; i0 < numEigenVectors; i0++)
		{
			eigenVectors[i0](row, col) = atof(c[i0]);
		}
		row++;
		if (row >= noPts)
		{
			row = 0; col++;
			if (col == 3) col = 0;
		}
		totalLines++;
	}
	loadFile.close();
	delete[] c;
	return eigenVectors;
}

int CMesh::shapeChangesToMesh(CVector<float> shapeParams, const std::vector<CMatrix<double> >& eigenVectors)
{
	unsigned int noPts = mPoints.size();
	for (unsigned int i1 = 0; i1 < mPoints.size(); i1++)
	{
		for (unsigned int i2 = 0; i2 < 3; i2++)
		{
			for (unsigned int i0 = 0; i0 < shapeParams.size(); i0++)
			{
				double tmp = shapeParams[i0];
				(mPoints[i1])[i2] += shapeParams[i0] * SKEL_SCALE_FACTOR*eigenVectors[i0](i1, i2);
			}
		}
	}
	//cout<<"Done!";
	return true;
}

void CMesh::fastShapeChangesToMesh(const double *shapeParamsIn, const uint32_t numEigenVectors, const double *eigenVectorsIn)
{
	const auto nPoints = mPoints.size();
	for (uint32_t col = 0; col < 3; col++)
		for (uint32_t row = 0; row < nPoints; row++)
		{
			float& obj = (mPoints[row])[col];
			double adder = 0;
			for (uint32_t i = 0; i < numEigenVectors; i++)
			{
				adder += shapeParamsIn[i] * SKEL_SCALE_FACTOR * (*eigenVectorsIn++);
			}
			obj += adder;
		}
}

void CMesh::fastShapeChangesToMesh(const miniBLAS::Vertex *shapeParamsIn, const miniBLAS::Vertex *eigenVectorsIn)
{
	using miniBLAS::Vertex;
	const uint32_t numEigenVectors = 20;
	//row * col(3) * z(20 = 5Vertex)
	//calculate vertex-dpps-vertex =====> 20 mul -> sum, sum added to mPoints[r,c]
	const __m128 sp1 = shapeParamsIn[0], sp2 = shapeParamsIn[1], sp3 = shapeParamsIn[2], sp4 = shapeParamsIn[3], sp5 = shapeParamsIn[4];
	const Vertex *__restrict pEvec = eigenVectorsIn;
	for (uint32_t row = 0; row < mNumPoints; row++, pEvec += 16)
	{
		_mm_prefetch((const char*)(pEvec + 16), _MM_HINT_NTA);
		_mm_prefetch((const char*)(pEvec + 20), _MM_HINT_NTA);
		_mm_prefetch((const char*)(pEvec + 24), _MM_HINT_NTA);
		_mm_prefetch((const char*)(pEvec + 28), _MM_HINT_NTA);
		const __m128 add01 = _mm_add_ps
		(
			_mm_load_ps(vPoints[row])/*x,y,z,?*/,
			_mm_blend_ps(_mm_dp_ps(sp1, pEvec[0], 0b11110001)/*sx1,0,0,0*/, _mm_dp_ps(sp1, pEvec[5], 0b11110010)/*0,sy1,0,0*/, 0b10)/*sx1,sy1,0,0*/
		);
		const __m128 add23 = _mm_add_ps
		(
			_mm_blend_ps(_mm_dp_ps(sp2, pEvec[1], 0b11110001)/*sx2,0,0,0*/, _mm_dp_ps(sp2, pEvec[6], 0b11110010)/*0,sy2,0,0*/, 0b10)/*sx2,sy2,0,0*/,
			_mm_blend_ps(_mm_dp_ps(sp3, pEvec[2], 0b11110001)/*sx3,0,0,0*/, _mm_dp_ps(sp3, pEvec[7], 0b11110010)/*0,sy3,0,0*/, 0b10)/*sx3,sy3,0,0*/
		);
		const __m128 add45 = _mm_add_ps
		(
			_mm_blend_ps(_mm_dp_ps(sp4, pEvec[3], 0b11110001)/*sx4,0,0,0*/, _mm_dp_ps(sp4, pEvec[8], 0b11110010)/*0,sy4,0,0*/, 0b10)/*sx4,sy4,0,0*/,
			_mm_blend_ps(_mm_dp_ps(sp5, pEvec[4], 0b11110001)/*sx5,0,0,0*/, _mm_dp_ps(sp5, pEvec[9], 0b11110010)/*0,sy5,0,0*/, 0b10)/*sx5,sy5,0,0*/
		);
		const __m128 addz = _mm_add_ps
		(
			_mm_add_ps(_mm_dp_ps(sp1, pEvec[10], 0b11110100)/*0,0,sz1,0*/, _mm_dp_ps(sp2, pEvec[11], 0b11110100)/*0,0,sz2,0*/),
			_mm_add_ps(_mm_dp_ps(sp3, pEvec[12], 0b11110100)/*0,0,sz3,0*/, _mm_dp_ps(sp4, pEvec[13], 0b11110100)/*0,0,sz4,0*/)
		);
		vPoints[row].assign(_mm_add_ps
		(
			_mm_dp_ps(sp5, pEvec[14], 0b11110100)/*0,0,sz5,0*/,
			_mm_add_ps(_mm_add_ps(add01, add23), _mm_add_ps(add45, addz))
		));
	}
}

void CMesh::fastShapeChangesToMesh_AVX(const miniBLAS::Vertex *shapeParamsIn, const miniBLAS::Vertex *eigenVectorsIn)
{
	using miniBLAS::Vertex;
	const uint32_t numEigenVectors = 20;
	//row * col(3) * z(20 = 5Vertex)
	//calculate vertex-dpps-vertex =====> 20 mul -> sum, sum added to mPoints[r,c]
	const __m256 sp12 = _mm256_load_ps(shapeParamsIn[0]), sp34 = _mm256_load_ps(shapeParamsIn[2]), sp45 = _mm256_loadu_ps(shapeParamsIn[3]);
	const __m256 sp23 = _mm256_loadu_ps(shapeParamsIn[1]), sp51 = _mm256_set_m128(shapeParamsIn[0], shapeParamsIn[4]);
	const __m128 sp1 = shapeParamsIn[0], sp2 = shapeParamsIn[1], sp3 = shapeParamsIn[2], sp4 = shapeParamsIn[3], sp5 = shapeParamsIn[4];
	const Vertex *__restrict pEvec = eigenVectorsIn;
	for (uint32_t row = 0; row < mNumPoints; row++, pEvec += 16)
	{
		_mm_prefetch((const char*)(pEvec + 16), _MM_HINT_NTA);
		_mm_prefetch((const char*)(pEvec + 20), _MM_HINT_NTA);
		_mm_prefetch((const char*)(pEvec + 24), _MM_HINT_NTA);
		_mm_prefetch((const char*)(pEvec + 28), _MM_HINT_NTA);
		const __m256 addA = _mm256_add_ps
		(
			_mm256_blend_ps(
				_mm256_dp_ps(sp12, _mm256_load_ps(pEvec[0]), 0b11110001)/*sx1,0,0,0;sx2,0,0,0*/,
				_mm256_dp_ps(sp23, _mm256_load_ps(pEvec[6]), 0b11110010)/*0,sy2,0,0;0,sy3,0,0*/,
				0b00100010),
			_mm256_blend_ps(
				_mm256_dp_ps(sp34, _mm256_load_ps(pEvec[2]), 0b11110001)/*sx3,0,0,0;sx4,0,0,0*/,
				_mm256_dp_ps(sp12, _mm256_load_ps(pEvec[10]), 0b11110100)/*0,0,sz1,0;0,0,sz2,0*/,
				0b01000100)
		)/*sx13,sy2,sz1,0;sx24,sy3,sz2,0*/;
		const __m256 addB = _mm256_add_ps
		(
			_mm256_add_ps(
				_mm256_blend_ps(
					_mm256_dp_ps(sp51, _mm256_load_ps(pEvec[4]), 0b11110011)/*sx5,sx5,0,0;sy1,sy1,0,0*/,
					_mm256_setzero_ps(), 0b00011110)/*sx5,0,0,0;0,sy1,0,0*/,
				_mm256_insertf128_ps(
					_mm256_dp_ps(sp51, _mm256_broadcast_ps((__m128*)&pEvec[14]), 0b11110100)/*0,0,sz5,0;0,0,?,0*/,
					vPoints[row], 1)/*0,0,sz5,0;x,y,z,1*/
			)/*sx5,0,sz5,0;sx0,sy01,sz0,1*/,
			_mm256_blend_ps(
				_mm256_dp_ps(sp34, _mm256_load_ps(pEvec[12]), 0b11110100)/*0,0,sz3,0;0,0,sz4,0*/,
				_mm256_dp_ps(sp45, _mm256_load_ps(pEvec[8]), 0b11110010)/*0,sy4,0,0;0,sy5,0,0*/,
				0b00100010)
		)/*sx5,sy4,sz35,0;sx0,sy015,sz04,1*/;
		const __m256 addAB = _mm256_add_ps(addA, addB)/*sx135,sy24,sz135,0;sx024,sy0135,sz024,1*/;
		const __m128 newval = _mm256_castps256_ps128(_mm256_add_ps(_mm256_permute2f128_ps(addAB, addAB, 0b01), addAB));
		vPoints[row].assign(newval);
	}
}
void CMesh::fastShapeChangesToMesh_AVX(const miniBLAS::Vertex *shapeParamsIn, const miniBLAS::Vertex *eigenVectorsIn, const char *__restrict validMask)
{
	using miniBLAS::Vertex;
	const uint32_t numEigenVectors = 20;
	//row * col(3) * z(20 = 5Vertex)
	//calculate vertex-dpps-vertex =====> 20 mul -> sum, sum added to mPoints[r,c]
	const __m256 sp12 = _mm256_load_ps(shapeParamsIn[0]), sp34 = _mm256_load_ps(shapeParamsIn[2]), sp45 = _mm256_loadu_ps(shapeParamsIn[3]);
	const __m256 sp23 = _mm256_loadu_ps(shapeParamsIn[1]), sp51 = _mm256_set_m128(shapeParamsIn[0], shapeParamsIn[4]);
	const __m128 sp1 = shapeParamsIn[0], sp2 = shapeParamsIn[1], sp3 = shapeParamsIn[2], sp4 = shapeParamsIn[3], sp5 = shapeParamsIn[4];
	const Vertex *__restrict pEvec = eigenVectorsIn;
	for (uint32_t row = 0, vldIdx = 0; row < mNumPoints; row++, pEvec += 16)
	{
		_mm_prefetch((const char*)(pEvec + 16), _MM_HINT_NTA);
		_mm_prefetch((const char*)(pEvec + 20), _MM_HINT_NTA);
		_mm_prefetch((const char*)(pEvec + 24), _MM_HINT_NTA);
		_mm_prefetch((const char*)(pEvec + 28), _MM_HINT_NTA);
		const __m256 addA = _mm256_add_ps
		(
			_mm256_blend_ps(
				_mm256_dp_ps(sp12, _mm256_load_ps(pEvec[0]), 0b11110001)/*sx1,0,0,0;sx2,0,0,0*/,
				_mm256_dp_ps(sp23, _mm256_load_ps(pEvec[6]), 0b11110010)/*0,sy2,0,0;0,sy3,0,0*/,
				0b00100010),
			_mm256_blend_ps(
				_mm256_dp_ps(sp34, _mm256_load_ps(pEvec[2]), 0b11110001)/*sx3,0,0,0;sx4,0,0,0*/,
				_mm256_dp_ps(sp12, _mm256_load_ps(pEvec[10]), 0b11110100)/*0,0,sz1,0;0,0,sz2,0*/,
				0b01000100)
		)/*sx13,sy2,sz1,0;sx24,sy3,sz2,0*/;
		const __m256 addB = _mm256_add_ps
		(
			_mm256_add_ps(
				_mm256_blend_ps(
					_mm256_dp_ps(sp51, _mm256_load_ps(pEvec[4]), 0b11110011)/*sx5,sx5,0,0;sy1,sy1,0,0*/,
					_mm256_setzero_ps(), 0b00011110)/*sx5,0,0,0;0,sy1,0,0*/,
				_mm256_insertf128_ps(
					_mm256_dp_ps(sp51, _mm256_broadcast_ps((__m128*)&pEvec[14]), 0b11110100)/*0,0,sz5,0;0,0,?,0*/,
					vPoints[row], 1)/*0,0,sz5,0;x,y,z,1*/
			)/*sx5,0,sz5,0;sx0,sy01,sz0,1*/,
			_mm256_blend_ps(
				_mm256_dp_ps(sp34, _mm256_load_ps(pEvec[12]), 0b11110100)/*0,0,sz3,0;0,0,sz4,0*/,
				_mm256_dp_ps(sp45, _mm256_load_ps(pEvec[8]), 0b11110010)/*0,sy4,0,0;0,sy5,0,0*/,
				0b00100010)
		)/*sx5,sy4,sz35,0;sx0,sy015,sz04,1*/;
		const __m256 addAB = _mm256_add_ps(addA, addB)/*sx135,sy24,sz135,0;sx024,sy0135,sz024,1*/;
		const __m128 newval = _mm256_castps256_ps128(_mm256_add_ps(_mm256_permute2f128_ps(addAB, addAB, 0b01), addAB));
		vPoints[row].assign(newval);

		if (validMask[row] != 0)
			validPts[vldIdx++].assign(newval);
	}
}

int CMesh::updateJntPos()
{
	// cout << "\nUpdating Joints.. ";	
	CMatrix<float> newJntPos; newJntPos.setSize(mJointNumber, 3); //3D joints
	//CVector<float> newJnt; newJnt.setSize(3);
	CMatrix<float> tmpMatrix; tmpMatrix.setSize(weightMatrix.xSize(), weightMatrix.ySize()); tmpMatrix = 0;
	CMatrix<float> joints0; joints0.setSize(mJointNumber, mJointNumber * 3);
	CVector<float> minEle; minEle.setSize(mNumPoints);
	CVector<float> singleWts; singleWts.setSize(mNumPoints); singleWts = 0;

	static const uint8_t idxmap[][3] = //i0,i1,i1*3
	{ { 2,0,0 },{ 3,2,6 },{ 4,3,9 },{ 6,0,0 },{ 7,6,18 },{ 8,7,21 },{ 10,0,0 },{ 11,10,30 },
	{ 14,10,30 },{ 15,14,42 },{ 16,15,45 },{ 19,10,30 },{ 20,19,57 },{ 21,20,60 } };

	for (uint32_t a = 0; a < 14; ++a)
	{
		const int i0 = idxmap[a][0], i1 = idxmap[a][1], i1x3 = idxmap[a][2];
		bool anyMinIsNotZero = minMLab(weightMatrix, i0, i1, minEle);

		if (anyMinIsNotZero)
		{	//printf("\nNon-zero min wt bw joints %d and %d", i0+1, i1+1);
			componet_wise_mul_with_pnts(minEle, tmpMatrix);
			double sumMinEle = sumTheVector(minEle);
			assert(sumMinEle > 0);
			miniBLAS::Vertex sum;
			tmpMatrix.sumToY3(sum);
			sum /= sumMinEle;
			joints0.putToY3(sum, i0, i1x3);
			//below are pieces of shit.
			/*
			CVector<float> t1; t1.setSize(tmpMatrix.ySize());
			sumTheMatrix(tmpMatrix, t1);
			newJnt(0) = t1(0) / sumMinEle;
			newJnt(1) = t1(1) / sumMinEle;
			newJnt(2) = t1(2) / sumMinEle;
			//what the fuck are you doing, newJnt?
			joints0(i0, 3 * i1 + 0) = newJnt(0);
			joints0(i0, 3 * i1 + 1) = newJnt(1);
			joints0(i0, 3 * i1 + 2) = newJnt(2);
			*/
		}
	}
	//below are all useless
	/*
	copyMatColToVector(weightMatrix, 0, singleWts);
	tmpMatrix = 0;//what the fuck? tmpMatrix will always be all-zero
	componet_wise_mul_with_pnts(singleWts, tmpMatrix);

	miniBLAS::Vertex sum;
	tmpMatrix.sumToY3(sum);
	float sumRootElse = sumTheVector(singleWts); assert(sumRootElse > 0);
	if (sumRootElse <= 0) sumRootElse = 1;
	sum /= sumRootElse;
	*/
	miniBLAS::Vertex sum(true);
	newJntPos.putToY3(sum, 0, 0);
	//shit code again
	/*
	CVector<float> t1; t1.setSize(tmpMatrix.ySize());
	sumTheMatrix(tmpMatrix, t1);
	double sumRootElse = sumTheVector(singleWts); assert(sumRootElse > 0);
	if (sumRootElse <= 0) sumRootElse = 1;
	newJntPos(0, 0) = t1(0) / sumRootElse;
	newJntPos(0, 1) = t1(1) / sumRootElse;
	newJntPos(0, 2) = t1(2) / sumRootElse;
	*/
	int parentMap[50]; // jNo=(jId -1) (to)---> parent jId
	for (unsigned int i0 = 0; i0 < mJointNumber; i0++)
	{
		int jId = i0 + 1;
		parentMap[i0] = mJoint(jId).mParent;
	}

	parentMap[2] = 1;
	parentMap[3] = 3;
	parentMap[5] = 5;
	parentMap[6] = 1;
	parentMap[7] = 7;
	parentMap[8] = 8;
	parentMap[9] = 9;
	parentMap[10] = 1;
	parentMap[14] = 11;
	parentMap[19] = 11;
	parentMap[22] = 22;
	parentMap[23] = 23;
	parentMap[24] = 24;
	parentMap[25] = 25;

	std::set<int> nonRoots; //has jId's

	nonRoots.insert(3); nonRoots.insert(4);
	nonRoots.insert(5); nonRoots.insert(7);
	nonRoots.insert(8); nonRoots.insert(9);
	nonRoots.insert(11); nonRoots.insert(12);
	nonRoots.insert(15); nonRoots.insert(16);
	nonRoots.insert(17); nonRoots.insert(20);
	nonRoots.insert(21); nonRoots.insert(22);
	nonRoots.insert(23); nonRoots.insert(24);
	nonRoots.insert(25);

	for (unsigned int i0 = 0; i0 < mJointNumber; i0++)
	{
		int jId = i0 + 1;
		if (nonRoots.count(jId) != 0)
		{	//non root joints
			newJntPos(i0, 0) = joints0(i0, (parentMap[i0] - 1) * 3 + 0);
			newJntPos(i0, 1) = joints0(i0, (parentMap[i0] - 1) * 3 + 1);
			newJntPos(i0, 2) = joints0(i0, (parentMap[i0] - 1) * 3 + 2);
		}
	}
	copyJointPos(1, 2, newJntPos);
	copyJointPos(5, 6, newJntPos);
	copyJointPos(9, 10, newJntPos);
	copyJointPos(12, 14, newJntPos);
	copyJointPos(13, 14, newJntPos);
	copyJointPos(17, 19, newJntPos);
	copyJointPos(18, 19, newJntPos);
	copyJointPos(22, 6, newJntPos);
	copyJointPos(23, 2, newJntPos);
	copyJointPos(24, 10, newJntPos);

	for (unsigned int i0 = 0; i0 < mJointNumber; i0++)
	{
		unsigned int jId = i0 + 1;
		CVector<float> jPos(3);
		jPos(0) = newJntPos(i0, 0);
		jPos(1) = newJntPos(i0, 1);
		jPos(2) = newJntPos(i0, 2);
		mJoint(i0 + 1).setPoint(jPos);
	}
	return 1;
}

static void sumVec(__m128 *ptr, const int32_t cnt, const uint32_t step = 1)
{
	int32_t a = 0;
	const int32_t off1 = 1 << step, off2 = 2 << step;
	const int32_t r0 = 1 << (step - 1), r1 = off1 + r0, r2 = off2 + r0;
	const int32_t adder = 6 * r0;
	for (int32_t times = (cnt + r0 - 1) / adder; times--; a += adder)
	{
		ptr[a +    0] = _mm_add_ps(ptr[a +    0], ptr[a + r0]);
		ptr[a + off1] = _mm_add_ps(ptr[a + off1], ptr[a + r1]);
		ptr[a + off2] = _mm_add_ps(ptr[a + off2], ptr[a + r2]);
	}
	switch (((cnt - a) + r0 - 1) / r0)
	{
	case 5:
		ptr[a +    0] = _mm_add_ps(ptr[a +    0], ptr[a + off2]);
	case 4:
		ptr[a + off1] = _mm_add_ps(ptr[a + off1], ptr[a + r1]);
	case 3:
		ptr[a +    0] = _mm_add_ps(ptr[a +    0], ptr[a + off1]);
	case 2:
		ptr[a +    0] = _mm_add_ps(ptr[a +    0], ptr[a + r0]);
	case 1:
		a++;
		break;
	case 0:
		a = cnt - step;
	default://wrong
		printf("seems wrong here");
		break;
	}
	if (a <= 1)
		return;
	sumVec(ptr, a, step + 1);
}

void CMesh::updateJntPosEx()
{
	using miniBLAS::Vertex;
	static const uint8_t idxmap[][2] = //i0,i1*3
	{ { 2,0 },{ 3,6 },{ 4,9 },{ 6,0 },{ 7,18 },{ 8,21 },{ 10,0 },{ 11,30 }, { 14,30 },{ 15,42 },{ 16,45 },{ 19,30 },{ 20,57 },{ 21,60 } };

	CMatrix<float> newJntPos(mJointNumber, 3);//3D joints
	CMatrix<float> joints0(mJointNumber, mJointNumber * 3);

	const Vertex *__restrict pMinWgt = theMinWgtMat;
	uint32_t mIdx = 0;
	for (const uint8_t(&item)[2] : idxmap)
	{
		//calc min
		__m256 sumPosA = _mm256_setzero_ps(), sumPosB = _mm256_setzero_ps();
		const Vertex *__restrict pOri = &vPoints[0];
		for (uint32_t b = wMatGap / 2; b--; pOri += 8, pMinWgt += 2)
		{
			const __m256 wgt = _mm256_load_ps(pMinWgt[0]);
			const __m256 tmpA = _mm256_add_ps
			(
				_mm256_mul_ps(_mm256_permute_ps(wgt, 0b00000000), _mm256_load_ps(pOri[0])),
				_mm256_mul_ps(_mm256_permute_ps(wgt, 0b01010101), _mm256_load_ps(pOri[2]))
			);
			const __m256 tmpB = _mm256_add_ps
			(
				_mm256_mul_ps(_mm256_permute_ps(wgt, 0b10101010), _mm256_load_ps(pOri[4])),
				_mm256_mul_ps(_mm256_permute_ps(wgt, 0b11111111), _mm256_load_ps(pOri[6]))
			);
			sumPosA = _mm256_add_ps(sumPosA, tmpA);
			sumPosB = _mm256_add_ps(sumPosB, tmpB);
		}
		sumPosA = _mm256_add_ps(sumPosA, sumPosB);
		sumPosA = _mm256_add_ps(sumPosA, _mm256_permute2f128_ps(sumPosA, sumPosA, 0b00000001));
		const Vertex puter(_mm_div_ps(_mm256_castps256_ps128(sumPosA), theMinEleSum[mIdx++]));
		joints0.putToY3(puter, item[0], item[1]);
	}

	const static Vertex sum(true);
	newJntPos.putToY3(sum, 0, 0);

	int parentMap[50]; // jNo=(jId -1) (to)---> parent jId
	for (unsigned int i0 = 0; i0 < mJointNumber; i0++)
	{
		int jId = i0 + 1;
		parentMap[i0] = mJoint(jId).mParent;
	}

	parentMap[2] = 1;
	parentMap[3] = 3;
	parentMap[5] = 5;
	parentMap[6] = 1;
	parentMap[7] = 7;
	parentMap[8] = 8;
	parentMap[9] = 9;
	parentMap[10] = 1;
	parentMap[14] = 11;
	parentMap[19] = 11;
	parentMap[22] = 22;
	parentMap[23] = 23;
	parentMap[24] = 24;
	parentMap[25] = 25;

	const static uint32_t nonRootId[] = { 2,3,4,6,7,8,10,11,14,15,16,19,20,21,22,23,24 };
	for (const uint32_t id : nonRootId)
	{
		const int tmp = (parentMap[id] - 1) * 3;
		newJntPos(id, 0) = joints0(id, tmp);
		newJntPos(id, 1) = joints0(id, tmp + 1);
		newJntPos(id, 2) = joints0(id, tmp + 2);
	}

	copyJointPos(1, 2, newJntPos);
	copyJointPos(5, 6, newJntPos);
	copyJointPos(9, 10, newJntPos);
	copyJointPos(12, 14, newJntPos);
	copyJointPos(13, 14, newJntPos);
	copyJointPos(17, 19, newJntPos);
	copyJointPos(18, 19, newJntPos);
	copyJointPos(22, 6, newJntPos);
	copyJointPos(23, 2, newJntPos);
	copyJointPos(24, 10, newJntPos);
	for (uint32_t i0 = 0; i0 < mJointNumber; i0++)
	{
		CVector<float> jPos(3);
		jPos(0) = newJntPos(i0, 0);
		jPos(1) = newJntPos(i0, 1);
		jPos(2) = newJntPos(i0, 2);
		mJoint(i0 + 1).setPoint(jPos);
	}
}

bool CMesh::minMLab(CMatrix<float> weightMatrix, int i0, int i1, CVector<float> &minEle)
{
	bool isNotZero = false;
	for (int i2 = 0; i2 < weightMatrix.xSize(); i2++)
	{
		float w1 = weightMatrix(i2, i0);
		float w2 = weightMatrix(i2, i1);
		if (weightMatrix(i2, i0) < weightMatrix(i2, i1))
			minEle(i2) = weightMatrix(i2, i0);
		else
			minEle(i2) = weightMatrix(i2, i1);
		if (minEle(i2) > 0) isNotZero = true;
	}
	return isNotZero;
}

void CMesh::componet_wise_mul_with_pnts(CVector<float> minEle, CMatrix<float> &tmpMatrix)
{
	assert(mPoints.size() == minEle.size());
	bool allZero = true;
	for (int i0 = 0; i0 < mPoints.size(); i0++)
	{
		if (minEle(i0) > 0)
		{
			double m_ele = minEle(i0);
			allZero = false;
		}
		float tmp[3];
		for (int i1 = 0; i1 < 3; i1++)
		{
			tmp[i1] = (mPoints[i0])(i1);
			tmpMatrix(i0, i1) = (mPoints[i0])(i1) * minEle(i0);
		}
	}
	if (allZero == true)
		printf("\nSomeing the matter, all weights were zero.");
}

void CMesh::componet_wise_mul_with_pntsEx(const float *minEle, CMatrix<float> &tmpMatrix)
{
	bool allZero = true;
	for (int i0 = 0; i0 < mPoints.size(); i0++)
	{
		const float obj = minEle[i0];
		if (obj > 0)
			allZero = false;
		for (int i1 = 0; i1 < 3; i1++)
			tmpMatrix(i0, i1) = (mPoints[i0])(i1) * obj;
	}
	if (allZero == true)
		printf("\nSomeing the matter, all weights were zero.");
}

double CMesh::sumTheVector(CVector<float> minEle)
{
	double sum = 0.;
	for (int i0 = 0; i0 < minEle.size(); i0++)
		sum += minEle(i0);
	return sum;
}

void CMesh::sumTheMatrix(CMatrix<float> tmpMatrix, CVector<float> & t1)
{
	t1 = 0.;
	for (int i0 = 0; i0 < tmpMatrix.xSize(); i0++)
	{
		for (int i1 = 0; i1 < tmpMatrix.ySize(); i1++)
		{
			t1(i1) += tmpMatrix(i0, i1);
		}
	}
}

bool CMesh::copyMatColToVector(CMatrix<float> weightMatrix, int col, CVector<float> &singleWts)
{
	bool isNotZero = false;
	double total = 0;
	for (int i0 = 0; i0 < weightMatrix.xSize(); i0++)
	{
		singleWts(i0) = weightMatrix(i0, col);
		if (singleWts(i0) > 0) isNotZero = true;
		total += weightMatrix(i0, col);
	}
	return isNotZero;
}

void CMesh::copyJointPos(int to, int from, CMatrix<float> &newJntPos)
{
	newJntPos(to, 0) = newJntPos(from, 0);
	newJntPos(to, 1) = newJntPos(from, 1);
	newJntPos(to, 2) = newJntPos(from, 2);
}

void CMesh::printPoints(std::string fname)
{
	NShow::mLogFile.open((NShow::mResultDir + fname).c_str(), std::ios_base::app);
	for (unsigned int i0 = 0; i0 < mPoints.size(); i0++)
	{
		NShow::mLogFile << mPoints[i0](0) << " " <<
			mPoints[i0](1) << " " <<
			mPoints[i0](2) << "\n";
	}

	NShow::mLogFile.close();
}

void CMesh::findNewAxesOrient(CVector<double> &axisLeftArm, CVector<double> &axisRightArm, CMatrix<float> newJntPos)
{
	CVector<double> upperArmL, upperArmR, lowerArmL, lowerArmR;
	upperArmL.setSize(3); lowerArmL.setSize(3);
	upperArmR.setSize(3); lowerArmR.setSize(3);
#define WRISTPTS 34
	unsigned int rightWristPts[WRISTPTS] = { 2842, 2843, 2845, 2846, 2847, 2848, 2849, 2851, 2852, 2853, 2855,
		2856, 2857, 2858, 2865, 2866, 2867, 2870, 2871, 2872, 2873, 2876, 2879, 2880, 2884, 2885,
		2886, 2887, 2888, 2914, 2932, 2934, 2957 };

	unsigned int leftWristPts[WRISTPTS] = { 5971, 5972, 5973, 5975, 5976, 5977, 5978, 5979, 5981, 5982,
		5983, 5985, 5986, 5987, 5988, 5995, 5996, 5997, 6000, 6001, 6002, 6003, 6006, 6009, 6010,
		6014, 6015, 6016, 6017, 6018, 6044, 6062, 6064, 6087 };
	CVector<double> leftWrist, rightWrist;
	leftWrist.setSize(3); rightWrist.setSize(3);
	leftWrist = rightWrist = 0;

	for (unsigned int i0 = 0; i0 < WRISTPTS; i0++)
	{
		unsigned int lIdx = leftWristPts[i0], rIdx = rightWristPts[i0];
		for (unsigned int i1 = 0; i1 < 3; i1++)
		{

			leftWrist(i1) += (mPoints[lIdx])(i1);
			rightWrist(i1) += (mPoints[rIdx])(i1);
		}
	}

	for (unsigned int i1 = 0; i1 < 3; i1++)
	{
		leftWrist(i1) = leftWrist(i1) / WRISTPTS;
		rightWrist(i1) = leftWrist(i1) / WRISTPTS;
	}

	for (unsigned i0 = 0; i0 < 3; i0++)
	{
		upperArmR(i0) = newJntPos(7, i0) - newJntPos(6, i0);
		lowerArmR(i0) = rightWrist(i0) - newJntPos(7, i0);
		upperArmL(i0) = newJntPos(11, i0) - newJntPos(10, i0);
		lowerArmL(i0) = leftWrist(i0) - newJntPos(11, i0);
	}
	double normUpArmR = upperArmR.norm();
	double normLowArmR = lowerArmR.norm();
	double normUpArmL = upperArmL.norm();
	double normLowArmL = lowerArmL.norm();
	for (unsigned i0 = 0; i0 < 3; i0++)
	{
		upperArmR(i0) /= normUpArmR;
		lowerArmR(i0) /= normLowArmR;
		upperArmL(i0) /= normUpArmL;
		lowerArmL(i0) /= normLowArmL;
	}
	//////////////////////////////////////////////////////////////////////////
	axisLeftArm = upperArmL / lowerArmL;
	axisRightArm = upperArmR / lowerArmR;

	double normAxisLeftArm = axisLeftArm.norm();
	double normAxisRightArm = axisRightArm.norm();

	if (normAxisLeftArm == 0 || axisRightArm == 0)
	{
		printf("\n!!!!--- WORRY WORRY AXIS 0 ---!!!");
		assert(0);
	}
	for (unsigned i0 = 0; i0 < 3; i0++)
	{
		axisLeftArm(i0) /= normAxisLeftArm;
		axisRightArm(i0) /= normAxisRightArm;
	}
}

void CMesh::writeMeshDat(std::string fname)
{
	NShow::mLogFile.open(fname.c_str());
	char buffer[10000];
	sprintf(buffer, "%d %d %d %d\n", mPoints.size(), mPatch.size(), joints(), mNoOfBodyParts);
	NShow::mLogFile << buffer;
	for (unsigned int i0 = 0; i0 < mNumPoints; i0++)
	{
		sprintf(buffer, "%f %f %f", mPoints[i0](0), mPoints[i0](1), mPoints[i0](2));
		NShow::mLogFile << buffer;
		getWeightsinBuffer(buffer, i0);
		NShow::mLogFile << buffer << "\n";
	}
	for (unsigned int i0 = 0; i0 < mPatch.size(); i0++)
	{
		sprintf(buffer, "%d %d %d\n", mPatch[i0](0), mPatch[i0](1), mPatch[i0](2));
		NShow::mLogFile << buffer;
	}
	for (unsigned int k0 = 0; k0 < mJointNumber; k0++)
	{
		int jId = k0 + 1;
		CVector<float> &t_jPos = mJoint(jId).getPoint();
		CVector<float> &t_jDir = mJoint(jId).getDirection();
		sprintf(buffer, "%d %f %f %f %f %f %f %d\n", jId,
			t_jDir(0), t_jDir(1), t_jDir(2), t_jPos(0), t_jPos(1), t_jPos(2),
			mJoint(jId).mParent);
		NShow::mLogFile << buffer;
	}
	NShow::mLogFile.close();
	cout << "\bDone Writing File!" << fname;
}

void CMesh::getWeightsinBuffer(char *buffer, int ptNo)
{
	std::vector< std::pair<double, int> > wts;
	std::pair<double, int> tmp;
	string wtStr = "";
	for (int i1 = joints() - 1; i1 >= 0; i1--)
	{
		if (mBoundJoints(i1 + 1) == true)
			wts.push_back(make_pair(weightMatrix(ptNo, i1), i1));
	}

	for (int i = 0; i < wts.size(); ++i)
	{
		for (int j = 1; j < wts.size() - i; ++j)
		{
			if (wts[j - 1].first < wts[j].first)
			{
				tmp = wts[j - 1];
				wts[j - 1] = wts[j];
				wts[j] = tmp;
			}
		}
	}

	for (int i0 = 0; i0 < wts.size(); i0++)
	{
		sprintf(buffer, " %d %f", wts[i0].second + 1, wts[i0].first);
		wtStr += buffer;
	}
	strcpy(buffer, wtStr.c_str());
}
