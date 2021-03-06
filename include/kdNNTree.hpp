#pragma once
/*
 * Use CRTP to save some code, for better maintainance
**/
#include <cstdint>

#include "miniBLAS.hpp"

namespace miniBLAS
{

struct NNResult;

//A.search(B) means, for each points in B, find its nearest point in A
template<typename CHILD>
class NNTreeBase
{
protected:
	VertexVec tree[8];
	VertexVec ntree[8];
	/*points may be devided into different parts, so its original index must be stored.
	 *indexes are stored seperated to avoid abundant calculation.
	 *Though it's vector<Vertex> (float), the data are uint32_t.
	 *They will not be in calculation, so use float AVX to avoid lentancy in switch float/integer AVX pipeline
	 **/
	VertexVec idtree[8];
	uint32_t ptCount;
	static const uint32_t minbase = 8;
	inline int judgeIdx(const Vertex& v) const
	{
		return ((const CHILD&)(*this)).judgeIdx(v);
	};
public:
	float MAXDist2 = 5e4f;
	NNTreeBase() = default;
	~NNTreeBase() = default;
	/*points should contain its index in w, though they may be the same as its index in vector*/
	void init(const VertexVec& points, const VertexVec& normals, const uint32_t nPoints);
	void searchBasic(const Vertex *__restrict pVert, const uint32_t count, int *idxs, float *dists) const;
	void search(const Vertex *pVert, const uint32_t count, int *__restrict idxs, float *__restrict dists) const;
	void searchOnAngle(NNResult& res, const VertexVec& pt, const VertexVec& norm, const float angle) const;
	//Search will consider angle.angMax means max-angle allowed. However, angle bigger than angleLim will receive pentalty on distance
	//For A.search(B), since the normal of scanbody is not reliable(sometimes 0,0,0),
	//some accurate match will be abandoned for its not satisfying angle limit.
	//Also, body surface tends to be smooth so their normals could be similar.
	//Hence, there is a situation that multi points in B matches the same point in A.
	//According to d2threashold, function store the limit distance for each point in A(min dist*d2threshold),
	//and we can use the value(stored in NNResult) to filter unwant match pairs
	void searchOnAnglePan(NNResult& res, const VertexVec& pt, const VertexVec& norm, const float angMax, const float angleLim, const float d2threshold = 1.25f/*1.12^2*/) const;
	uint32_t PTCount() const { return ptCount + 7; };
};

/*container for the result of nnsearch*/
struct NNResult
{
private:
	template<typename T>
	friend class NNTreeBase;
	uint32_t baseSize, objSize;
public:
	std::vector<int32_t> idxs;
	std::vector<float> dists;
	std::vector<float> mdists;
	std::vector<uint8_t> mthcnts;
	NNResult(const uint32_t oSize, const uint32_t bSize) : objSize(oSize), baseSize(bSize)
	{
		idxs.resize(objSize, 0);
		dists.resize(objSize);
		mdists.resize(baseSize, 1e10f);
		mthcnts.resize(baseSize, 0);
	};
};


template<typename CHILD>
void NNTreeBase<CHILD>::init(const VertexVec& points, const VertexVec& normals, const uint32_t nPoints)
{
	for (uint32_t a = 0, s = nPoints / 8; a < 8; ++a)
	{
		tree[a].clear();
		tree[a].reserve(nPoints / 8);
		ntree[a].clear();
		ntree[a].reserve(nPoints / 8);
	}

	for (uint32_t i = 0; i < nPoints; ++i)
	{
		Vertex v = points[i]; v.int_w = i;
		const auto tid = judgeIdx(v);
		tree[tid].push_back(v);
		ntree[tid].push_back(normals[i]);
	}

	ptCount = nPoints;
	static const Vertex empty(1e10f, 1e10f, 1e10f);
	static const Vertex emptyN(0, 0, 0);
	for (uint32_t a = 0, s = nPoints / 8; a < 8; ++a)
	{
		auto& pts = tree[a];
		for (uint32_t b = pts.size() % minbase; b != 0 && b < minbase; ++b)
		{
			pts.push_back(empty);
			ntree[a].push_back(emptyN);
		}
		const uint32_t idsize = pts.size() / 4;
		auto& ids = idtree[a];
		ids.resize(idsize);
		for (uint32_t b = 0, c = 0; b < idsize; b += 2, c += 8)
		{
			//calculate 4 vector represent 8 (point-Obj ---> point-Base)
			const __m256 vb01 = _mm256_load_ps(pts[c + 0]), vb23 = _mm256_load_ps(pts[c + 2]), 
				vb45 = _mm256_load_ps(pts[c + 4]), vb67 = _mm256_load_ps(pts[c + 6]);
			//make up vector contain 8 new idx from point-base's extra idx data
			const __m256 id8 = _mm256_shuffle_ps(_mm256_unpackhi_ps(vb01, vb23)/*xx,i1,i3;xx,i2,i4*/,
				_mm256_unpackhi_ps(vb45, vb67)/*xx,i5,i7;xx,i6,i8*/, 0b11101110);
			/*i1,i3,i5,i7;i2,i4,i6,i8*/
			_mm256_store_ps(ids[b], id8);
		}
	}
}

template<typename CHILD>
void NNTreeBase<CHILD>::searchBasic(const Vertex *__restrict pVert, const uint32_t count, int *idxs, float *dists) const
{
	for (uint32_t a = 0; a < count; ++a, ++pVert)
	{
		float minval = 65536;
		uint32_t minidx = 65536;
		for (const auto & __restrict vBase : tree[judgeIdx(*pVert)])
		{
			const float dis = (*pVert - vBase).length_sqr();
			if (dis < minval)
			{
				minidx = vBase.int_w;
				minval = dis;
			}
		}
		*idxs++ = minidx;
		*dists++ = minval;
	}
}

/*output dists are length, not square of length*/
template<typename CHILD>
void NNTreeBase<CHILD>::search(const Vertex *pVert, const uint32_t count, int *__restrict idxs, float *__restrict dists) const
{
	const uint32_t cntV = (count + 7) / 4;
	Vertex *tmp = new Vertex[cntV];
	float *pFtmp = tmp[0];
	for (uint32_t a = count; a--; ++pVert)
	{
		//object vertex being searched
		const Vertex vObj = *pVert; const __m256 mObj = _mm256_broadcast_ps((__m128*)&vObj);
		//find proper subtree
		const auto& part = tree[judgeIdx(vObj)];
		const float *__restrict pBase = part[0];
		//min dist * 8   AND   min idx * 8
		__m256 min8 = _mm256_set1_ps(1e10f); __m256 minpos8 = _mm256_setzero_ps();

		for (uint32_t b = part.size() / 8; b--; )
		{
			//calculate 4 vector represent 8 (point-Obj ---> point-Base)
			const __m256 vb01 = _mm256_load_ps(pBase + 0), a1 = _mm256_sub_ps(mObj, vb01);
			const __m256 vb23 = _mm256_load_ps(pBase + 8), a2 = _mm256_sub_ps(mObj, vb23);
			const __m256 vb45 = _mm256_load_ps(pBase + 16), a3 = _mm256_sub_ps(mObj, vb45);
			const __m256 vb67 = _mm256_load_ps(pBase + 24), a4 = _mm256_sub_ps(mObj, vb67);

			//prefetch
			_mm_prefetch((const char*)(pBase += 32), _MM_HINT_T1);

			//make up vector contain 8 dist data(dist^2)
			__m256 this8 = _mm256_blend_ps
			(
				_mm256_blend_ps(_mm256_dp_ps(a1, a1, 0x71)/*d1,000;d2,000*/, _mm256_dp_ps(a2, a2, 0x72)/*0,d3,00;0,d4,00*/, 0x22)/*d1,d3,00;d2,d4,00*/,
				_mm256_blend_ps(_mm256_dp_ps(a3, a3, 0x74)/*00,d5,0;00,d6,0*/, _mm256_dp_ps(a4, a4, 0x78)/*000,d7;000,d8*/, 0x88)/*00,d5,d7;00,d6,d8*/,
				0b11001100
			)/*d1,d3,d5,d7;d2,d4,d6,d8*/;

			//find out which idx need to be updated and refresh min dist
			const __m256 mask = _mm256_cmp_ps(this8, min8, _CMP_LT_OS);
			min8 = _mm256_min_ps(min8, this8);

			//if it neccessary to update idx
			if (_mm256_movemask_ps(mask))
			{
				//make up vector contain 4 new idx from point-base's extra idx data
				this8 = _mm256_shuffle_ps(_mm256_unpackhi_ps(vb01, vb23)/*xx,i1,i3;xx,i2,i4*/,
					_mm256_unpackhi_ps(vb45, vb67)/*xx,i5,i7;xx,i6,i8*/, 0b11101110)/*i1,i3,i5,i7;i2,i4,i6,i8*/;

				//refresh min idx
				minpos8 = _mm256_blendv_ps(minpos8, this8, mask);
			}
		}
		//after uprolled search, need to extract min dist&idx among 8 data
		{
			//float tmpdat[8]; _mm256_storeu_ps(tmpdat, min8);
			//printf("min8 = %e,%e,%e,%e;%e,%e,%e,%e\n", tmpdat[0], tmpdat[1], tmpdat[2], tmpdat[3], tmpdat[4], tmpdat[5], tmpdat[6], tmpdat[7]);
			// find out whether each dist is the min among 8,
			// consider they could be all the same so "less OR equal" should be used
			const __m256 com1 = _mm256_cmp_ps(min8, _mm256_permute_ps(min8, _MM_SHUFFLE(0, 3, 2, 1)), _CMP_LE_OS)/*a<=b,b<=c,c<=d,d<=a*/;
			const __m256 com2 = _mm256_cmp_ps(min8, _mm256_permute_ps(min8, _MM_SHUFFLE(1, 0, 3, 2)), _CMP_LE_OS)/*a<=c,b<=d,c<=a,d<=b*/;
			const __m256 com3 = _mm256_cmp_ps(min8, _mm256_permute_ps(min8, _MM_SHUFFLE(2, 1, 0, 3)), _CMP_LE_OS)/*a<=d,b<=a,c<=b,d<=c*/;
			const int res = _mm256_movemask_ps(_mm256_and_ps(_mm256_and_ps(com1, com2), com3))/*1357;2468*/;
			// final result may contain multi "1"(means min) when some dists are the same
			// so need to use bit scan to find a pos(whatever pos)
		#if defined(__GNUC__)
			const int idx0 = _bit_scan_forward(res), idx1 = _bit_scan_reverse(res);
		#else
			unsigned long idx0, idx1;
			_BitScanForward(&idx0, res); _BitScanReverse(&idx1, res);
		#endif
			// neccessary to copy data out from __m128, 
			// or you may not be able to get the true result from mem,
			// since they may still be in register
			int ALIGN32 theIDX[8]; float ALIGN32 theDIST[8];
			_mm256_store_ps((float*)theIDX, minpos8); _mm256_store_ps(theDIST, min8);
			if (theDIST[idx0] <= theDIST[idx1])
			{
				*idxs = theIDX[idx0];
				*pFtmp = theDIST[idx0];
			}
			else
			{
				*idxs = theIDX[idx1];
				*pFtmp = theDIST[idx1];
			}
			//printf("res:%d,idx:%d; ==== DIST %e; IDX %d\n", res, idx0, *pFtmp, *idxs);
			idxs++; pFtmp++;
		}
	}
	float *pTmp = tmp[0];
	for (uint32_t a = cntV / 2; a--; pTmp += 8)
		_mm256_store_ps(pTmp, _mm256_sqrt_ps(_mm256_load_ps(pTmp)));
	memcpy(dists, tmp, count * sizeof(float));
	delete[] tmp;
}

template<typename CHILD>
void NNTreeBase<CHILD>::searchOnAngle(NNResult& ret, const VertexVec& pt, const VertexVec& norm, const float angle) const
{
	const uint32_t count = ret.objSize;
	int *idxs = ret.idxs.data(); float *dists = ret.dists.data();
	const __m256 mincos = _mm256_set1_ps(cos(3.1415926 * angle / 180));
	const Vertex *pVert = &pt[0], *pNorm = &norm[0];
	for (uint32_t a = count; a--; ++pVert, ++pNorm)
	{
		if (*idxs > 65530)//fast skip
		{
			idxs++;
			*dists++ = 1e10f;
			continue;
		}
		//object vertex being searched
		const Vertex vObj = *pVert; const __m256 mObj = _mm256_broadcast_ps((__m128*)pVert), mNorm = _mm256_broadcast_ps((__m128*)pNorm);
		//find proper subtree
		const auto tid = judgeIdx(vObj);
		const auto& part = tree[tid];
		const float *__restrict pBase = part[0], *__restrict pNBase = ntree[tid][0], *__restrict pID = idtree[tid][0];
		//min dist * 8   AND   min idx * 8
		__m256 min8 = _mm256_set1_ps(1e10f), minpos8 = _mm256_setzero_ps();

		for (uint32_t b = part.size() / 8; b--; pID += 8)
		{
			//calculate 4 vector represent 8 (point-Obj ---> point-Base)
			const __m256 vn01 = _mm256_load_ps(pNBase + 0), a1 = _mm256_sub_ps(mObj, _mm256_load_ps(pBase + 0));
			const __m256 vn23 = _mm256_load_ps(pNBase + 8), a2 = _mm256_sub_ps(mObj, _mm256_load_ps(pBase + 8));
			const __m256 vn45 = _mm256_load_ps(pNBase + 16), a3 = _mm256_sub_ps(mObj, _mm256_load_ps(pBase + 16));
			const __m256 vn67 = _mm256_load_ps(pNBase + 32), a4 = _mm256_sub_ps(mObj, _mm256_load_ps(pBase + 24));

			//prefetch
			_mm_prefetch((const char*)(pBase += 32), _MM_HINT_T1);
			_mm_prefetch((const char*)(pNBase += 32), _MM_HINT_T1);

			const __m256 cos8 = _mm256_blend_ps
			(
				_mm256_blend_ps(_mm256_dp_ps(vn01, mNorm, 0x71)/*c1,000;c2,000*/, _mm256_dp_ps(vn23, mNorm, 0x72)/*0,c3,00;0,c4,00*/,
					0x22)/*c1,c3,00;c2,c4,00*/,
				_mm256_blend_ps(_mm256_dp_ps(vn45, mNorm, 0x74)/*00,c5,0;00,c6,0*/, _mm256_dp_ps(vn67, mNorm, 0x78)/*000,c7;000,c8*/,
					0x88)/*00,c5,c7;00,c6,c8*/,
				0b11001100
			)/*c1,c3,c5,c7;c2,c4,c6,c8*/;
			const __m256 cosRes = _mm256_cmp_ps(cos8, mincos, _CMP_GE_OS);
			if (!_mm256_movemask_ps(cosRes))//angle all unsatisfied
				continue;

			//make up vector contain 8 dist data(dist^2)
			const __m256 this8 = _mm256_blend_ps
			(
				_mm256_blend_ps(_mm256_dp_ps(a1, a1, 0x71)/*d1,000;d2,000*/, _mm256_dp_ps(a2, a2, 0x72)/*0,d3,00;0,d4,00*/, 0x22)/*d1,d3,00;d2,d4,00*/,
				_mm256_blend_ps(_mm256_dp_ps(a3, a3, 0x74)/*00,d5,0;00,d6,0*/, _mm256_dp_ps(a4, a4, 0x78)/*000,d7;000,d8*/, 0x88)/*00,d5,d7;00,d6,d8*/,
				0b11001100
			)/*d1,d3,d5,d7;d2,d4,d6,d8*/;

			//find out which idx need to be updated(less than current and satisfy angle-requierment)
			const __m256 mask = _mm256_and_ps(_mm256_cmp_ps(this8, min8, _CMP_LT_OS), cosRes);
			min8 = _mm256_blendv_ps(min8, this8, mask);

			//if it neccessary to update idx
			if (_mm256_movemask_ps(mask))
			{
				//load idx data and refresh min idx
				minpos8 = _mm256_blendv_ps(minpos8, _mm256_load_ps(pID), mask);
			}
		}
		// after uprolled search, need to extract min dist&idx among 8 data
		// find out whether each dist is the min among 8,
		// consider they could be all the same so "less OR equal" should be used
		const __m256 com1 = _mm256_cmp_ps(min8, _mm256_permute_ps(min8, _MM_SHUFFLE(0, 3, 2, 1)), _CMP_LE_OS)/*a<=b,b<=c,c<=d,d<=a*/;
		const __m256 com2 = _mm256_cmp_ps(min8, _mm256_permute_ps(min8, _MM_SHUFFLE(1, 0, 3, 2)), _CMP_LE_OS)/*a<=c,b<=d,c<=a,d<=b*/;
		const __m256 com3 = _mm256_cmp_ps(min8, _mm256_permute_ps(min8, _MM_SHUFFLE(2, 1, 0, 3)), _CMP_LE_OS)/*a<=d,b<=a,c<=b,d<=c*/;
		const int res = _mm256_movemask_ps(_mm256_and_ps(_mm256_and_ps(com1, com2), com3))/*1357;2468*/;
		// final result may contain multi "1"(means min) when some dists are the same
		// so need to use bit scan to find a pos(whatever pos)
	#if defined(__GNUC__)
		const int idx0 = _bit_scan_forward(res), idx1 = _bit_scan_reverse(res);
	#else
		unsigned long idx0, idx1;
		_BitScanForward(&idx0, res); _BitScanReverse(&idx1, res);
	#endif
		// neccessary to copy data out from __m128, 
		// or you may not be able to get the true result from mem,
		// since they may still be in register
		int ALIGN32 theIDX[8]; float ALIGN32 theDIST[8];
		_mm256_store_ps((float*)theIDX, minpos8); _mm256_store_ps(theDIST, min8);
		int IDX; float DIST;
		if (theDIST[idx0] <= theDIST[idx1])
		{
			IDX = theIDX[idx0]; DIST = theDIST[idx0];
		}
		else
		{
			IDX = theIDX[idx1]; DIST = theDIST[idx1];
		}
		*dists = DIST;
		if (DIST > MAXDist2)
			*idxs = 65536;
		else
		{
			*idxs = IDX;
			ret.mthcnts[IDX]++;
			DIST *= 1.25f;//1.12*1.12
			if (ret.mdists[IDX] > DIST)
				ret.mdists[IDX] = DIST;
		}
		idxs++; dists++;
	}
}

template<typename CHILD>
void NNTreeBase<CHILD>::searchOnAnglePan(NNResult& ret, const VertexVec& pt, const VertexVec& norm, const float angMax, const float angLim, 
	const float d2threshold) const
{
	const uint32_t count = ret.objSize;
	int *idxs = ret.idxs.data(); float *dists = ret.dists.data();
	//angle's cos decrease when angle's increasing
	const __m256 mincos = _mm256_set1_ps(cos(3.1415926 * angMax / 180)), limcos = _mm256_set1_ps(cos(3.1415926 * angLim / 180));
	const __m256 panMin = _mm256_set1_ps(1);
	const Vertex *pVert = &pt[0], *pNorm = &norm[0];
	for (uint32_t a = count; a--; ++pVert, ++pNorm)
	{
		if (*idxs > 65530)//fast skip
		{
			idxs++;
			*dists++ = 1e10f;
			continue;
		}
		//object vertex being searched
		const Vertex vObj = *pVert; const __m256 mObj = _mm256_broadcast_ps((__m128*)pVert), mNorm = _mm256_broadcast_ps((__m128*)pNorm);
		//find proper subtree
		const auto tid = judgeIdx(vObj);
		const auto& part = tree[tid];
		const float *__restrict pBase = part[0], *__restrict pNBase = ntree[tid][0], *__restrict pID = idtree[tid][0];
		//min dist * 8   AND   min idx * 8
		__m256 min8 = _mm256_set1_ps(1e10f), minpos8 = _mm256_setzero_ps();

		//use AVX which contains 8 float, hence 8 vertex was tested at one cycle
		for (uint32_t b = part.size() / 8; b--; pID += 8)
		{
			//calc angles(cos) between obj point and 8 base point
			const __m256 cos8 = _mm256_blend_ps
			(
				_mm256_blend_ps(
					_mm256_dp_ps(_mm256_load_ps(pNBase + 0 ), mNorm, 0x71)/*c1,0,0,0;c2,0,0,0*/,
					_mm256_dp_ps(_mm256_load_ps(pNBase + 8 ), mNorm, 0x72)/*0,c3,0,0;0,c4,0,0*/,
					0x22)/*c1,c3,00;c2,c4,00*/,
				_mm256_blend_ps(
					_mm256_dp_ps(_mm256_load_ps(pNBase + 16), mNorm, 0x74)/*0,0,c5,0;0,0,c6,0*/,
					_mm256_dp_ps(_mm256_load_ps(pNBase + 32), mNorm, 0x78)/*0,0,0,c7;0,0,0,c8*/,
					0x88)/*00,c5,c7;00,c6,c8*/,
				0b11001100
			)/*c1,c3,c5,c7;c2,c4,c6,c8*/;

			//calculate 4 vector represent 8 (point-Obj ---> point-Base)
			const __m256 a1 = _mm256_sub_ps(mObj, _mm256_load_ps(pBase + 0));
			const __m256 a2 = _mm256_sub_ps(mObj, _mm256_load_ps(pBase + 8));
			const __m256 a3 = _mm256_sub_ps(mObj, _mm256_load_ps(pBase + 16));
			const __m256 a4 = _mm256_sub_ps(mObj, _mm256_load_ps(pBase + 24));
			//prefetch
			_mm_prefetch((const char*)(pBase += 32), _MM_HINT_T1);
			_mm_prefetch((const char*)(pNBase += 32), _MM_HINT_T1);

			const __m256 cosRes = _mm256_cmp_ps(cos8, mincos, _CMP_GE_OS);
			if (!_mm256_movemask_ps(cosRes))//angle all unsatisfied
				continue;

			//for those angles smaller than lim-angle, the penalty will be less than 1(which means decrease of distance)
			//hence use max to make sure that all penalties are at least zero
			const __m256 pan8 = _mm256_max_ps(_mm256_div_ps(limcos, cos8), panMin);//those need to be panish

			//make up vector contain 8 dist data(dist^2)
			const __m256 the8 = _mm256_blend_ps
			(
				_mm256_blend_ps(_mm256_dp_ps(a1, a1, 0x71)/*d1,000;d2,000*/, _mm256_dp_ps(a2, a2, 0x72)/*0,d3,00;0,d4,00*/, 0x22)/*d1,d3,00;d2,d4,00*/,
				_mm256_blend_ps(_mm256_dp_ps(a3, a3, 0x74)/*00,d5,0;00,d6,0*/, _mm256_dp_ps(a4, a4, 0x78)/*000,d7;000,d8*/, 0x88)/*00,d5,d7;00,d6,d8*/,
				0b11001100
			)/*d1,d3,d5,d7;d2,d4,d6,d8*/;
			//const __m256 this8 = _mm256_blendv_ps(the8, _mm256_mul_ps(the8, pan8), panMask);
			const __m256 this8 = _mm256_mul_ps(the8, pan8);

			//find out which idx need to be updated(less than current and satisfy angle-requierment)
			const __m256 mask = _mm256_and_ps(_mm256_cmp_ps(this8, min8, _CMP_LT_OS), cosRes);
			min8 = _mm256_blendv_ps(min8, this8, mask);

			//if it neccessary to update idx
			if (_mm256_movemask_ps(mask))
			{
				//load idx data and refresh min idx
				minpos8 = _mm256_blendv_ps(minpos8, _mm256_load_ps(pID), mask);
			}
		}
		// after uprolled search, need to extract min dist&idx among 8 data
		// find out whether each dist is the min among 8,
		// consider they could be all the same so "less OR equal" should be used
		const __m256 com1 = _mm256_cmp_ps(min8, _mm256_permute_ps(min8, _MM_SHUFFLE(0, 3, 2, 1)), _CMP_LE_OS)/*a<=b,b<=c,c<=d,d<=a*/;
		const __m256 com2 = _mm256_cmp_ps(min8, _mm256_permute_ps(min8, _MM_SHUFFLE(1, 0, 3, 2)), _CMP_LE_OS)/*a<=c,b<=d,c<=a,d<=b*/;
		const __m256 com3 = _mm256_cmp_ps(min8, _mm256_permute_ps(min8, _MM_SHUFFLE(2, 1, 0, 3)), _CMP_LE_OS)/*a<=d,b<=a,c<=b,d<=c*/;
		const int res = _mm256_movemask_ps(_mm256_and_ps(_mm256_and_ps(com1, com2), com3))/*1357;2468*/;
		// final result may contain multi "1"(means min) when some dists are the same
		// so need to use bit scan to find a pos(whatever pos)
	#if defined(__GNUC__)
		const int idx0 = _bit_scan_forward(res), idx1 = _bit_scan_reverse(res);
	#else
		unsigned long idx0, idx1;
		_BitScanForward(&idx0, res); _BitScanReverse(&idx1, res);
	#endif
		// neccessary to copy data out from __m128, 
		// or you may not be able to get the true result from mem,
		// since they may still be in register
		int ALIGN32 theIDX[8]; float ALIGN32 theDIST[8];
		_mm256_store_ps((float*)theIDX, minpos8); _mm256_store_ps(theDIST, min8);
		int IDX; float DIST;
		if (theDIST[idx0] <= theDIST[idx1])
		{
			IDX = theIDX[idx0]; DIST = theDIST[idx0];
		}
		else
		{
			IDX = theIDX[idx1]; DIST = theDIST[idx1];
		}
		*dists = DIST;
		if (DIST > MAXDist2)
			*idxs = 65536;
		else
		{
			*idxs = IDX;
			ret.mthcnts[IDX]++;
			DIST *= d2threshold;
			if (ret.mdists[IDX] > DIST)
				ret.mdists[IDX] = DIST;
		}
		idxs++; dists++;
	}
}


class xzNNTree : public NNTreeBase<xzNNTree>
{
protected:
	friend NNTreeBase;
	inline int judgeIdx(const Vertex& v) const
	{
	#ifdef USE_SSE4
		return _mm_movemask_ps(v) & 0b0101;
	#else
		const uint8_t idx = ((v.int_x >> 31) & 0b1) /*+ ((v.int_y >> 30) & 0b10) */ + ((v.int_z >> 29) & 0b100);
		return idx;
	#endif
	}
};

class NNTree : public NNTreeBase<NNTree>
{
protected:
	friend NNTreeBase;
	inline int judgeIdx(const Vertex& v) const
	{
		return 0;
	}
};

/* h3NNTree is an enhanced tree which reduce unnecessary search by deviding the model to 3 parts.
 * assuming 0 is the average of the model's max height and min height, the model is devided into upper and lowwer parts
 * since there's may be some points linking to the other part, the points near 0 is collected as the middle part.
 * Let D be the difference of model's height, points whose height satisfied(abs(z)<=0.2D) is the middle,
 * z>0 is the upper and z<0 is the lowwer.
 * While searching, object points whose abs(z)<=0.1D are searched in middle part, z>0.1D:upper part, z<-0.1D:lowwer part
 **/
class h3NNTree : public NNTreeBase<h3NNTree>
{
protected:
	friend NNTreeBase;
	//0.1D and -0.1D
	float hup, hlow;
	inline int judgeIdx(const Vertex& v) const
	{
		if (v.z >= hup)
			return 1;
		else if (v.z <= hlow)
			return 2;
		else
			return 0;
	}
public:
	void init(const VertexVec& points, const VertexVec& normals, const uint32_t nPoints)
	{
		for (auto& t : tree)
			t.clear();
		for (auto& t : ntree)
			t.clear();
		tree[0].reserve(nPoints / 4); ntree[0].reserve(nPoints / 4);
		tree[1].reserve(nPoints * 2 / 3); ntree[1].reserve(nPoints * 2 / 3);
		tree[2].reserve(nPoints * 2 / 3); ntree[2].reserve(nPoints * 2 / 3);
		float limup, limlow;
		//calculate the difference of height first
		{
			__m128 minV = _mm_set_ps1(1e10f), maxV = _mm_set_ps1(-1e10f);
			for (uint32_t i = 0; i < nPoints; ++i)
			{
				minV = _mm_min_ps(minV, points[i]);
				maxV = _mm_max_ps(maxV, points[i]);
			}
			const Vertex diff(_mm_sub_ps(maxV, minV));
			limup = diff.z * 0.2f; limlow = -limup;
			hup = diff.z * 0.1f; hlow = -hup;
		}
		for (uint32_t i = 0; i < nPoints; ++i)
		{
			Vertex v = points[i]; v.int_w = i;
			uint8_t tid = v.z >= 0 ? 1 : 2;
			tree[tid].push_back(v);
			ntree[tid].push_back(normals[i]);
			if (v.z > limlow && v.z < limup)
			{
				tree[0].push_back(v);
				ntree[0].push_back(normals[i]);
			}
		}
		
		ptCount = nPoints;
		static const Vertex empty(1e10f, 1e10f, 1e10f);
		static const Vertex emptyN(0, 0, 0);
		for (uint32_t a = 0, s = nPoints / 8; a < 3; ++a)
		{
			auto& pts = tree[a];
			for (uint32_t b = pts.size() % minbase; b != 0 && b < minbase; ++b)
			{
				pts.push_back(empty);
				ntree[a].push_back(emptyN);
			}
			const uint32_t idsize = pts.size() / 4;
			auto& ids = idtree[a];
			ids.resize(idsize);
			for (uint32_t b = 0, c = 0; b < idsize; b += 2, c += 8)
			{
				//calculate 4 vector represent 8 (point-Obj ---> point-Base)
				const __m256 vb01 = _mm256_load_ps(pts[c + 0]), vb23 = _mm256_load_ps(pts[c + 2]),
					vb45 = _mm256_load_ps(pts[c + 4]), vb67 = _mm256_load_ps(pts[c + 6]);
				//make up vector contain 4 new idx from point-base's extra idx data
				const __m256 id8 = _mm256_shuffle_ps(_mm256_unpackhi_ps(vb01, vb23)/*xx,i1,i3;xx,i2,i4*/,
					_mm256_unpackhi_ps(vb45, vb67)/*xx,i5,i7;xx,i6,i8*/, 0b11101110)/*i1,i3,i5,i7;i2,i4,i6,i8*/;
				_mm256_store_ps(ids[b], id8);
			}
		}
	}
};


}