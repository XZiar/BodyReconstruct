#include "kdNNTree.h"

namespace miniBLAS
{
using std::vector;

void kdNNTree::init(const arma::mat& points)
{
	for (auto& t : tree)
	{
		t.clear();
		t.reserve(points.n_rows / 8);
	}
	
	const double *__restrict px = points.memptr(), *__restrict py = px + points.n_rows, *__restrict pz = py + points.n_rows;
	for (uint32_t i = 0; i < points.n_rows; ++i)
	{
		const Vertex v(*px++, *py++, *pz++, (int32_t)i);
		tree[judgeIdx(v)].push_back(v);
	}

	const Vertex empty(1e10f, 1e10f, 1e10f);
	static const uint32_t minbase = 8;
	for (auto& t : tree)
	{
		for (uint32_t a = t.size() % minbase; a != 0 && a < minbase; ++a)
			t.push_back(empty);
	}
}

void kdNNTree::searchBasic(const Vertex *__restrict pVert, const uint32_t count, int *idxs, float *dists) const
{
	const uint32_t cntV = (count + 3) / 4;
	Vertex *tmp = new Vertex[cntV];
	float *pFtmp = tmp[0];
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
		*pFtmp++ = minval;
	}
	Vertex *pTmp = tmp;
	for (uint32_t a = 0; a < cntV; ++a, ++pTmp)
		(*pTmp).do_sqrt();
	memcpy(dists, tmp, count * sizeof(float));
	delete[] tmp;
}

void kdNNTree::searchOld(const Vertex* pVert, const uint32_t count, int *__restrict idxs, float *__restrict dists) const
{
#ifdef USE_SSE4
	const uint32_t cntV = (count + 3) / 4;
	Vertex *tmp = new Vertex[cntV];
	float *pFtmp = tmp[0];
	for (uint32_t a = count; a--; ++pVert)
	{
		//object vertex being searched
		const Vertex vObj = *pVert; const __m128 mObj = vObj;
		//find proper subtree
		const auto& part = tree[judgeIdx(vObj)];
		const Vertex *__restrict pBase = &part[0];
		//min dist * 4   AND   min idx * 4
		__m128 min4 = _mm_set_ps1(1e10f); __m128 minpos4 = _mm_setzero_ps();

		for (uint32_t b = part.size() / 4; b--; )
		{
			//calculate 4 vector represent point-Obj ---> point-Base
			const __m128 vb1 = pBase[0], a1 = _mm_sub_ps(mObj, vb1);
			const __m128 vb2 = pBase[1], a2 = _mm_sub_ps(mObj, vb2);
			const __m128 vb3 = pBase[2], a3 = _mm_sub_ps(mObj, vb3);
			const __m128 vb4 = pBase[3], a4 = _mm_sub_ps(mObj, vb4);

			//prefetch
			_mm_prefetch((const char*)(pBase += 4), _MM_HINT_T1);

			//make up vector contain 4 dist data(dist^2)
			__m128 this4 = _mm_blend_ps
			(
				_mm_movelh_ps(_mm_dp_ps(a1, a1, 0b01110001)/*i1,0,0,0*/, _mm_dp_ps(a3, a3, 0b01110001)/*i3,0,0,0*/)/*i1,0,i3,0*/,
				_mm_movelh_ps(_mm_dp_ps(a2, a2, 0b01110010)/*0,i2,0,0*/, _mm_dp_ps(a4, a4, 0b01110010)/*0,i4,0,0*/)/*0,i2,0,i4*/,
				0b1010
			)/*i1,i2,i3,i4*/;
			//below is an old implement(slower due to less preferable of unpack)
			//__m128 this4 = _mm_unpacklo_ps
			//(
			//	_mm_insert_ps(_mm_dp_ps(a1, a1, 0b01110001)/*i1,0,0,0*/, _mm_dp_ps(a3, a3, 0b01110010)/*0,i3,0,0*/, 0b01011100)/*i1,i3,0,0*/,
			//	_mm_insert_ps(_mm_dp_ps(a2, a2, 0b01110001)/*i2,0,0,0*/, _mm_dp_ps(a4, a4, 0b01110010)/*0,i4,0,0*/, 0b01011100)/*i2,i4,0,0*/
			//)/*i1,i2,i3,i4*/;

			//find out which idx need to be updated and refresh min dist
			const __m128 mask = _mm_cmplt_ps(this4, min4);
			min4 = _mm_min_ps(min4, this4);

			//if it neccessary to update idx
			if (_mm_movemask_ps(mask))
			{
				//make up vector contain 4 new idx from point-base's extra idx data
				this4 = _mm_shuffle_ps(_mm_unpackhi_ps(vb1, vb2)/*x,x,i1,i2*/, _mm_unpackhi_ps(vb3, vb4)/*x,x,i3,i4*/, 0b11101110)/*i1,i2,i3,i4*/;

				//refresh min idx
				minpos4 = _mm_blendv_ps(minpos4, this4, mask);
			}
		}
		//after uprolled search, need to extract min dist&idx among 4 data
		{
			//float tmpdat[4]; _mm_storeu_ps(tmpdat, min4);
			//printf("min4 = %e,%e,%e,%e\n", tmpdat[0], tmpdat[1], tmpdat[2], tmpdat[3]);
			// find out whether each dist is the min among 4,
			// consider they could be all the same so "less OR equal" should be used
			const __m128 com1 = _mm_cmple_ps(min4, _mm_shuffle_ps(min4, min4, _MM_SHUFFLE(0, 3, 2, 1)) )/*a<=b,b<=c,c<=d,d<=a*/;
			const __m128 com2 = _mm_cmple_ps(min4, _mm_shuffle_ps(min4, min4, _MM_SHUFFLE(1, 0, 3, 2)) )/*a<=c,b<=d,c<=a,d<=b*/;
			const __m128 com3 = _mm_cmple_ps(min4, _mm_shuffle_ps(min4, min4, _MM_SHUFFLE(2, 1, 0, 3)) )/*a<=d,b<=a,c<=b,d<=c*/;
			const int res = _mm_movemask_ps(_mm_and_ps(_mm_and_ps(com1, com2), com3));
			// final result may contain multi "1"(means min) when some dists are the same
			// so need to use bit scan to find a pos(whatever pos)
		#if defined(__GNUC__)
			const int idx = _bit_scan_forward(res);
		#else
			unsigned long idx;
			_BitScanForward(&idx, res);
		#endif
			// neccessary to copy data out from __m128, 
			// or you may not be able to get the true result from mem,
			// since they may still be in register
			const VertexI vIdx(_mm_castps_si128(minpos4));
			const Vertex vDist(min4);
			*idxs = vIdx[idx];
			*pFtmp = vDist[idx];
			//printf("res:%d,idx:%d; ==== DIST %e; IDX %d\n", res, idx, *pFtmp, *idxs);
			idxs++; pFtmp++;
		}
	}
	Vertex *pTmp = tmp;
	for (uint32_t a = cntV; a--; ++pTmp)
		(*pTmp).do_sqrt();
	memcpy(dists, tmp, count * sizeof(float));
	delete[] tmp;
#else
	search(pVert, count, idxs, dists);
#endif
}

void kdNNTree::search(const Vertex* pVert, const uint32_t count, int *__restrict idxs, float *__restrict dists) const
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

}