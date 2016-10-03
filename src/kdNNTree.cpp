#include "kdNNTree.h"

namespace miniBLAS
{
using std::vector;
using std::move;

void kdNNTree::init(const arma::mat& points)
{
	const uint32_t cnt = points.n_rows;
	for (auto& t : tree)
	{
		t.clear();
		t.reserve(cnt / 8);
	}
	
	const double *__restrict px = points.memptr(), *__restrict py = px + cnt, *__restrict pz = py + cnt;
	for (uint32_t i = 0; i < cnt; ++i)
	{
		const Vertex v(*px++, *py++, *pz++, (int32_t)i);
		tree[judgeIdx(v)].push_back(v);
	}

	const Vertex empty(1e10f, 1e10f, 1e10f);
	for (auto& t : tree)
	{
		for (uint32_t a = t.size() % 8; a != 0 && a < 8; ++a)
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

void kdNNTree::search(const Vertex* pVert, const uint32_t count, int *__restrict idxs, float *__restrict dists) const
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
		__m128 min4 = _mm_set_ps1(1e10f); __m128i minpos4 = _mm_setzero_si128();

		for (uint32_t b = part.size() / 4; b--; )
		{
			//calculate 4 vector represent point-Obj ---> point-Base
			const Vertex vb1 = _mm_load_ps((const float*)(pBase++)); const __m128 a1 = _mm_sub_ps(mObj, vb1);
			const Vertex vb2 = _mm_load_ps((const float*)(pBase++)); const __m128 a2 = _mm_sub_ps(mObj, vb2);
			const Vertex vb3 = _mm_load_ps((const float*)(pBase++)); const __m128 a3 = _mm_sub_ps(mObj, vb3);
			const Vertex vb4 = _mm_load_ps((const float*)(pBase++)); const __m128 a4 = _mm_sub_ps(mObj, vb4);
			
			//make up vector contain 4 dist data(dist^2)
			const __m128 this4 = _mm_unpacklo_ps
			(
				_mm_insert_ps(
					_mm_dp_ps(a1, a1, 0b01110001)/*i1,0,0,0*/, _mm_dp_ps(a3, a3, 0b01110010)/*0,i3,0,0*/,
					0b01011100)/*i1,i3,0,0*/,
				_mm_insert_ps(
					_mm_dp_ps(a2, a2, 0b01110001)/*i2,0,0,0*/, _mm_dp_ps(a4, a4, 0b01110010)/*0,i4,0,0*/,
					0b01011100)/*i2,i4,0,0*/
			);/*i1,i2,i3,i4*/

			//find out which idx need to be updated and refresh min dist
			const __m128 mask = _mm_cmplt_ps(this4, min4);
			min4 = _mm_min_ps(min4, this4);
			
			//if it neccessary to update idx
			if(_mm_movemask_ps(mask))
			{
				//make up vector contain 4 new idx from point-base's extra idx data
				const __m128 newpos = _mm_shuffle_ps(
					_mm_unpackhi_ps(vb1, vb2)/*x,x,i1,i2*/,
					_mm_unpackhi_ps(vb3, vb4)/*x,x,i3,i4*/,
					0b11101110);/*i1,i2,i3,i4*/
				//refresh min idx
				minpos4 = _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(minpos4), newpos, mask));
			}
		}
		//after uprolled search, need to extract min dist&idx among 4 data
		{
			// find out whether each dist is the min among 4,
			// consider they could be all the same so "less OR equal" should be used
			const __m128 com1 = _mm_cmple_ps(min4, _mm_shuffle_ps(min4, min4, _MM_SHUFFLE(1, 2, 3, 0))
			);/*a<=b,b<=c,c<=d,d<=a*/
			const __m128 com2 = _mm_cmple_ps(min4, _mm_shuffle_ps(min4, min4, _MM_SHUFFLE(2, 3, 0, 1))
			);/*a<=c,b<=d,c<=a,d<=b*/
			const __m128 com3 = _mm_cmple_ps(min4, _mm_shuffle_ps(min4, min4, _MM_SHUFFLE(3, 0, 1, 2))
			);/*a<=d,b<=a,c<=b,d<=c*/
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
			const VertexI vIdx(minpos4);
			const Vertex vDist(min4);
			*idxs++ = vIdx[idx];
			*pFtmp++ = vDist[idx];
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

}