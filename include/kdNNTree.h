#pragma once

#include <cstdint>
#include <vector>
#include <armadillo>

#define USE_SSE
#define USE_SSE2
#define USE_SSE4

#include "miniBLAS.hpp"

namespace miniBLAS
{

class kdNNTree
{
private:
	miniBLAS::VertexVec tree[8];
	int judgeIdx(const Vertex& v) const;
public:
	kdNNTree() = default;
	~kdNNTree() = default;
	void init(const arma::mat& points);
	void searchBasic(const Vertex* pVert, const uint32_t count, int *idxs, float *dists) const;
	void searchOld(const Vertex* pVert, const uint32_t count, int *idxs, float *dists) const;
	void search(const Vertex* pVert, const uint32_t count, int *idxs, float *dists) const;
};


inline int kdNNTree::judgeIdx(const Vertex& v) const
{
#ifdef USE_SSE
	return _mm_movemask_ps(v) & 0b0101;
#else
	const uint8_t idx = ((v.int_x >> 31) & 0b1) /*+ ((v.int_y >> 30) & 0b10) */ + ((v.int_z >> 29) & 0b100);
	return idx;
#endif
}

}