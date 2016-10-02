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
	std::vector<Vertex> tree[8];
	uint8_t judgeIdx(const Vertex& v) const;
public:
	kdNNTree() = default;
	~kdNNTree() = default;
	void init(const arma::mat& points);
	void searchBasic(const Vertex* pVert, const uint32_t count, int *idxs, float *dists) const;
	void search(const Vertex* pVert, const uint32_t count, int *idxs, float *dists) const;
};


inline uint8_t kdNNTree::judgeIdx(const miniBLAS::Vertex& v) const
{
	const uint8_t idx = ((*(int*)&v.x & 0x80000000) >> 29) /*+ ((*(int*)&v.y & 0x80000000) >> 30)*/ + ((*(int*)&v.z & 0x80000000) >> 31);
	return idx;
}

}