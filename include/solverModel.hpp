#pragma once
/*
 * Use CRTP to save some code, for better maintainance
 * Solver-models for pose and shape are quite similar.
 * At most time, the only differecnce is that pose-solving
 * could use cache data of shape2mesh to speed up.
 * Since they will be called millions of times while solving,
 * CRTP could save the overhead of virtual-function
*/
#include "fitMesh.h"

using miniBLAS::Vertex;
using miniBLAS::VertexVec;

static atomic_uint64_t runcnt(0), runtime(0), selftime(0);

/*basic clculation, Point to Point distance only*/
template<typename CHILD>
struct NormalCostFunctor
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose;
	const std::vector<int8_t> isValid;
	const VertexVec& scanCache;
	NormalCostFunctor(const CShapePose *shapepose_, const std::vector<int8_t>& isValid_, const miniBLAS::VertexVec& scanCache_)
		: shapepose(shapepose_), isValid(isValid_), scanCache(scanCache_)
	{
	}
public:
	bool operator()(const double* param, double* residual) const
	{
		SimpleTimer timer;
		const VertexVec pts = ((const CHILD&)(*this)).getPts(param);
		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
		{
			if (isValid[j])
				residual[j] = (scanCache[j] - pts[j]).length();
			else
				residual[j] = 0;
		}
		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	}
};
struct PoseCostFunctor : public NormalCostFunctor<PoseCostFunctor>
{
protected:
	const CMesh baseMesh;
public:
	inline VertexVec getPts(const double* pose) const
	{
		return shapepose->getModelByPose(baseMesh, pose);
	}
	PoseCostFunctor(const CShapePose *shapepose_, const ModelParam& modelParam, const std::vector<int8_t>& isValid_, const miniBLAS::VertexVec& scanCache_)
		: NormalCostFunctor<PoseCostFunctor>(shapepose_, isValid_, scanCache_), baseMesh(shapepose_->getBaseModel(modelParam.shape.data()))
	{
	}
};
struct ShapeCostFunctor : public NormalCostFunctor<ShapeCostFunctor>
{
protected:
	const ModelParam::PoseParam pose;
public:
	inline VertexVec getPts(const double* shape) const
	{
		return shapepose->getModelFast(shape, pose.data());
	}
	ShapeCostFunctor(const CShapePose *shapepose_, const ModelParam& modelParam, const std::vector<int8_t>& isValid_, const miniBLAS::VertexVec& scanCache_)
		: NormalCostFunctor<ShapeCostFunctor>(shapepose_, isValid_, scanCache_), pose(modelParam.pose)
	{
	}
};

/*simple optimization, Point to Point distance only. Use validMask to accelerate*/
template<typename CHILD>
struct EarlyCutCostFunctor
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose;
	const std::vector<float> weights;
	const VertexVec& scanCache;
	uint32_t count;
	EarlyCutCostFunctor(const CShapePose *shapepose_, const std::vector<float>& weights_, const miniBLAS::VertexVec& scanCache_)
		: shapepose(shapepose_), weights(weights_), scanCache(scanCache_), count(scanCache_.size())
	{
	}
	EarlyCutCostFunctor(const CShapePose *shapepose_, const miniBLAS::VertexVec& scanCache_)
		: shapepose(shapepose_), weights(std::vector<float>(scanCache_.size(), 1.0f)), scanCache(scanCache_), count(scanCache_.size())
	{
	}
public:
	bool operator()(const double* param, double* residual) const
	{
		SimpleTimer timer;
		const VertexVec pts = ((const CHILD&)(*this)).getPts(param);
		for (uint32_t j = 0; j < count; ++j)
		{
			residual[j] = (scanCache[j] - pts[j]).length() * weights[j];
		}
		memset(&residual[count], 0, sizeof(double) * (EVALUATE_POINTS_NUM - count));
		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	}
};
struct PoseCostEC : public EarlyCutCostFunctor<PoseCostEC>
{
protected:
	const CMesh baseMesh;
public:
	inline VertexVec getPts(const double* pose) const
	{
		return shapepose->getModelByPose2(baseMesh, pose);
	}
	PoseCostEC(const CShapePose *shapepose_, const ModelParam& modelParam, const std::vector<int8_t>& isValid_, const miniBLAS::VertexVec& scanCache_)
		: EarlyCutCostFunctor<PoseCostEC>(shapepose_, scanCache_), 
		baseMesh(shapepose_->getBaseModel2(modelParam.shape.data(), isValid_.data()))
	{
	}
	PoseCostEC(const CShapePose *shapepose_, const ModelParam& modelParam, const std::vector<int8_t>& isValid_, const miniBLAS::VertexVec& scanCache_,
		const std::vector<float>& weights_)
		: EarlyCutCostFunctor<PoseCostEC>(shapepose_, weights_, scanCache_),
		baseMesh(shapepose_->getBaseModel2(modelParam.shape.data(), isValid_.data()))
	{
	}
};
struct ShapeCostEC : public EarlyCutCostFunctor<ShapeCostEC>
{
protected:
	const ModelParam::PoseParam pose;
	const PtrModSmooth mSmooth;
	const std::vector<int8_t> isValid;
public:
	inline VertexVec getPts(const double* shape) const
	{
		return shapepose->getModelFast2(mSmooth, shape, pose.data(), isValid.data());
	}
	ShapeCostEC(const CShapePose *shapepose_, const ModelParam& modelParam, const std::vector<int8_t>& isValid_, const miniBLAS::VertexVec& scanCache_)
		: EarlyCutCostFunctor<ShapeCostEC>(shapepose_, scanCache_), pose(modelParam.pose),
		mSmooth(shapepose_->preCompute(isValid_.data())), isValid(isValid_)
	{
	}
};


/*simple optimization, Point to TangentSurface distance with penalty. Use validMask to accelerate*/
template<typename CHILD>
struct EarlyCutP2SCostFunctor
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose;
	VertexVec scanCache;
	const VertexVec normCache;
	uint32_t count;
	EarlyCutP2SCostFunctor(const CShapePose *shapepose_, const std::vector<float>& weights_, 
		const miniBLAS::VertexVec& scanCache_, const VertexVec& normCache_)
		: shapepose(shapepose_), scanCache(scanCache_), normCache(normCache_), count(scanCache_.size())
	{
		for (uint32_t idx = 0; idx < count; ++idx)
			scanCache[idx].w = weights_[idx];
	}
	EarlyCutP2SCostFunctor(const CShapePose *shapepose_, const miniBLAS::VertexVec& scanCache_, const VertexVec& normCache_)
		: shapepose(shapepose_), scanCache(scanCache_), normCache(normCache_), count(scanCache_.size())
	{
		for (uint32_t idx = 0; idx < count; ++idx)
			scanCache[idx].w = 1.0f;
	}
public:
	bool operator()(const double* param, double* residual) const
	{
		SimpleTimer timer;
		const VertexVec pts = ((const CHILD&)(*this)).getPts(param);
		const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(INT32_MAX));
		for (uint32_t j = 0; j < count; ++j)
		{
			/*len2 is the absolute value of the square of the distance between point and the surface*/
			const __m128 len2 = _mm_and_ps(_mm_dp_ps(pts[j] - scanCache[j], normCache[j], 0x7f), absmask);
			/*penalty(weight) is stored in scanCache's w, so multiply the len with it*/
			const __m128 vcurlen = _mm_mul_ps(_mm_sqrt_ps(len2)/*l,l,l,l*/, _mm_permute_ps(scanCache[j], 0xff)/*w,w,w,w*/);
			/*cast float to double, SSE instrin may be faster*/
			residual[j] = _mm_cvtsd_f64(_mm_cvtps_pd(vcurlen));
		}
		memset(&residual[count], 0, sizeof(double) * (EVALUATE_POINTS_NUM - count));
		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	}
};
struct PoseCostP2SEC : public EarlyCutP2SCostFunctor<PoseCostP2SEC>
{
protected:
	const CMesh baseMesh;
public:
	inline VertexVec getPts(const double* pose) const
	{
		return shapepose->getModelByPose2(baseMesh, pose);
	}
	PoseCostP2SEC(const CShapePose *shapepose_, const ModelParam& modelParam, const std::vector<int8_t>& isValid_, const std::vector<float>& weights_,
		const miniBLAS::VertexVec& scanCache_, const VertexVec& normCache_)
		: EarlyCutP2SCostFunctor<PoseCostP2SEC>(shapepose_, weights_, scanCache_, normCache_),
		baseMesh(shapepose_->getBaseModel2(modelParam.shape.data(), isValid_.data()))
	{
	}
};
struct ShapeCostP2SEC : public EarlyCutP2SCostFunctor<ShapeCostP2SEC>
{
protected:
	const ModelParam::PoseParam pose;
	const PtrModSmooth mSmooth;
	const std::vector<int8_t> isValid;
public:
	inline VertexVec getPts(const double* shape) const
	{
		return shapepose->getModelFast2(mSmooth, shape, pose.data(), isValid.data());
	}
	ShapeCostP2SEC(const CShapePose *shapepose_, const ModelParam& modelParam, const std::vector<int8_t>& isValid_,
		const miniBLAS::VertexVec& scanCache_, const VertexVec& normCache_)
		: EarlyCutP2SCostFunctor<ShapeCostP2SEC>(shapepose_, scanCache_, normCache_), pose(modelParam.pose),
		mSmooth(shapepose_->preCompute(isValid_.data())), isValid(isValid_)
	{
	}
};

/*Point to TangentSurface distance with penalty. unmatched points are calculated with predict points*/
struct PoseCostFunctorPred
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const VertexVec ptCache;
	const CMesh baseMesh;
	const std::vector<float> weights;

	static VertexVec buildCache(CShapePose *shapepose, const ModelParam& modelParam, const arColIS& isValidNN, const VertexVec& validScanCache_)
	{
		VertexVec baseVec = shapepose->getModelFast(modelParam.shape.data(), modelParam.pose.data());
		for (uint32_t a = 0, b = 0; a < EVALUATE_POINTS_NUM; ++a)
		{
			if (isValidNN[a])
				baseVec[a] = validScanCache_[b++];
		}
		return baseVec;
	}
	static std::vector<float> buildCache(const arColIS& isValidNN, const std::vector<float>& wgts = std::vector<float>())
	{
		std::vector<float> weights(EVALUATE_POINTS_NUM, 0.2f);
		const bool isWgt = (wgts.size() > 0);
		for (uint32_t a = 0, b = 0; a < EVALUATE_POINTS_NUM; ++a)
		{
			if (isValidNN[a])
				weights[a] = (isWgt ? wgts[b++] : 1);
		}
		return weights;
	}
public:
	PoseCostFunctorPred(CShapePose *shapepose, const ModelParam& modelParam, const ModelParam& preParam, const arColIS isValidNN, const miniBLAS::VertexVec& validScanCache,
		const std::vector<float>& wgts = std::vector<float>())
		: shapepose_(shapepose), weights(buildCache(isValidNN, wgts)),
		baseMesh(shapepose_->getBaseModel(modelParam.shape.data())), ptCache(buildCache(shapepose, preParam, isValidNN, validScanCache))
	{
	}
	bool operator()(const double *pose, double *residual) const
	{
		SimpleTimer timer;

		const auto pts = shapepose_->getModelByPose(baseMesh, pose);
		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
		{
			residual[j] = (ptCache[j] - pts[j]).length() * weights[j];
		}
		timer.Stop();
		runcnt++;
		runtime += uint32_t(timer.ElapseUs());
		return true;
	}
};


/*basic clculation of reverse search, Point to Point distance with penalty. idxMapper is needed since model-points may be less than object-points*/
template<typename CHILD>
struct ReverseCostFunctor
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose;
	const std::vector<uint32_t> idxMapper;
	VertexVec scanCache;
	const uint32_t count;
	ReverseCostFunctor(const CShapePose *shapepose_, const std::vector<uint32_t>& idxMapper_, const std::vector<float>& weights_, const VertexVec& scanCache_)
		: shapepose(shapepose_), idxMapper(idxMapper_), scanCache(scanCache_), count(scanCache_.size())
	{
		for (uint32_t idx = 0; idx < count; ++idx)
			scanCache[idx].w = weights_[idx] * weights_[idx];
	}
public:
	bool operator()(const double* param, double* residual) const
	{
		SimpleTimer timer, t2;
		const VertexVec pts = ((const CHILD&)(*this)).getPts(param);
		t2.Start();
		Vertex tmp[1 + EVALUATE_POINTS_NUM / 4]; float *__restrict pTmp = tmp[0];
		memset(pTmp, 0x0, sizeof(tmp));
		for (uint32_t j = 0; j < count; ++j)
		{
			const auto idx = idxMapper[j];
			pTmp[idx] += (scanCache[j] - pts[idx]).length_sqr() * scanCache[j].w;
		}
		{
			uint32_t i = 0, j = 0;
			for (; j < EVALUATE_POINTS_NUM / 4; ++j, i += 4)
				_mm256_storeu_pd(&residual[i], _mm256_cvtps_pd(_mm_sqrt_ps(tmp[j])));
			tmp[j].do_sqrt();
			for (; i < EVALUATE_POINTS_NUM; i++)
				residual[i] = pTmp[i];
		}
		timer.Stop(); t2.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		selftime += t2.ElapseNs();
		return true;
	}
};
struct PoseCostRe : public ReverseCostFunctor<PoseCostRe>
{
protected:
	const CMesh baseMesh;
public:
	inline VertexVec getPts(const double* pose) const
	{
		return shapepose->getModelByPose(baseMesh, pose);
	}
	PoseCostRe(const CShapePose *shapepose_, const ModelParam& modelParam, const std::vector<uint32_t>& idxMapper_, const std::vector<float>& weights_,
		const VertexVec& scanCache_) 
		: ReverseCostFunctor<PoseCostRe>(shapepose_, idxMapper_, weights_, scanCache_), baseMesh(shapepose_->getBaseModel(modelParam.shape.data()))
	{
	}
};
struct ShapeCostRe : public ReverseCostFunctor<ShapeCostRe>
{
protected:
	const ModelParam::PoseParam pose;
public:
	inline VertexVec getPts(const double* shape) const
	{
		return shapepose->getModelFast(shape, pose.data());
	}
	ShapeCostRe(const CShapePose *shapepose_, const ModelParam& modelParam, const std::vector<uint32_t>& idxMapper_, const std::vector<float>& weights_,
		const VertexVec& scanCache_) 
		: ReverseCostFunctor<ShapeCostRe>(shapepose_, idxMapper_, weights_, scanCache_), pose(modelParam.pose)
	{
	}
};


/*reverse search with shift, Point to Point distance with penalty. idxMapper is needed since model-points may be less than object-points
 *shift means that, among multi object-points with the same model-points, choose the shortest one
 **/
template<typename CHILD>
struct ReverseShiftCostFunctor
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose;
	std::vector<uint32_t> idxMapper;
	VertexVec scanCache;
	uint32_t count;
	ReverseShiftCostFunctor(const CShapePose *shapepose_, const std::vector<uint32_t>& idxMapper_, const std::vector<float>& weights_,
		const VertexVec& scanCache_)
		: shapepose(shapepose_), idxMapper(idxMapper_), scanCache(scanCache_), count(scanCache_.size())
	{
		for (uint32_t idx = 0; idx < count; ++idx)
			scanCache[idx].w = weights_[idx];
	}
	/*for predict version, predict points are inserted into cache here, with given weight*/
	ReverseShiftCostFunctor(const CShapePose *shapepose_, const std::vector<uint32_t>& idxMapper_, const std::vector<float>& weights_,
		const VertexVec& scanCache_, const ModelParam& predParam, const float fillwgt)
		: shapepose(shapepose_), idxMapper(idxMapper_)
	{
		int8_t flags[EVALUATE_POINTS_NUM] = { 0 };
		count = scanCache_.size();
		for (uint32_t a = 0; a < count; ++a)
		{
			scanCache.push_back(scanCache_[a]);
			scanCache.back().w = weights_[a];
			flags[idxMapper[a]] = 1;
		}
		const VertexVec fillpts = ((const CHILD&)(*this)).getPts(predParam);
		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
		{
			if (!flags[j])
			{
				scanCache.push_back(fillpts[j]);
				scanCache.back().w = fillwgt;
				idxMapper.push_back(j);
			}
		}
		count = scanCache.size();
	}
public:
	bool operator()(const double* param, double* residual) const
	{
		SimpleTimer timer, t2;
		const VertexVec pts = ((const CHILD&)(*this)).getPts(param);
		t2.Start();

		Vertex weight[EVALUATE_POINTS_NUM / 4 + 1], tmp[EVALUATE_POINTS_NUM / 4 + 1];
		memset(weight, 0x0, sizeof(weight)); memset(tmp, 0x7f, sizeof(tmp));
		float *__restrict finwgt = (float*)weight, *__restrict pTmp = (float*)tmp;
		const uint32_t *__restrict idx = idxMapper.data();
		for (const auto& obj : scanCache)
		{
			const float val = (obj - pts[*idx]).length_sqr();
			if (val < pTmp[*idx])
			{
				pTmp[*idx] = val;
				finwgt[*idx] = obj.w;
			}
			idx++;
		}
		{
			uint32_t i = 0, j = 0;
			for (; j < EVALUATE_POINTS_NUM / 4; ++j, i += 4)
			{
				const __m128 ans = _mm_mul_ps(weight[j], _mm_sqrt_ps(tmp[j]));
				_mm256_storeu_pd(&residual[i], _mm256_cvtps_pd(ans));
			}
			for (tmp[j].do_sqrt(); i < EVALUATE_POINTS_NUM; i++)
				residual[i] = finwgt[i] * pTmp[i];
		}
		timer.Stop(); t2.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		selftime += t2.ElapseNs();
		return true;
	}
};
struct PoseCostRShift : public ReverseShiftCostFunctor<PoseCostRShift>
{
protected:
	const CMesh baseMesh;
public:
	inline VertexVec getPts(const double* pose) const
	{
		return shapepose->getModelByPose(baseMesh, pose);
	}
	PoseCostRShift(const CShapePose *shapepose_, const ModelParam& modelParam, const std::vector<uint32_t>& idxMapper_, const std::vector<float>& weights_,
		const VertexVec& scanCache_)
		: ReverseShiftCostFunctor<PoseCostRShift>(shapepose_, idxMapper_, weights_, scanCache_),
		baseMesh(shapepose_->getBaseModel(modelParam.shape.data()))
	{
	}
};
struct ShapeCostRShift : public ReverseShiftCostFunctor<ShapeCostRShift>
{
protected:
	const ModelParam::PoseParam pose;
public:
	inline VertexVec getPts(const double* shape) const
	{
		return shapepose->getModelFast(shape, pose.data());
	}
	ShapeCostRShift(const CShapePose *shapepose_, const ModelParam& modelParam, const std::vector<uint32_t>& idxMapper_, const std::vector<float>& weights_,
		const VertexVec& scanCache_)
		: ReverseShiftCostFunctor<ShapeCostRShift>(shapepose_, idxMapper_, weights_, scanCache_), pose(modelParam.pose)
	{
	}
};
/*with predict*/
struct PoseCostRPredShift : public ReverseShiftCostFunctor<PoseCostRPredShift>
{
protected:
	const CMesh baseMesh;
public:
	inline VertexVec getPts(const double* pose) const
	{
		return shapepose->getModelByPose(baseMesh, pose);
	}
	inline VertexVec getPts(const ModelParam& predParam) const
	{
		return shapepose->getModelFast(predParam.shape.data(), predParam.pose.data());
	}
	PoseCostRPredShift(const CShapePose *shapepose_, const ModelParam& modelParam, const ModelParam& predParam, const float fillwgt,
		const std::vector<uint32_t>& idxMapper_, const std::vector<float>& weights_, const VertexVec& scanCache_)
		: ReverseShiftCostFunctor<PoseCostRPredShift>(shapepose_, idxMapper_, weights_, scanCache_, predParam, fillwgt),
		baseMesh(shapepose_->getBaseModel(modelParam.shape.data()))
	{
	}
};
struct ShapeCostRPredShift : public ReverseShiftCostFunctor<ShapeCostRPredShift>
{
protected:
	const ModelParam::PoseParam pose;
public:
	inline VertexVec getPts(const double* shape) const
	{
		return shapepose->getModelFast(shape, pose.data());
	}
	inline VertexVec getPts(const ModelParam& predParam) const
	{
		return shapepose->getModelFast(predParam.shape.data(), predParam.pose.data());
	}
	ShapeCostRPredShift(const CShapePose *shapepose_, const ModelParam& modelParam, const ModelParam& predParam, const float fillwgt,
		const std::vector<uint32_t>& idxMapper_, const std::vector<float>& weights_, const VertexVec& scanCache_)
		: ReverseShiftCostFunctor<ShapeCostRPredShift>(shapepose_, idxMapper_, weights_, scanCache_, predParam, fillwgt), pose(modelParam.pose)
	{
	}
};


/*reverse search, Point to TangentSurface distance with penalty. idxMapper is needed since model-points may be less than object-points*/
template<typename CHILD>
struct ReverseP2SCostFunctor
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose;
	std::vector<uint32_t> idxMapper;
	VertexVec scanCache;
	const VertexVec normCache;
	uint32_t count;
	ReverseP2SCostFunctor(const CShapePose *shapepose_, const std::vector<uint32_t>& idxMapper_, const std::vector<float>& weights_,
		const VertexVec& scanCache_, const VertexVec& normCache_)
		: shapepose(shapepose_), idxMapper(idxMapper_), scanCache(scanCache_), normCache(normCache_), count(scanCache_.size())
	{
		for (uint32_t idx = 0; idx < count; ++idx)
			scanCache[idx].w = weights_[idx] * weights_[idx];
	}
public:
	bool operator()(const double* param, double* residual) const
	{
		SimpleTimer timer, t2;
		const VertexVec pts = ((const CHILD&)(*this)).getPts(param);
		t2.Start();
		Vertex tmp[1 + EVALUATE_POINTS_NUM / 4]; float *__restrict pTmp = tmp[0];
		memset(pTmp, 0x0, sizeof(tmp));
		const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(INT32_MAX));
		for (uint32_t j = 0; j < count; ++j)
		{
			const auto idx = idxMapper[j];
			const __m128 vcurlen = _mm_mul_ps(
				_mm_dp_ps(pts[idx] - scanCache[j], normCache[j], 0x7f)/*l2,l2,l2,l2*/,
				_mm_permute_ps(scanCache[j], 0xff)/*w,w,w,w*/);
			pTmp[idx] += _mm_cvtss_f32(_mm_and_ps(vcurlen, absmask));
		}
		{
			uint32_t i = 0, j = 0;
			for (; j < EVALUATE_POINTS_NUM / 4; ++j, i += 4)
				_mm256_storeu_pd(&residual[i], _mm256_cvtps_pd(_mm_sqrt_ps(tmp[j])));
			tmp[j].do_sqrt();
			for (; i < EVALUATE_POINTS_NUM; i++)
				residual[i] = pTmp[i];
		}
		timer.Stop(); t2.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		selftime += t2.ElapseNs();
		return true;
	}
};
struct PoseCostP2SRe : public ReverseP2SCostFunctor<PoseCostP2SRe>
{
protected:
	const CMesh baseMesh;
public:
	inline VertexVec getPts(const double* pose) const
	{
		return shapepose->getModelByPose(baseMesh, pose);
	}
	PoseCostP2SRe(const CShapePose *shapepose_, const ModelParam& modelParam, const std::vector<uint32_t>& idxMapper_, const std::vector<float>& weights_,
		const VertexVec& scanCache_, const VertexVec& normCache_)
		: ReverseP2SCostFunctor<PoseCostP2SRe>(shapepose_, idxMapper_, weights_, scanCache_, normCache_),
		baseMesh(shapepose_->getBaseModel(modelParam.shape.data()))
	{
	}
};
struct ShapeCostP2SRe : public ReverseP2SCostFunctor<ShapeCostP2SRe>
{
protected:
	const ModelParam::PoseParam pose;
public:
	inline VertexVec getPts(const double* shape) const
	{
		return shapepose->getModelFast(shape, pose.data());
	}
	ShapeCostP2SRe(const CShapePose *shapepose_, const ModelParam& modelParam, const std::vector<uint32_t>& idxMapper_, const std::vector<float>& weights_,
		const VertexVec& scanCache_, const VertexVec& normCache_)
		: ReverseP2SCostFunctor<ShapeCostP2SRe>(shapepose_, idxMapper_, weights_, scanCache_, normCache_), pose(modelParam.pose)
	{
	}
};


/*constrain when solving pose.
 *Due to mono-camera, some parts of body may not be capture and no match link will be found.
 *That means whatever the joints param changes, residual won't change, which may cause big error.
 *This functor applys direct to the param, input weight is the total-error calculated during match-searching.
 *Though the input error may not be the same with costfunctor's result, they are in the same scale.
 *First 6 params are associate with total movement of model, which will not be considered.
 **/
struct MovementSofter
{
protected:
	const ModelParam::PoseParam lastpose;
	const double weight;
public:
	MovementSofter(const ModelParam& modelParam, const double err) :lastpose(modelParam.pose),
		weight(sqrt(err / (POSPARAM_NUM - 6))) { }
	bool operator()(const double* pose, double* residual) const
	{
		uint32_t i = 0;
		for (; i < 6; ++i)
			residual[i] = 0;
		for (; i < POSPARAM_NUM; ++i)
			residual[i] = weight * abs(pose[i] - lastpose[i]);
		return true;
	}
};
/*constrain when solving shape.
*Due to mono-camera, some parts of body may not be capture and no match link will be found.
*That means solving result may not be accurate and may differ a lot from the last result.
*This functor applys direct to the param, input weight is the total-error calculated during match-searching.
*Though the input error may not be the same with costfunctor's result, they are in the same scale.
**/
struct ShapeSofter
{
protected:
	const ModelParam::ShapeParam lastshape;
	const double weight;
public:
	ShapeSofter(const ModelParam& modelParam, const double err) :lastshape(modelParam.shape), weight(sqrt(err / SHAPEPARAM_NUM)) { }
	bool operator()(const double* shape, double* residual) const
	{
		for (uint32_t i = 0; i < SHAPEPARAM_NUM; ++i)
			residual[i] = weight * abs(shape[i] - lastshape[i]);
		return true;
	}
};

/*Function to calculate y = a+bx+cx^2*/
struct PNCalc
{
	template <typename T>
	static T calcute(const T* x, const uint32_t i)
	{
		const T val = T(i);
		return x[0] + x[1] * val + x[2] * ceres::pow(val, 2);
	}
};
/*Function to calculate y = a+bx+c*sin(d+x)*/
struct PSCalc
{
	template <typename T>
	static T calcute(const T* x, const uint32_t i)
	{
		const T val = T(i);
		return x[0] + x[1] * val + x[2] * ceres::sin(val + x[3]);
	}
};
/*Accept function and try to fit curve using that function's format
 *It directly applys to pose's NO.idx param.
 *There is no definate function to predict pose, so PNCalc which has x^2 may cause high error.
 *Hence joints usaully use PSCalc, whose sin may stop error.
 */
template<typename C>
struct ParamPredictor
{
protected:
	const std::vector<ModelParam>& params;
	const uint8_t idx;
public:
	using Calc = C;
	ParamPredictor(const std::vector<ModelParam>& params_, const uint8_t idx_) :params(params_), idx(idx_) { }
	template <typename T>
	bool operator ()(const T* const x, T* residual) const
	{
		const int32_t framesize = params.size();
		const int32_t maxcnt = calcCount(framesize);
		/*predict based on past frames, while further the frame, lesser impact on prediction.
		 *The way to control its impact is multiply difference with weight.
		 *Since ceres will try to eliminate every difference and this function is somehow too simple for it,
		 *I have to make the weight extremely low.
		 **/
		for (int32_t i = framesize - 1, j = 0; j < maxcnt; i--, j++)
		{
			const int32_t level = j / 2 * 3;//level increases every 2 frames, while every time it increase 3
			const T weight = ceres::pow(T(2), T(-level));//2^(-n)
			const T obj = C::calcute(x, i);
			residual[j] = weight * (T(params[i].pose[idx]) - obj);
		}
		return true;
	}
	static uint32_t calcCount(const uint32_t frames)
	{
		return std::min(frames, (uint32_t)14);
	}
};
template<typename C>
struct ParamSofter
{
protected:
	const std::vector<ModelParam>& params;
	const uint8_t idx;
	const int32_t objframe;
public:
	using Calc = C;
	ParamSofter(const std::vector<ModelParam>& params_, const uint32_t frame_, const uint8_t idx_) :params(params_), idx(idx_), objframe(frame_) { }
	template <typename T>
	bool operator ()(const T* const x, T* residual) const
	{
		const int32_t maxframe = std::min(int32_t(params.size() - 1), objframe + 8);
		const int32_t minframe = objframe >= 8 ? (objframe - 8) : 0;

		for (int32_t i = minframe, j = 0; i < maxframe; i++, j++)
		{
			const int32_t level = std::max((std::abs((int32_t)objframe - i) - 1), 0);//0,+/-1,frame are in the same level
			const T weight = ceres::pow(T(2), T(-level));//2^(-n)
			const T obj = C::calcute(x, i);
			residual[j] = weight * (T(params[i].pose[idx]) - obj);
		}
		return true;
	}
	static uint32_t calcCount(const uint32_t frames, const uint32_t obj)
	{
		const uint32_t maxframe = std::min(frames - 1, obj + 8);
		const uint32_t minframe = obj >= 8 ? (obj - 8) : 0;
		return maxframe - minframe;
	}
};
using PosePredictor = ParamPredictor<PNCalc>;
using PoseAniSofter = ParamSofter<PNCalc>;
using JointPredictor = ParamPredictor<PSCalc>;
using JointAniSofter = ParamSofter<PSCalc>;

