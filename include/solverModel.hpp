#pragma once

#include "fitMesh.h"

using miniBLAS::Vertex;
using miniBLAS::VertexVec;

static atomic_uint32_t runcnt(0), runtime(0);
static uint32_t nncnt = 0, nntime = 0;

//Definition of optimization functions
struct PoseCostFunctor
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const arColIS isValidNN_;
	const VertexVec& scanCache_;
	const VertexVec basePts;

public:
	PoseCostFunctor(CShapePose *shapepose, const ModelParam& modelParam, const arColIS& isValidNN, const miniBLAS::VertexVec& scanCache)
		: shapepose_(shapepose), isValidNN_(isValidNN), scanCache_(scanCache), basePts(shapepose_->getBaseModel(modelParam.shape.data()))
	{
	}
	bool operator()(const double* pose, double* residual) const
	{
		SimpleTimer timer;
		const auto pts = shapepose_->getModelByPose(basePts, pose);

		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
		{
			if (isValidNN_[j])
				residual[j] = (scanCache_[j] - pts[j]).length();
			else
				residual[j] = 0;
		}

		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	}
};
struct ShapeCostFunctor
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const ModelParam::PoseParam pose;
	const arColIS isValidNN_;
	const VertexVec& scanCache_;
public:
	ShapeCostFunctor(CShapePose *shapepose, const ModelParam& modelParam, const arColIS& isValidNN, const miniBLAS::VertexVec& scanCache)
		: shapepose_(shapepose), pose(modelParam.pose), isValidNN_(isValidNN), scanCache_(scanCache)
	{
	}
	bool operator()(const double* shape, double* residual) const
	{
		SimpleTimer timer;
		const auto pts = shapepose_->getModelFast(shape, pose.data());

		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
		{
			if (isValidNN_[j])
				residual[j] = (scanCache_[j] - pts[j]).length();
			else
				residual[j] = 0;
		}

		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	}
};

struct PoseCostFunctorEx2
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const arColIS isValidNN_;
	const VertexVec& validScanCache_;
	const CMesh baseMesh;
	const PtrModSmooth mSmooth;
	std::vector<float> weights;
public:
	PoseCostFunctorEx2(CShapePose *shapepose, const ModelParam& modelParam, const arColIS& isValidNN, const miniBLAS::VertexVec& validScanCache,
		PtrModSmooth mSmooth_, const std::vector<float>& wgts = std::vector<float>())
		: shapepose_(shapepose), isValidNN_(isValidNN), validScanCache_(validScanCache), mSmooth(mSmooth_),
		baseMesh(shapepose_->getBaseModel2(modelParam.shape.data(), isValidNN_.data()))
	{
		weights = wgts;
	}
	bool operator()(const double* pose, double* residual) const
	{
		const bool isWgt = (weights.size() > 0);
		SimpleTimer timer;

		const auto pts = shapepose_->getModelByPose2(mSmooth, baseMesh, pose, isValidNN_.data());

		const uint32_t cnt = validScanCache_.size();
		for (uint32_t i = 0, j = 0; j < cnt; ++j)
		{
			const Vertex delta = (validScanCache_[j] - pts[j]);
			residual[i++] = delta.length() * (isWgt ? weights[j] : 1);
		}
		memset(&residual[cnt], 0, sizeof(double) * (EVALUATE_POINTS_NUM - cnt));

		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	}
};

struct ShapeCostFunctorEx2
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const ModelParam::PoseParam pose;
	const arColIS isValidNN_;
	const VertexVec& validScanCache_;
	const PtrModSmooth mSmooth;
public:
	ShapeCostFunctorEx2(CShapePose *shapepose, const ModelParam& modelParam, const arColIS& isValidNN, const miniBLAS::VertexVec& validScanCache,
		PtrModSmooth mSmooth_)
		: shapepose_(shapepose), pose(modelParam.pose), isValidNN_(isValidNN), validScanCache_(validScanCache), mSmooth(mSmooth_)
	{
	}
	bool operator()(const double* shape, double* residual) const
	{
		SimpleTimer timer;

		const auto pts = shapepose_->getModelFast2(mSmooth, shape, pose.data(), isValidNN_.data());

		const uint32_t cnt = validScanCache_.size();
		for (uint32_t i = 0, j = 0; j < cnt; ++j)
		{
			residual[i++] = (validScanCache_[j] - pts[j]).length();
		}
		memset(&residual[cnt], 0, sizeof(double) * (EVALUATE_POINTS_NUM - cnt));

		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	}
};

struct PoseCostFunctorPred
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const VertexVec ptCache;
	const VertexVec basePt;
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
		basePt(shapepose_->getBaseModel(modelParam.shape.data())), ptCache(buildCache(shapepose, preParam, isValidNN, validScanCache))
	{
	}
	bool operator()(const double *pose, double *residual) const
	{
		SimpleTimer timer;

		const auto pts = shapepose_->getModelByPose(basePt, pose);
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

struct PoseCostFunctorRe
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	std::vector<float> weights;
	const std::vector<uint32_t> idxMapper;
	const uint32_t ptcount;
	const miniBLAS::VertexVec& scanCache;
	const VertexVec basePts;
public:
	PoseCostFunctorRe(CShapePose *shapepose, const ModelParam& modelParam, const miniBLAS::VertexVec& scanCache_,
		const std::vector<float>& weights_, const std::vector<uint32_t>& idxMapper_)
		: shapepose_(shapepose), idxMapper(idxMapper_), basePts(shapepose_->getBaseModel(modelParam.shape.data())),
		scanCache(scanCache_), ptcount(scanCache_.size())
	{
		weights.reserve(weights_.size());
		for (const auto w : weights_)
			weights.push_back(w*w);
	}
	bool operator() (const double* pose, double* residual) const
	{
		SimpleTimer timer;
		const auto pts = shapepose_->getModelByPose(basePts, pose);
		VertexVec tmp(1 + EVALUATE_POINTS_NUM / 4, Vertex(0, 0, 0, 0)); float *__restrict pTmp = tmp[0];
		//memset(pTmp, 0x0, (1 + EVALUATE_POINTS_NUM / 4) * sizeof(Vertex));
		for (uint32_t j = 0; j < ptcount; ++j)
		{
			const auto idx = idxMapper[j];
			pTmp[idx] += (scanCache[j] - pts[idx]).length_sqr() * weights[j];
		}
		for (auto& obj : tmp)
			obj.do_sqrt();
		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
			residual[j] = pTmp[j];
		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	}
};
struct ShapeCostFunctorRe
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const ModelParam::PoseParam pose;
	std::vector<float> weights;
	const std::vector<uint32_t> idxMapper;
	const uint32_t ptcount;
	const miniBLAS::VertexVec& scanCache;
public:
	ShapeCostFunctorRe(CShapePose *shapepose, const ModelParam& modelParam, const miniBLAS::VertexVec& scanCache_,
		const std::vector<float>& weights_, const std::vector<uint32_t>& idxMapper_)
		: shapepose_(shapepose), pose(modelParam.pose), idxMapper(idxMapper_), scanCache(scanCache_), ptcount(scanCache_.size())
	{
		weights.reserve(weights_.size());
		for (const auto w : weights_)
			weights.push_back(w*w);
	}
	bool operator() (const double* shape, double* residual) const
	{
		SimpleTimer timer;
		const auto pts = shapepose_->getModelFast(shape, pose.data());
		VertexVec tmp(1 + EVALUATE_POINTS_NUM / 4, Vertex(0, 0, 0, 0)); float *__restrict pTmp = tmp[0];
		//memset(pTmp, 0x0, (1 + EVALUATE_POINTS_NUM / 4) * sizeof(Vertex));
		for (uint32_t j = 0; j < ptcount; ++j)
		{
			const auto idx = idxMapper[j];
			pTmp[idx] += (scanCache[j] - pts[idx]).length_sqr() * weights[j];
		}
		for (auto& obj : tmp)
			obj.do_sqrt();
		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
			residual[j] = pTmp[j];

		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	}
};

static miniBLAS::VertexVec DataFilter(const miniBLAS::VertexVec& ptin, const std::vector<float>& weights)
{
	const uint32_t count = ptin.size();
	miniBLAS::VertexVec ptout;
	for (uint32_t a = 0; a < count; ++a)
	{
		ptout.push_back(ptin[a]);
		ptout.back().w = weights[a];
	}
	return ptout;
}
static miniBLAS::VertexVec DataFilter(const miniBLAS::VertexVec& ptin, const std::vector<float>& weights,
	const miniBLAS::VertexVec& fillpt, const float fillw, std::vector<uint32_t>& idxs)
{
	int8_t flags[EVALUATE_POINTS_NUM] = { 0 };
	const uint32_t count = ptin.size();
	miniBLAS::VertexVec ptout;
	for (uint32_t a = 0; a < count; ++a)
	{
		ptout.push_back(ptin[a]);
		ptout.back().w = weights[a];
		flags[idxs[a]] = 1;
	}
	for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
	{
		if (!flags[j])
		{
			ptout.push_back(fillpt[j]);
			ptout.back().w = fillw;
			idxs.push_back(j);
		}
	}
	return ptout;
}
struct PoseCostFunctorReShift
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const VertexVec basePts;
	std::vector<uint32_t> shiftParam;
	miniBLAS::VertexVec scanCache;
	uint32_t count;
public:
	PoseCostFunctorReShift(CShapePose *shapepose, const ModelParam& modelParam, const miniBLAS::VertexVec& scanCache_,
		const std::vector<float>& weights_, const std::vector<uint32_t>& idxMapper_)
		: shapepose_(shapepose), basePts(shapepose_->getBaseModel(modelParam.shape.data())), shiftParam(idxMapper_),
		scanCache(DataFilter(scanCache_, weights_)), count(idxMapper_.size())
	{
	}
	bool operator() (const double* pose, double* residual) const
	{
		SimpleTimer timer;
		const auto pts = shapepose_->getModelByPose(basePts, pose);

		float finwgt[EVALUATE_POINTS_NUM] = { 0 };
		memset(residual, 0x7f, sizeof(double) * EVALUATE_POINTS_NUM);
		uint32_t i = 0;
		for (const auto& obj : scanCache)
		{
			const uint32_t idx = shiftParam[i++];
			const double val = (obj - pts[idx]).length();
			if (val < residual[idx])
			{
				residual[idx] = val;
				finwgt[idx] = obj.w;
			}
		}
		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
			residual[j] *= finwgt[j];

		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	};
};
struct ShapeCostFunctorReShift
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const ModelParam::PoseParam pose;
	std::vector<uint32_t> shiftParam;
	miniBLAS::VertexVec scanCache;
	uint32_t count;
public:
	ShapeCostFunctorReShift(CShapePose *shapepose, const ModelParam& modelParam, const miniBLAS::VertexVec& scanCache_,
		const std::vector<float>& weights_, const std::vector<uint32_t>& idxMapper_)
		: shapepose_(shapepose), pose(modelParam.pose), shiftParam(idxMapper_), scanCache(DataFilter(scanCache_, weights_)), count(idxMapper_.size())
	{
	}
	bool operator() (const double* shape, double* residual) const
	{
		SimpleTimer timer;
		const auto pts = shapepose_->getModelFast(shape, pose.data());

		float finwgt[EVALUATE_POINTS_NUM] = { 0 };
		memset(residual, 0x7f, sizeof(double) * EVALUATE_POINTS_NUM);
		uint32_t i = 0;
		for (const auto& obj : scanCache)
		{
			const uint32_t idx = shiftParam[i++];
			const double val = (obj - pts[idx]).length();
			if (val < residual[idx])
			{
				residual[idx] = val;
				finwgt[idx] = obj.w;
			}
		}
		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
			residual[j] *= finwgt[j];
		
		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	}
};

struct PoseCostFunctorPredReShift
{
protected:
	const CShapePose *shapepose_;
	const VertexVec basePts;
	miniBLAS::VertexVec scanCache;
	std::vector<uint32_t> idxMapper;
	uint32_t ptcount;
public:
	PoseCostFunctorPredReShift(CShapePose *shapepose, const ModelParam& modelParam, const ModelParam& predParam, const miniBLAS::VertexVec& scanCache_,
		const std::vector<float>& weights_, const std::vector<uint32_t>& idxMapper_)
		: shapepose_(shapepose), basePts(shapepose_->getBaseModel(modelParam.shape.data())), idxMapper(idxMapper_)
	{
		const VertexVec predVec = shapepose->getModelByPose(basePts, predParam.pose.data());
		scanCache = DataFilter(scanCache_, weights_, predVec, 0.15f, idxMapper);
		ptcount = scanCache.size();
	}
	bool operator() (const double* pose, double* residual) const
	{
		SimpleTimer timer;
		const auto pts = shapepose_->getModelByPose(basePts, pose);
		float finwgt[EVALUATE_POINTS_NUM];
		memset(residual, 0x7f, sizeof(double)*EVALUATE_POINTS_NUM);
		uint32_t i = 0;
		for (const auto& obj : scanCache)
		{
			const uint32_t idx = idxMapper[i++];
			const double val = (obj - pts[idx]).length();
			if (val < residual[idx])
			{
				residual[idx] = val;
				finwgt[idx] = obj.w;
			}
		}
		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
			residual[j] *= finwgt[j];
		
		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	};
};
struct ShapeCostFunctorPredReShift
{
protected:
	const CShapePose *shapepose_;
	const ModelParam::PoseParam pose;
	std::vector<uint32_t> idxMapper;
	miniBLAS::VertexVec scanCache;
	uint32_t ptcount;
public:
	ShapeCostFunctorPredReShift(CShapePose *shapepose, const ModelParam& modelParam, const ModelParam& predParam, const miniBLAS::VertexVec& scanCache_,
		const std::vector<float>& weights_, const std::vector<uint32_t>& idxMapper_)
		: shapepose_(shapepose), pose(modelParam.pose), idxMapper(idxMapper_)
	{
		const VertexVec predVec = shapepose->getModelFast(predParam.shape.data(), pose.data());
		scanCache = DataFilter(scanCache_, weights_, predVec, 0.15f, idxMapper);
		ptcount = scanCache.size();
	}
	bool operator() (const double* shape, double* residual) const
	{
		SimpleTimer timer;
		const auto pts = shapepose_->getModelFast(shape, pose.data());
		float finwgt[EVALUATE_POINTS_NUM];
		memset(residual, 0x7f, sizeof(double)*EVALUATE_POINTS_NUM);
		uint32_t i = 0;
		for (const auto& obj : scanCache)
		{
			const uint32_t idx = idxMapper[i++];
			const double val = (obj - pts[idx]).length();
			if (val < residual[idx])
			{
				residual[idx] = val;
				finwgt[idx] = obj.w;
			}
		}
		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
			residual[j] *= finwgt[j];

		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	};
};


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

struct PNCalc
{
	template <typename T>
	static T calcute(const T* x, const uint32_t i)
	{
		const T val = T(i);
		return x[0] + x[1] * val + x[2] * ceres::pow(val, 2);
	}
};
struct PSCalc
{
	template <typename T>
	static T calcute(const T* x, const uint32_t i)
	{
		const T val = T(i);
		return x[0] + x[1] * val + x[2] * ceres::sin(val + x[3]);
	}
};
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
		const uint8_t step = 2;
		const uint32_t framesize = params.size();
		const uint32_t maxcnt = calcCount(framesize);
		for (uint32_t i = framesize - 1, j = 0; j < maxcnt; i--, j++)
		{
			const uint32_t level = j / step;
			const T weight = T(1.0) / T(std::pow(level, level));//1/n^n
			const T obj = C::calcute(x, i);
			residual[j] = weight * (T(params[i].pose[idx]) - obj);
		}
		return true;
	}
	static uint32_t calcCount(const uint32_t frames)
	{
		return std::min(frames, (uint32_t)16);
	}
};
template<typename C>
struct ParamSofter
{
protected:
	const std::vector<ModelParam>& params;
	const uint8_t idx;
	const uint32_t objframe;
public:
	using Calc = C;
	ParamSofter(const std::vector<ModelParam>& params_, const uint32_t frame_, const uint8_t idx_) :params(params_), idx(idx_), objframe(frame_) { }
	template <typename T>
	bool operator ()(const T* const x, T* residual) const
	{
		const uint32_t maxframe = std::min(uint32_t(params.size() - 1), objframe + 8);
		const uint32_t minframe = objframe >= 8 ? (objframe - 8) : 0;

		for (uint32_t i = minframe, j = 0; i < maxframe; i++, j++)
		{
			const uint32_t level = 1 + (objframe >= i ? (objframe - i) : (i - objframe));
			const T weight = T(1.0) / T(std::pow(2, level));//1/2^n
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

