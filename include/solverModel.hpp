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
		std::vector<float> weights(EVALUATE_POINTS_NUM, 0.35f);
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
		const std::vector<float>& weights_, const std::vector<uint32_t>& idxMapper_, const uint32_t ptcount_)
		: shapepose_(shapepose), idxMapper(idxMapper_), basePts(shapepose_->getBaseModel(modelParam.shape.data())),
		scanCache(scanCache_), ptcount(ptcount_)
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
		const std::vector<float>& weights_, const std::vector<uint32_t>& idxMapper_, const uint32_t ptcount_)
		: shapepose_(shapepose), pose(modelParam.pose), idxMapper(idxMapper_), scanCache(scanCache_), ptcount(ptcount_)
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

struct PoseCostFunctorReShift
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
	PoseCostFunctorReShift(CShapePose *shapepose, const ModelParam& modelParam, const miniBLAS::VertexVec& scanCache_,
		const std::vector<float>& weights_, const std::vector<uint32_t>& idxMapper_, const uint32_t ptcount_)
		: shapepose_(shapepose), idxMapper(idxMapper_), weights(weights_), basePts(shapepose_->getBaseModel(modelParam.shape.data())),
		scanCache(scanCache_), ptcount(ptcount_)
	{
	}
	bool operator() (const double* pose, double* residual) const
	{
		SimpleTimer timer;
		const auto pts = shapepose_->getModelByPose(basePts, pose);

		int8_t flags[EVALUATE_POINTS_NUM] = { 0 };
		for (uint32_t j = 0; j < ptcount; ++j)
		{
			const auto idx = idxMapper[j];
			const double val = (scanCache[j] - pts[idx]).length() * weights[j];
			if (flags[idx])//choose min
				residual[idx] = std::min(residual[idx], val);
			else
			{
				residual[idx] = val;
				flags[idx] = 1;
			}
		}
		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
			if (!flags[j])
				residual[j] = 0;

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
	std::vector<float> weights;
	const std::vector<uint32_t> idxMapper;
	const uint32_t ptcount;
	const miniBLAS::VertexVec& scanCache;
public:
	ShapeCostFunctorReShift(CShapePose *shapepose, const ModelParam& modelParam, const miniBLAS::VertexVec& scanCache_,
		const std::vector<float>& weights_, const std::vector<uint32_t>& idxMapper_, const uint32_t ptcount_)
		: shapepose_(shapepose), pose(modelParam.pose), idxMapper(idxMapper_), weights(weights_), scanCache(scanCache_), ptcount(ptcount_)
	{
	}
	bool operator() (const double* shape, double* residual) const
	{
		SimpleTimer timer;
		const auto pts = shapepose_->getModelFast(shape, pose.data());

		int8_t flags[EVALUATE_POINTS_NUM] = { 0 };
		for (uint32_t j = 0; j < ptcount; ++j)
		{
			const auto idx = idxMapper[j];
			const double val = (scanCache[j] - pts[idx]).length() * weights[j];
			if (flags[idx])//choose min
				residual[idx] = std::min(residual[idx], val);
			else
			{
				residual[idx] = val;
				flags[idx] = 1;
			}
		}
		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
			if (!flags[j])
				residual[j] = 0;

		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	}
};

struct PoseRegularizer
{
protected:
	int dim_;
	double weight_;
public:
	PoseRegularizer(double weight, int dim) :weight_(weight), dim_(dim)
	{
		//cout<<"the weight:"<<weight_<<endl;
	}
	template <typename T>
	bool operator ()(const T* const w, T* residual) const
	{
		//        T sum=T(0);
		for (int i = 0; i < dim_; i++)
		{
			//            sum += T(w[i]);
			residual[i] = weight_*T(w[i]);
		}
		//        residual[0]=weight_*sqrt(sum);
		return true;
	}
};

struct ShapeRegularizer
{
protected:
	int dim_;
	double weight_;
public:
	ShapeRegularizer(double weight, int dim) :weight_(weight), dim_(dim)
	{
		//cout<<"the weight:"<<weight_<<endl;
	}
	template <typename T>
	bool operator ()(const T* const w, T* residual) const
	{
		for (int i = 0; i < dim_; i++)
		{
			//            sum += T(w[i]);
			residual[i] = weight_*T(w[i]);
		}
		return true;
	}
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
	ShapeSofter(const ModelParam& modelParam, const double w) :lastshape(modelParam.shape), weight(w) { }
	bool operator()(const double* shape, double* residual) const
	{
		for (uint32_t i = 0; i < SHAPEPARAM_NUM; ++i)
			residual[i] = weight * abs(shape[i] - lastshape[i]);
		return true;
	}
};
