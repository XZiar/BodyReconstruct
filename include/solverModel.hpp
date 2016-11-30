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
		: shapepose_(shapepose), isValidNN_(isValidNN), scanCache_(scanCache), basePts(shapepose_->getBaseModel(modelParam.shape))
	{
	}
	bool operator()(const double* pose, double* residual) const
	{
		SimpleTimer timer;

		const auto pts = shapepose_->getModelByPose(basePts, pose);
		auto *__restrict pValid = isValidNN_.memptr();
		uint32_t i = 0;
		for (int j = 0; j < isValidNN_.n_elem; ++j)
		{
			if (pValid[j])
			{
				const Vertex delta = scanCache_[j] - pts[j];
				delta.save<3>(&residual[i]);
				i += 3;
			}
		}
		//printf("now i=%d, total=%d, demand=%d\n", i, isValidNN_.n_elem * 3, 3 * EVALUATE_POINTS_NUM);
		memset(&residual[i], 0, sizeof(double) * (3 * EVALUATE_POINTS_NUM - i));

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
	const double(&poseParam_)[POSPARAM_NUM];
	const arColIS isValidNN_;
	const VertexVec& scanCache_;
public:
	ShapeCostFunctor(CShapePose *shapepose, const ModelParam& modelParam, const arColIS& isValidNN, const miniBLAS::VertexVec& scanCache)
		: shapepose_(shapepose), poseParam_(modelParam.pose), isValidNN_(isValidNN), scanCache_(scanCache)
	{
	}
	bool operator() (const double* shape, double* residual) const
	{
		SimpleTimer timer;

		const auto *__restrict pValid = isValidNN_.memptr();
		const auto pts = shapepose_->getModelFast(shape, poseParam_);
		uint32_t i = 0;
		for (int j = 0; j < isValidNN_.n_elem; ++j)
		{
			if (pValid[j])
			{
				const Vertex delta = scanCache_[j] - pts[j];
				delta.save<3>(&residual[i]);
				i += 3;
			}
		}
		memset(&residual[i], 0, sizeof(double) * (3 * EVALUATE_POINTS_NUM - i));

		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	}
};
struct ShapeCostFunctor_D : public ShapeCostFunctor
{
	using ShapeCostFunctor::ShapeCostFunctor;
	bool operator()(const double* shape, double* residual) const
	{
		SimpleTimer timer;

		const auto *__restrict pValid = isValidNN_.memptr();
		const auto pts = shapepose_->getModelFast(shape, poseParam_);

		for (uint32_t j = 0; j < EVALUATE_POINTS_NUM; ++j)
		{
			if (pValid[j])
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
		baseMesh(shapepose_->getBaseModel2(modelParam.shape, isValidNN_.memptr()))
	{
		weights = wgts;
	}
	bool operator()(const double* pose, double* residual) const
	{
		const bool isWgt = (weights.size() > 0);
		SimpleTimer timer;

		auto *__restrict pValid = isValidNN_.memptr();
		const auto pts = shapepose_->getModelByPose2(mSmooth, baseMesh, pose, pValid);

		const uint32_t cnt = validScanCache_.size();
		for (uint32_t i = 0, j = 0; j < cnt; ++j, i += 3)
		{
			const Vertex delta = (validScanCache_[j] - pts[j]) * (isWgt ? weights[j] : 1);
			delta.save<3>(&residual[i]);
		}
		memset(&residual[3 * cnt], 0, sizeof(double) * 3 * (EVALUATE_POINTS_NUM - cnt));

		timer.Stop();
		runcnt++;
		runtime += timer.ElapseUs();
		return true;
	}
};
struct PoseCostFunctorEx2_D : public PoseCostFunctorEx2
{
	using PoseCostFunctorEx2::PoseCostFunctorEx2;
	bool operator()(const double* pose, double* residual) const
	{
		const bool isWgt = (weights.size() > 0);
		SimpleTimer timer;

		auto *__restrict pValid = isValidNN_.memptr();
		const auto pts = shapepose_->getModelByPose2(mSmooth, baseMesh, pose, pValid);

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
	const double(&poseParam_)[POSPARAM_NUM];
	const arColIS isValidNN_;
	const VertexVec& validScanCache_;
	const PtrModSmooth mSmooth;
public:
	ShapeCostFunctorEx2(CShapePose *shapepose, const ModelParam& modelParam, const arColIS& isValidNN, const miniBLAS::VertexVec& validScanCache,
		PtrModSmooth mSmooth_)
		: shapepose_(shapepose), poseParam_(modelParam.pose), isValidNN_(isValidNN), validScanCache_(validScanCache), mSmooth(mSmooth_)
	{
	}
	bool operator() (const double* shape, double* residual) const
	{
		SimpleTimer timer;

		const auto *__restrict pValid = isValidNN_.memptr();
		const auto pts = shapepose_->getModelFast2(mSmooth, shape, poseParam_, pValid);

		const uint32_t cnt = validScanCache_.size();
		for (uint32_t i = 0, j = 0; j < cnt; ++j, i += 3)
		{
			const Vertex delta = validScanCache_[j] - pts[j];
			delta.save<3>(&residual[i]);
		}
		memset(&residual[3 * cnt], 0, sizeof(double) * 3 * (EVALUATE_POINTS_NUM - cnt));

		timer.Stop();
		runcnt++; 
		runtime += timer.ElapseUs();
		return true;
	}
};
struct ShapeCostFunctorEx2_D : public ShapeCostFunctorEx2
{
	using ShapeCostFunctorEx2::ShapeCostFunctorEx2;
	bool operator()(const double* shape, double* residual) const
	{
		SimpleTimer timer;

		const auto *__restrict pValid = isValidNN_.memptr();
		const auto pts = shapepose_->getModelFast2(mSmooth, shape, poseParam_, pValid);

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
	const arColIS isValidNN_;
	const VertexVec ptCache;
	const VertexVec basePt;
	const std::vector<float> weights;

	static VertexVec buildCache(CShapePose *shapepose, const ModelParam& modelParam, const arColIS& isValidNN, const VertexVec& validScanCache_)
	{
		VertexVec baseVec = shapepose->getModelFast(modelParam.shape, modelParam.pose);
		const char *pValid = isValidNN.memptr();
		for (uint32_t a = 0, b = 0; a < EVALUATE_POINTS_NUM; ++a)
		{
			if (pValid[a])
				baseVec[a] = validScanCache_[b++];
		}
		return baseVec;
	}
	static std::vector<float> buildCache(const arColIS& isValidNN, const std::vector<float>& wgts = std::vector<float>())
	{
		std::vector<float> weights(EVALUATE_POINTS_NUM, 0.35f);
		const bool isWgt = (wgts.size() > 0);
		const char *pValid = isValidNN.memptr();
		for (uint32_t a = 0, b = 0; a < EVALUATE_POINTS_NUM; ++a)
		{
			if (pValid[a])
				weights[a] = (isWgt ? wgts[b++] : 1);
		}
		return weights;
	}
public:
	PoseCostFunctorPred(CShapePose *shapepose, const ModelParam& modelParam, const ModelParam& preParam, const arColIS isValidNN, const miniBLAS::VertexVec& validScanCache,
		const std::vector<float>& wgts = std::vector<float>())
		: shapepose_(shapepose), isValidNN_(isValidNN), weights(buildCache(isValidNN, wgts)),
		basePt(shapepose_->getBaseModel(modelParam.shape)), ptCache(buildCache(shapepose, preParam, isValidNN, validScanCache))
	{
	}
	bool operator()(const double *pose, double *residual) const
	{
		SimpleTimer timer;

		const auto pts = shapepose_->getModelByPose(basePt, pose);
		for (uint32_t i = 0, j = 0; j < EVALUATE_POINTS_NUM; ++j, i += 3)
		{
			const Vertex delta = (ptCache[j] - pts[j]) * weights[j];
			delta.save<3>(&residual[i]);
		}
		timer.Stop();
		runcnt++;
		runtime += uint32_t(timer.ElapseUs());
		return true;
	}
};
struct PoseCostFunctorPred_D : public PoseCostFunctorPred
{
	using PoseCostFunctorPred::PoseCostFunctorPred;
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

struct ShapeCostFunctorRe_D
{
protected:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const double(&poseParam_)[POSPARAM_NUM];
	const std::vector<float> weights;
	const std::vector<uint32_t> idxMapper;
	const uint32_t ptcount;
	const miniBLAS::VertexVec& scanCache;
public:
	ShapeCostFunctorRe_D(CShapePose *shapepose, const ModelParam& modelParam, const miniBLAS::VertexVec& scanCache_,
		const std::vector<float>& weights_, const std::vector<uint32_t>& idxMapper_, const uint32_t ptcount_)
		: shapepose_(shapepose), poseParam_(modelParam.pose), weights(weights_), idxMapper(idxMapper_), scanCache(scanCache_), ptcount(ptcount_)
	{
	}
	bool operator() (const double* shape, double* residual) const
	{
		SimpleTimer timer;
		const auto pts = shapepose_->getModelFast(shape, poseParam_);
		VertexVec tmp(1 + EVALUATE_POINTS_NUM / 4);
		memset(&tmp[0], 0x0, (1 + EVALUATE_POINTS_NUM / 4) * sizeof(Vertex));
		float *__restrict pTmp = tmp[0];
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
	const double(&poseParam)[POSPARAM_NUM];
	const double weight;
public:
	MovementSofter(const ModelParam& modelParam, const double err) :poseParam(modelParam.pose),
		weight(sqrt(err / (POSPARAM_NUM - 6))) { }
	bool operator()(const double* pose, double* residual) const
	{
		uint32_t i = 0;
		for (; i < 6; ++i)
			residual[i] = 0;
		for (; i < POSPARAM_NUM; ++i)
			residual[i] = weight * abs(pose[i] - poseParam[i]);
		return true;
	}
};
struct ShapeSofter
{
protected:
	const double(&shapeParam)[SHAPEPARAM_NUM];
	const double weight;
public:
	ShapeSofter(const ModelParam& modelParam, const double w) :shapeParam(modelParam.shape), weight(w) { }
	bool operator()(const double* shape, double* residual) const
	{
		for (uint32_t i = 0; i < SHAPEPARAM_NUM; ++i)
			residual[i] = weight * abs(shape[i] - shapeParam[i]);
		return true;
	}
};
