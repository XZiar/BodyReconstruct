#pragma once

#include "fitMesh.h"

using miniBLAS::Vertex;
using miniBLAS::VertexVec;

static atomic_uint32_t runcnt(0), runtime(0);
static uint32_t nncnt = 0, nntime = 0;

//Definition of optimization functions
struct PoseCostFunctor
{
private:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const arColIS isValidNN_;
	const VertexVec& scanCache_;
	const VertexVec basePts;

public:
	PoseCostFunctor(CShapePose *shapepose, const ModelParam& modelParam, const arColIS isValidNN, const miniBLAS::VertexVec& scanCache)
		: shapepose_(shapepose), isValidNN_(isValidNN), scanCache_(scanCache), basePts(shapepose_->getBaseModel(modelParam.shape))
	{
	}
	//pose is the parameters to be estimated, b is the bias, residual is to return
	bool operator()(const double* pose, double* residual) const
	{
		uint64_t t1, t2;
		t1 = getCurTimeNS();

		const auto pts = shapepose_->getModelByPose(basePts, pose);
		auto *__restrict pValid = isValidNN_.memptr();
		uint32_t i = 0;
		for (int j = 0; j < isValidNN_.n_elem; ++j)
		{
			if (pValid[j])
			{
				const Vertex delta = scanCache_[j] - pts[j];
				residual[i + 0] = delta.x;
				residual[i + 1] = delta.y;
				residual[i + 2] = delta.z;
				i += 3;
			}
		}
		//printf("now i=%d, total=%d, demand=%d\n", i, isValidNN_.n_elem * 3, 3 * EVALUATE_POINTS_NUM);
		memset(&residual[i], 0, sizeof(double) * (3 * EVALUATE_POINTS_NUM - i));

		runcnt++;
		t2 = getCurTimeNS();
		runtime += (uint32_t)((t2 - t1) / 1000);
		return true;
	}
};
struct ShapeCostFunctor
{
private:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const double(&poseParam_)[POSPARAM_NUM];
	const arColIS isValidNN_;
	const VertexVec& scanCache_;
public:
	ShapeCostFunctor(CShapePose *shapepose, const ModelParam& modelParam, const arColIS isValidNN, const miniBLAS::VertexVec& scanCache)
		: shapepose_(shapepose), poseParam_(modelParam.pose), isValidNN_(isValidNN), scanCache_(scanCache)
	{
	}
	//w is the parameters to be estimated, b is the bias, residual is to return
	bool operator() (const double* shape, double* residual) const
	{
		uint64_t t1, t2;
		t1 = getCurTimeNS();

		const auto *__restrict pValid = isValidNN_.memptr();
		const auto pts = shapepose_->getModelFast(shape, poseParam_);
		uint32_t i = 0;
		for (int j = 0; j < isValidNN_.n_elem; ++j)
		{
			if (pValid[j])
			{
				const Vertex delta = scanCache_[j] - pts[j];
				residual[i + 0] = delta.x;
				residual[i + 1] = delta.y;
				residual[i + 2] = delta.z;
				i += 3;
			}
		}
		memset(&residual[i], 0, sizeof(double) * (3 * EVALUATE_POINTS_NUM - i));

		runcnt++;
		t2 = getCurTimeNS();
		runtime += (uint32_t)((t2 - t1) / 1000);
		return true;
	}
};
struct PoseCostFunctorEx2
{
private:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const arColIS isValidNN_;
	const VertexVec& validScanCache_;
	const CMesh baseMesh;
	const PtrModSmooth mSmooth;
	std::vector<float> weights;
public:
	PoseCostFunctorEx2(CShapePose *shapepose, const ModelParam& modelParam, const arColIS isValidNN, const miniBLAS::VertexVec& validScanCache,
		PtrModSmooth mSmooth_)
		: shapepose_(shapepose), isValidNN_(isValidNN), validScanCache_(validScanCache), mSmooth(mSmooth_),
		baseMesh(shapepose_->getBaseModel2(modelParam.shape, isValidNN_.memptr()))
	{
	}
	PoseCostFunctorEx2(CShapePose *shapepose, const ModelParam& modelParam, const arColIS isValidNN, const miniBLAS::VertexVec& validScanCache,
		PtrModSmooth mSmooth_, const std::vector<float>& wgts)
		: shapepose_(shapepose), isValidNN_(isValidNN), validScanCache_(validScanCache), mSmooth(mSmooth_),
		baseMesh(shapepose_->getBaseModel2(modelParam.shape, isValidNN_.memptr()))
	{
		weights = wgts;
	}
	//pose is the parameters to be estimated, b is the bias, residual is to return
	bool operator()(const double* pose, double* residual) const
	{
		const bool isWgt = (weights.size() > 0);
		uint64_t t1, t2;
		t1 = getCurTimeNS();

		auto *__restrict pValid = isValidNN_.memptr();
		const auto pts = shapepose_->getModelByPose2(mSmooth, baseMesh, pose, pValid);

		const uint32_t cnt = validScanCache_.size();
		for (uint32_t i = 0, j = 0; j < cnt; ++j)
		{
			const Vertex delta = (validScanCache_[j] - pts[j]) * (isWgt ? weights[j] : 1);
			residual[i + 0] = delta.x;
			residual[i + 1] = delta.y;
			residual[i + 2] = delta.z;
			i += 3;
		}
		memset(&residual[3 * cnt], 0, sizeof(double) * 3 * (EVALUATE_POINTS_NUM - cnt));

		runcnt++;
		t2 = getCurTimeNS();
		runtime += (uint32_t)((t2 - t1) / 1000);
		return true;
	}
};
struct ShapeCostFunctorEx2
{
private:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const double(&poseParam_)[POSPARAM_NUM];
	const arColIS isValidNN_;
	const VertexVec& validScanCache_;
	const PtrModSmooth mSmooth;
public:
	ShapeCostFunctorEx2(CShapePose *shapepose, const ModelParam& modelParam, const arColIS isValidNN, const miniBLAS::VertexVec& validScanCache,
		PtrModSmooth mSmooth_)
		: shapepose_(shapepose), poseParam_(modelParam.pose), isValidNN_(isValidNN), validScanCache_(validScanCache), mSmooth(mSmooth_)
	{
	}
	//w is the parameters to be estimated, b is the bias, residual is to return
	bool operator() (const double* shape, double* residual) const
	{
		uint64_t t1, t2;
		t1 = getCurTimeNS();

		const auto *__restrict pValid = isValidNN_.memptr();
		const auto pts = shapepose_->getModelFast2(mSmooth, shape, poseParam_, pValid);

		const uint32_t cnt = validScanCache_.size();
		for (uint32_t i = 0, j = 0; j < cnt; ++j)
		{
			const Vertex delta = validScanCache_[j] - pts[j];
			residual[i + 0] = delta.x;
			residual[i + 1] = delta.y;
			residual[i + 2] = delta.z;
			i += 3;
		}
		memset(&residual[3 * cnt], 0, sizeof(double) * 3 * (EVALUATE_POINTS_NUM - cnt));

		runcnt++;
		t2 = getCurTimeNS();
		runtime += (uint32_t)((t2 - t1) / 1000);
		return true;
	}
};

struct PoseRegularizer
{
private:
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
private:
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
private:
	const double(&poseParam)[POSPARAM_NUM];
	const double weight;
public:
	MovementSofter(const ModelParam& modelParam, const double w) :poseParam(modelParam.pose), weight(w) { }
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
private:
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

struct MultiShapeCostFunctor
{
private:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const std::vector<ModelParam>& datParams;
	const std::vector<arColIS>& datNNs;
	const std::vector<VertexVec>& datScans;
public:
	MultiShapeCostFunctor(CShapePose *shapepose,
		const std::vector<ModelParam>& datParams_, const std::vector<arColIS>& datNNs_, const std::vector<VertexVec>& datScans_)
		: shapepose_(shapepose), datParams(datParams_), datNNs(datNNs_), datScans(datScans_)
	{
		runtime = 0;
	}
	bool operator() (const double* shape, double* residual) const
	{
		const auto len = datNNs[0].n_elem;
		memset(residual, 0x0, sizeof(double)*len);
		uint64_t t1, t2;
		t1 = getCurTimeNS();
		for (uint32_t a = 0; a < datParams.size(); ++a)
		{
			const auto pts = shapepose_->getModelFast(shape, datParams[a].pose);
			const auto *__restrict pValid = datNNs[a].memptr();
			const auto& curScan = datScans[a];
			for (uint32_t i = 0, j = 0; j < len; ++j, i += 3)
				if (pValid[j])
				{
					const Vertex delta = curScan[j] - pts[j];
					residual[i + 0] += delta.x;
					residual[i + 1] += delta.y;
					residual[i + 2] += delta.z;
				}
		}
		runcnt++;
		t2 = getCurTimeNS();
		runtime += (uint32_t)((t2 - t1) / 1000);
		return true;
	}
};