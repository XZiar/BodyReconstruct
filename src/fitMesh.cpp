#include "fitMesh.h"

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::isnan;
using std::move;
using std::function;
using std::tuple;
using std::shared_ptr;
using std::unique_ptr;
using std::make_shared;
using std::make_tuple;
using atomic_uint32_t = std::atomic_uint_least32_t;

using arma::zeros;
using arma::ones;
using arma::mean;
using ceres::Problem;
using ceres::Solver;
using miniBLAS::Vertex;
using miniBLAS::VertexVec;

using PointT = pcl::PointXYZRGBNormal;
using PointCloudT = pcl::PointCloud<PointT>;


bool FastTriangle::intersect(const miniBLAS::Vertex& origin, const miniBLAS::Vertex& direction, const miniBLAS::VertexI& idx, const float dist) const
{
	if (_mm_movemask_epi8(_mm_cmpeq_epi32(idx, pidx)) != 0)//point is one ertex of the triangle
		return false;
	/*
	** Point(u,v) = (1-u-v)*p0 + u*p1 + v*p2
	** Ray:Point(t) = o + t*dir
	** o + t*dir = (1-u-v)*p0 + u*p1 + v*p2
	*/

	const Vertex tmp1 = direction * axisv;
	float f = axisu % tmp1;
	if (abs(f) < 1e-6f)
		return false;
	f = 1.0f / f;

	const Vertex t2r = origin - p0;
	const float u = (t2r % tmp1) * f;
	if (u <= 1e-6f || u >= 1.0f - 1e-6f)
		return false;
	const Vertex tmp2 = t2r * axisu;
	const float v = (direction % tmp2) * f;
	if (v <= 1e-6f || u + v >= 1.0f - 1e-6f)
		return false;
	const float t = (axisv % tmp2) * f;
	//printf("u:%f,v:%f,f:%f,t:%f,obj-dist:%f\n", u, v, f, t, dist);
	//getchar();
	if (t < dist)
		return true;
	else
		return false;
}


void CScan::prepare()
{
	vPts.resize(nPoints); vNorms.resize(nPoints);

	const double *__restrict px = points.memptr(), *__restrict py = px + nPoints, *__restrict pz = py + nPoints;
	const double *__restrict pnx = normals.memptr(), *__restrict pny = pnx + nPoints, *__restrict pnz = pny + nPoints;
	for (uint32_t i = 0; i < nPoints; ++i)
	{
		vPts[i].assign(*px++, *py++, *pz++);
		vNorms[i].assign(*pnx++, *pny++, *pnz++);
	}
}


void CTemplate::init(const arma::mat& p, const arma::mat& f)
{
	points = p;
	nPoints = points.n_rows;
	updPoints();

	faces = f;
	nFaces = faces.n_rows;
	vFaces.resize(nFaces);
	faceMap.resize(nPoints, UINT32_MAX);
	const double *px = faces.memptr(), *py = px + nFaces, *pz = py + nFaces;
	for (uint32_t i = 0, j = 0; i < nFaces; ++i, j += 3)
	{
		auto& objidx = vFaces[i].pidx;
		objidx.assign(int32_t(px[i]), int32_t(py[i]), int32_t(pz[i]), 65536);

		auto& tx = faceMap[objidx.x];
		if (tx == UINT32_MAX)
			tx = i;
		auto& ty = faceMap[objidx.y];
		if (ty == UINT32_MAX)
			ty = i;
		auto& tz = faceMap[objidx.z];
		if (tz == UINT32_MAX)
			tz = i;
	}
}

void CTemplate::updPoints()
{
	vPts.resize(nPoints);
	const double *__restrict px = points.memptr(), *__restrict py = px + nPoints, *__restrict pz = py + nPoints;
	for (uint32_t i = 0; i < nPoints; ++i)
		vPts[i].assign(*px++, *py++, *pz++);
}

void CTemplate::updPoints(miniBLAS::VertexVec&& pts)
{
	vPts = pts;
	auto *__restrict px = points.memptr(), *__restrict py = px + nPoints, *__restrict pz = py + nPoints;
	for (uint32_t a = 0; a < nPoints; ++a)
	{
		const Vertex tmp = vPts[a];
		*px++ = tmp.x; *py++ = tmp.y; *pz++ = tmp.z;
	}
}

void CTemplate::calcFaces()
{
	for (uint32_t i = 0, j = 0; i < nFaces; ++i, j += 3)
	{
		auto& obj = vFaces[i];
		obj.p0 = vPts[obj.pidx.x];
		obj.axisu = vPts[obj.pidx.y] - obj.p0;
		obj.axisv = vPts[obj.pidx.z] - obj.p0;
		obj.norm = (obj.axisu * obj.axisv).norm();
	}
}

void CTemplate::calcNormals()
{
	vNorms.resize(nPoints); 
	for (uint32_t i = 0; i < nPoints; i++)
		vNorms[i] = vFaces[faceMap[i]].norm;
}


SimpleLog::SimpleLog()
{
	fname = std::to_string(getCurTime()) + ".log";
}

SimpleLog::~SimpleLog()
{
	if (fp != nullptr)
		fclose(fp);
}

SimpleLog & SimpleLog::log(const std::string& str, const bool isPrint)
{
	cache += str;
	if (isPrint)
		printf("%s", str.c_str());
	return *this;
}

void SimpleLog::flush()
{
	if (fp == nullptr)
		fp = fopen(fname.c_str(), "w");
	fprintf(fp, cache.c_str());
	cache = "";
}


static VertexVec armaTOcache(const arma::mat in)
{
	const uint32_t cnt = in.n_rows;
	VertexVec cache(cnt);
	Vertex *__restrict pVert = &cache[0];

	const double *__restrict px = in.memptr(), *__restrict py = px + cnt, *__restrict pz = py + cnt;
	for (uint32_t i = 0; i < cnt; ++i)
		*pVert++ = Vertex(*px++, *py++, *pz++);

	return cache;
}
static VertexVec armaTOcache(const arma::mat in, const uint32_t *idx, const uint32_t cnt)
{
	VertexVec cache(cnt);
	Vertex *__restrict pVert = &cache[0];

	const double *__restrict px = in.memptr(), *__restrict py = in.memptr() + in.n_rows, *__restrict pz = in.memptr() + 2 * in.n_rows;
	for (uint32_t i = 0; i < cnt; ++i)
	{
		const uint32_t off = idx[i];
		pVert[i] = Vertex(px[off], py[off], pz[off]);
	}
	return cache;
}
static VertexVec shuffleANDfilter(const VertexVec& in, const uint32_t cnt, const uint32_t *idx, const char *mask = nullptr)
{
	VertexVec cache;
	if (mask != nullptr)
	{
		cache.reserve(cnt / 2);
		for (uint32_t i = 0; i < cnt; ++i)
			if (mask[i] != 0)
				cache.push_back(in[idx[i]]);
	}
	else
	{
		cache.resize(cnt);
		for (uint32_t i = 0; i < cnt; ++i)
			cache[i] = in[idx[i]];
	}
	return cache;
}

static pcl::visualization::CloudViewer viewer("viewer");

static atomic_uint32_t runcnt(0), runtime(0);
static uint32_t nncnt = 0, nntime = 0;
//Definition of optimization functions
struct PoseCostFunctorEx
{
private:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const arColIS isValidNN_;
	const VertexVec& scanCache_;
	const VertexVec basePts;

public:
	PoseCostFunctorEx(CShapePose *shapepose, const ModelParam& modelParam, const arColIS isValidNN, const miniBLAS::VertexVec& scanCache)
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
struct ShapeCostFunctorEx
{
private:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const double (&poseParam_)[POSPARAM_NUM];
	const arColIS isValidNN_;
	const VertexVec& scanCache_;
public:
	ShapeCostFunctorEx(CShapePose *shapepose, const ModelParam& modelParam, const arColIS isValidNN, const miniBLAS::VertexVec& scanCache)
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

public:
	PoseCostFunctorEx2(CShapePose *shapepose, const ModelParam& modelParam, const arColIS isValidNN, const miniBLAS::VertexVec& validScanCache)
		: shapepose_(shapepose), isValidNN_(isValidNN), validScanCache_(validScanCache),
		baseMesh(shapepose_->getBaseModel2(modelParam.shape, isValidNN_.memptr()))
	{
	}
	//pose is the parameters to be estimated, b is the bias, residual is to return
	bool operator()(const double* pose, double* residual) const
	{
		uint64_t t1, t2;
		t1 = getCurTimeNS();

		auto *__restrict pValid = isValidNN_.memptr();
		const auto pts = shapepose_->getModelByPose2(baseMesh, pose, pValid);

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
struct ShapeCostFunctorEx2
{
private:
	// this should be the firtst to declare in order to be initialized before other things
	const CShapePose *shapepose_;
	const double (&poseParam_)[POSPARAM_NUM];
	const arColIS isValidNN_;
	const VertexVec& validScanCache_;
public:
	ShapeCostFunctorEx2(CShapePose *shapepose, const ModelParam& modelParam, const arColIS isValidNN, const miniBLAS::VertexVec& validScanCache)
		: shapepose_(shapepose), poseParam_(modelParam.pose), isValidNN_(isValidNN), validScanCache_(validScanCache)
	{
	}
	//w is the parameters to be estimated, b is the bias, residual is to return
	bool operator() (const double* shape, double* residual) const
	{
		uint64_t t1, t2;
		t1 = getCurTimeNS();

		const auto *__restrict pValid = isValidNN_.memptr();
		const auto pts = shapepose_->getModelFast2(shape, poseParam_, pValid);

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
//===============================================================

static int inputNumber(const char *str)
{
    printf("%s : ", str);
    int num;
    scanf("%d", &num);
    getchar();
    return num;
}

void printMat(const char * str, arma::mat v)
{
    printf("%s: ",str);
	v.for_each([](const double& val) { printf("%.4e,", val); });
    printf("\n");
}
void printMatAll(const char * str, arma::mat m)
{
    printf("matrix %s :\n", str);
    const uint32_t ccnt = m.n_cols;
    m.each_row([&](const arma::rowvec row)
    {
       for(uint32_t a=0;a<ccnt;++a)
           printf("%e,",row(a));
       printf("\n");
    });
}
template<uint32_t N>
static void printArray(const char* str, const double(&array)[N])
{
	printf("%s:\t", str);
	for (uint32_t a = 0; a < N; ++a)
		printf("%f ", array[a]);
	printf("\n");
}

static void showPoints(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const arma::mat& model, pcl::PointXYZRGB color)
{
    model.each_row([&](const arma::rowvec &row)
    {
        color.x = row(0);
        color.y = row(1);
        color.z = row(2);
        cloud->push_back(color);
    });
}
static void showPoints(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const arma::mat& model, const arma::mat& norm, pcl::PointXYZRGB color)
{
    uint32_t idx = 0;
    model.each_row([&](const arma::rowvec &row)
    {
        const double y = norm.row(idx++)(1);
		color.b = (uint8_t)(y * 32 + (y > 0 ? 255-32 : 32));
        color.x = row(0);
        color.y = row(1);
        color.z = row(2);
        cloud->push_back(color);
    });
}
static void showPoints(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const VertexVec& model, const VertexVec& norm, pcl::PointXYZRGB color)
{
	const uint32_t cnt = std::min(model.size(), norm.size());
	for (uint32_t i = 0; i < cnt; ++i)
	{
		const Vertex& pt = model[i];
		const auto y = norm[i].y;
		color.b = (uint8_t)(y * 32 + (y > 0 ? 255 - 32 : 32));
		color.x = pt.x; color.y = pt.y; color.z = pt.z;
		cloud->push_back(color);
	}
}
/*
static cv::Mat armaTOcv(const arma::mat in)
{
    cv::Mat out(in.n_rows, in.n_cols, CV_32FC1);
    float *__restrict const pOut = out.ptr<float>(0);
    const double *__restrict pIn = in.memptr();
    const uint32_t step = in.n_cols;
    for(uint32_t off = 0; off < in.n_cols; ++off)
    {
        float *__restrict ptr = pOut + off;
        for(uint32_t a = 0; a < in.n_rows; ++a)
        {
            *ptr = (float)*(pIn++);
            ptr += step;
        }
    }
    return out;
}
*/
static void saveMat(const char* fname, const arma::mat& m)
{
	FILE *fp = fopen(fname, "w");
	m.each_row([&](const arma::rowvec& row)
	{
		double x = row(0), y = row(1), z = row(2), s = x*x + y*y + z*z;
		fprintf(fp, "%8.6f %8.6f %8.6f = %8.6f\n", x, y, z, s);
	});
	fclose(fp);
}

CShapePose fitMesh::shapepose;
CTemplate fitMesh::tempbody;
ctools fitMesh::tools;

fitMesh::fitMesh(std::string dir)
{
    nSamplePoints = 60000;
	dataDir = dir;
	loadModel();
	loadTemplate();
}

fitMesh::~fitMesh()
{
	
}
/** @brief loadLandmarks
  *
  * @todo: document this function
  */
void fitMesh::loadLandmarks()
{

}

/** @brief loadScan
  *
  * @todo: document this function
  */
bool fitMesh::loadScan(CScan& scan)
{
	PointCloudT::Ptr cloud_tmp(new PointCloudT);
	pcl::PolygonMesh mesh;
	string fname = dataDir + baseFName;
	if (mode)
		fname += ".ply";
	else
		fname += "_" + std::to_string(curFrame) + ".ply";
	{
		FILE* fp = fopen(fname.c_str(), "rb");
		if (fp == nullptr)
		{
			printf("File %s not exist!\n", fname.c_str());
			return false;
		}
		fclose(fp);
	}

	cout << "loading " << fname << endl;
	const int read_status = pcl::io::loadPLYFile(fname, *cloud_tmp);
	printf("read status: %d, load %lld points AND normals.\n", read_status, cloud_tmp->points.size());

	scan.nPoints = cloud_tmp->points.size();
	scan.points_orig.resize(scan.nPoints, 3);
	scan.normals_orig.resize(scan.nPoints, 3);
	uint32_t idx = 0;
	if (mode)
	{
		for (const PointT& pt : cloud_tmp->points)//m 2 mm
		{
			scan.points_orig.row(idx) = arma::rowvec({ pt.x, -pt.y, pt.z }) * 1000;
			scan.normals_orig.row(idx) = arma::rowvec({ pt.normal_x, -pt.normal_y, pt.normal_z });
			idx++;
		}
	}
	else
	{
		for (const PointT& pt : cloud_tmp->points)//m 2 mm
		{
			scan.points_orig.row(idx) = arma::rowvec({ -pt.y, pt.z, -pt.x }) * 1000;
			scan.normals_orig.row(idx) = arma::rowvec({ -pt.normal_y, pt.normal_z, -pt.normal_x });
			idx++;
		}
	}
	//normalize the normals
	scan.normals_orig = arma::normalise(scan.normals_orig, 2, 1);

	//no faces will be loaded, and the normals are calculated when sampling the scan datas
	if (scan.nPoints <= nSamplePoints)//Not sample the scan points
	{
		scan.sample_point_idxes.clear();
		scan.points = scan.points_orig;
		scan.normals = scan.normals_orig;
	}
	else//sample the scan points if necessary;
	{
		scan.sample_point_idxes = tools.randperm(scan.nPoints, nSamplePoints);
		scan.nPoints = nSamplePoints;
		scan.points = arma::zeros(scan.nPoints, 3);
		scan.normals = arma::zeros(scan.nPoints, 3);
		for (uint32_t i = 0; i < nSamplePoints; i++)
		{
			scan.points.row(i) = scan.points_orig.row(scan.sample_point_idxes[i]);
			scan.normals.row(i) = scan.normals_orig.row(scan.sample_point_idxes[i]);
		}
	}
	return true;
}

/** @brief loadTemplate
  *
  * @todo: document this function
  */
void fitMesh::loadTemplate()
{
	arma::mat pts;
	//template.mat should be pre-calculated : meanshape.mat-mean(meanshape)
	pts.load(dataDir + "template.mat");
	cout << "Template points loaded: " << pts.n_rows << "," << pts.n_cols << endl;
	
	arma::mat faces;
	faces.load(dataDir + "faces.mat");
	cout << "Template faces loaded: " << faces.n_rows << "," << faces.n_cols << endl;
	//face index starts from 1
	faces -= 1;

	arma::mat landmarksIdxes;
	landmarksIdxes.load(dataDir + "landmarksIdxs73.mat");
	//    cout<<"Landmark indexes loaded: "<<landmarksIdxes.n_rows<<endl;
	tempbody.landmarksIdx = landmarksIdxes;

	tempbody.init(pts, faces);
	tempbody.calcFaces();
	
    isVisible = ones<arColIS>(tempbody.nPoints);
}

/** @brief loadModel
  *
  * @todo: document this function
  */
void fitMesh::loadModel()
{
    //cout<<"loading eigen vectors...\n";
	evectors.load(dataDir + "reduced_evectors.mat");
	cout << "eigen vectors loaded: " << evectors.n_rows << "," << evectors.n_cols << endl;

    //cout<<"loading eigen values...\n";
	evalues.load(dataDir + "evalues.mat");
	evalues = evalues.cols(0, SHAPEPARAM_NUM - 1);
	cout << "eigen values loaded: " << evalues.n_rows << "," << evalues.n_cols << endl;
    shapepose.setEvectors(evectors);
	shapepose.setEvalues(evalues);
}

arma::vec fitMesh::searchShoulder(const arma::mat& model, const uint32_t level, vector<double>& widAvg, vector<double>& depAvg, vector<double>& depMax)
{
	const arma::mat max_m = max(model,0);
	const arma::mat min_m = min(model,0);
    const double height = max_m(2) - min_m(2);
	const double width = max_m(0) - min_m(0);
	const double top = max_m(2);
	const double step = height / level;
    vector<int> lvCnt(level + 1, 0);
    model.each_row([&](const arma::rowvec& col)
    {
		uint8_t level = (uint8_t)((top - col(2)) / step);
		if(col(1) < depMax[level])//compare front(toward negtive)
			depMax[level] = col(1);
		widAvg[level] += abs(col(0));
		depAvg[level] += col(1);
		lvCnt[level]++;
    });
	for (uint32_t a = 0; a <= level; ++a)
    {
        if(lvCnt[a] == 0)
            widAvg[a] = 0;
        else
        {
            widAvg[a] /= lvCnt[a];
            depAvg[a] /= lvCnt[a];
        }
    }
    return arma::vec({width, height, step});
}

arma::mat fitMesh::rigidAlignFix(const arma::mat& input, const arma::mat& R, double& dDepth)
{
	//res = {tObjHei, tAvgDep[a], sAvgDep[a]/sST, tMaxDep[a], sMaxDep[a]/sST};
	arma::vec res(5, arma::fill::ones);
    {
        const uint32_t lv = 64;
		vector<double> sAvgWid(lv + 1, 0), sAvgDep(lv + 1, 0), sMaxDep(lv + 1, 0);
		vector<double> tAvgWid(lv + 1, 0), tAvgDep(lv + 1, 0), tMaxDep(lv + 1, 0);
        arma::vec sret = searchShoulder(input*R, lv, sAvgWid, sAvgDep, sMaxDep);
        arma::vec tret = searchShoulder(tempbody.points, lv, tAvgWid, tAvgDep, tMaxDep);

		const double sST = sret(0) / tret(0);
		const double sWidLow = 0.12*sret(0), sWidHi = 0.2*sret(0);
        const double tWidLow = 0.12*tret(0), tWidHi = 0.2*tret(0);
        for(unsigned int a=0; a <= lv; ++a)
        {
            if(sAvgWid[a] < sWidHi && sAvgWid[a] > sWidLow && tAvgWid[a] < tWidHi && tAvgWid[a] > tWidLow)
            {
                const double sObjHei = sret(1) - (a+0.5)*sret(2), tObjHei = tret(1) - (a+0.5)*tret(2);
                printf("find shoulder in lv %d of %d\n", a, lv);
                printf("sbody\tavgDepth=%f, maxDepth=%f, objHeight=%f\n",
                       sAvgDep[a], sMaxDep[a], sObjHei);
                printf("tbody\tavgDepth=%f, maxDepth=%f, objHeight=%f\n",
                       tAvgDep[a], tMaxDep[a], tObjHei);
                res = {tObjHei, tAvgDep[a], sAvgDep[a]/sST, tMaxDep[a], sMaxDep[a]/sST};
                break;
            }
        }
    }
    dDepth += (res(3) - res(4))/2;
    double tanOri = res(2) / res(0), tanObj = (res(1) + res(3) - res(4)) / res(0);

    double cosx = (1 + tanOri*tanObj)/( sqrt(1+tanOri*tanOri) * sqrt(1+tanObj*tanObj) );
    double sinx = sqrt(1- cosx*cosx);
    printf("rotate: sinx=%f, cosx=%f\n",sinx,cosx);
    arma::mat Rmat(3, 3, arma::fill::zeros);
    Rmat(0,0) = 1;
    Rmat(1,1) = Rmat(2,2) = cosx;
    Rmat(1,2) = sinx; Rmat(2,1) = -sinx;

    return Rmat * R;
}

void fitMesh::rigidAlignTemplate2ScanPCA(CScan& scanbody)
{
    //align the scan points to the template based on PCA, note the coordinates direction
    cout<<"align with pca\n";
    arma::rowvec meanpoints;
    arma::mat Rsc;
    arma::mat eig_vec;
    arma::vec eig_val;
    arma::mat scpoint;
    {
		baseShift = meanpoints = mean(scanbody.points,0);
		scpoint = scanbody.points.each_row() - meanpoints;
        arma::mat sctmppoint = scpoint.t()*scpoint;
        if(! arma::eig_sym(eig_val,eig_vec,sctmppoint))
        {
            cout<<"eigen decomposition failed for scanbody\n";
            return;
        }
        Rsc = eig_vec;
    }

    {
        meanpoints = mean(tempbody.points,0);
        arma::mat tppoint = tempbody.points.each_row() - meanpoints;
        tppoint = tppoint.t()*tppoint;
        if(! arma::eig_sym(eig_val,eig_vec,tppoint))
        {
            cout<<"eigen decomposition failed for tempbody\n";
            return;
        }
    }
    arma::mat R = (eig_vec * Rsc.i()).t();

    double dDepth = 0;
    R = rigidAlignFix(scpoint, R, dDepth);

    //finally rotate scan body
    scpoint *= R;

    const arma::rowvec max_s = arma::max(scpoint, 0);
	const arma::rowvec min_s = arma::min(scpoint, 0);
	const arma::rowvec max_t = arma::max(tempbody.points, 0);
	const arma::rowvec min_t = arma::min(tempbody.points, 0);

   const double s = (max_s(2)-min_s(2)) / (max_t(2)-min_t(2));
    cout<<"the scale is: "<<s<<endl;
    //translate the scan points

    meanpoints = mean(tempbody.points,0);
    meanpoints(1) = dDepth;
    meanpoints(2) += min_t(2) - min_s(2)/s;//align to the foot by moving

    scpoint /= s;
    scanbody.points = scpoint.each_row() + meanpoints;
	{
		rotateMat = R;
		totalScale = s;
		totalShift = meanpoints;
		const arma::mat cpos = (-baseShift) * (R / totalScale);
		camPos.assign(cpos(0), cpos(1), cpos(2));
	}
    arma::mat T(4, 4, arma::fill::zeros);
    T.submat(0,0,2,2) = R/s;
    T(3,3) = 1;
    T.submat(0,3,2,3) = meanpoints.t();
	scanbody.T = T;

    //Translate the landmarks
    if(!scanbody.landmarks.empty())
    {
        //problems,should reindexed from the scanbody.points
        //TODO...
    }
    //Translate the normals
    if(!scanbody.normals.empty())
    {
        scanbody.normals *= R;
    }

	scanbody.prepare();
	scanbody.nntree.init(scanbody.vPts, scanbody.vNorms, scanbody.nPoints);
}
void fitMesh::rigidAlignTemplate2ScanLandmarks()
{
    //align the scan according to the landmarks
}

void fitMesh::DirectRigidAlign(CScan& scan)
{
	scan.points.each_row() -= baseShift;
	//Translate the normals
	if (!scan.normals.empty())
		scan.normals *= rotateMat;
	//finally rotate scan body
	scan.points *= (rotateMat / totalScale);

	scan.points.each_row() += totalShift;
	//prepare nn-tree
	scan.prepare();
	scan.nntree.init(scan.vPts, scan.vNorms, scan.nPoints);
}

void fitMesh::fitShapePose(const CScan& scan, const uint32_t iter, function<tuple<double, bool, bool>(const uint32_t, const uint32_t, const double)> paramer)
{
	//Initialization of the optimizer
	vector<int> idxHand;//the index of hands
	double err = 0;
	uint32_t sumVNN;
	tSPose = tSShape = tMatchNN = 0;
	cSPose = cSShape = cMatchNN = 0;
	/*
	if (useFLANN)
	{
	scan.kdtree = new cv::flann::Index(armaTOcv(scan.points), cv::flann::KDTreeIndexParams(8));
	cout << "cv kd tree build, scan points number: " << pointscv.rows << endl;
	}
	*/
	bool solvedS = false, solvedP = false;
	vector<uint32_t> idxMapper(tempbody.nPoints);
	//Optimization Loop
	for (uint32_t a = 0; a < iter; ++a)
	{
		const auto param = paramer(a, iter, angleLimit);
		sumVNN = updatePoints(scan, std::get<0>(param), idxMapper, isValidNN_, err);
		showResult(scan, false);

		const auto scanCache = shuffleANDfilter(scan.vPts, tempbody.nPoints, &idxMapper[0], isFastCost ? isValidNN_.memptr() : nullptr);
		if (isFastCost)
			shapepose.preCompute(isValidNN_.memptr());
		if (std::get<1>(param))
		{
			cout << "fit pose\n"; 
			solvedP = true;
			solvePose(scanCache, isValidNN_, curMParam, err);
		}
		if (std::get<2>(param))
		{
			cout << "fit shape\n";
			solvedS = true;
			solveShape(scanCache, isValidNN_, curMParam, err);
			if (curFrame > 0)
			{
				for (uint32_t a = 0; a < SHAPEPARAM_NUM; ++a)
					curMParam.shape[a] = (curMParam.shape[a] + bakMParam.shape[a]) * 0.5;
			}
		}
		cout << "========================================\n";
		printArray("pose param", curMParam.pose);
		printArray("shape param", curMParam.shape);
		cout << "----------------------------------------\n";
	}
	sumVNN = updatePoints(scan, angleLimit, idxMapper, isValidNN_, err);
	showResult(scan, false);
	//wait until the window is closed
	cout << "optimization finished\n";
	if (solvedS)
		logger.log(true, "POSE : %d times, %f ms each.\n", cSPose, tSPose / (cSPose * 1000));
	if (solvedP)
		logger.log(true, "SHAPE: %d times, %f ms each.\n", cSShape, tSShape / (cSShape * 1000));
	logger.log(true, "\n\nKNN : %d times, %f ms each.\nFinally valid nn : %d, total error : %f\n", cMatchNN, tMatchNN / cMatchNN, sumVNN, err).flush();
}

void fitMesh::solvePose(const miniBLAS::VertexVec& scanCache, const arColIS& isValidNN, ModelParam& tpParam, const double lastErr)
{
	double *pose = tpParam.pose;

    cout<<"construct problem: pose\n";
	Problem problem;

	if (isFastCost)
	{
		auto *cost_functionEx2 = new ceres::NumericDiffCostFunction<PoseCostFunctorEx2, ceres::CENTRAL, EVALUATE_POINTS_NUM * 3, POSPARAM_NUM>
			(new PoseCostFunctorEx2(&shapepose, tpParam, isValidNN, scanCache));
		problem.AddResidualBlock(cost_functionEx2, NULL, pose);
	}
	else
	{
		auto *cost_functionEx = new ceres::NumericDiffCostFunction<PoseCostFunctorEx, ceres::CENTRAL, EVALUATE_POINTS_NUM * 3, POSPARAM_NUM>
			(new PoseCostFunctorEx(&shapepose, tpParam, isValidNN, scanCache));
		problem.AddResidualBlock(cost_functionEx, NULL, pose);
	}
	if (curFrame == 0)
	{
		auto *reg_function = new ceres::AutoDiffCostFunction<PoseRegularizer, POSPARAM_NUM, POSPARAM_NUM>
			(new PoseRegularizer(1.0 / tempbody.nPoints, POSPARAM_NUM));
		problem.AddResidualBlock(reg_function, NULL, pose);
	}
	else
	{
		auto *soft_function = new ceres::NumericDiffCostFunction<MovementSofter, ceres::CENTRAL, POSPARAM_NUM, POSPARAM_NUM>
			(new MovementSofter(modelParams.back(), sqrt(lastErr / POSPARAM_NUM)));
		problem.AddResidualBlock(soft_function, NULL, pose);
	}
    Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.linear_solver_type = ceres::DENSE_QR;
	options.num_threads = 2;
    options.num_linear_solver_threads = 2;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    
	cout << "solving...\n";
	runcnt.store(0);
	runtime.store(0);
    ceres::Solve(options, &problem, &summary);
	
    cout << summary.BriefReport();
	tSPose += runtime; cSPose += runcnt;
	const double rt = runtime; const uint32_t rc = runcnt;
	logger.log(summary.FullReport()).log(true, "\nposeCost  invoked %d times, avg %f ms\n\n", rc, rt / (rc * 1000));
}

void fitMesh::solveShape(const miniBLAS::VertexVec& scanCache, const arColIS& isValidNN, ModelParam& tpParam, const double lastErr)
{
	double *shape = tpParam.shape;

    Problem problem;
	cout << "construct problem: SHAPE\n";

	shapepose.isFastFitShape = isFastCost;
	if (isFastCost)
	{
		auto cost_functionEx2 = new ceres::NumericDiffCostFunction<ShapeCostFunctorEx2, ceres::CENTRAL, EVALUATE_POINTS_NUM * 3, SHAPEPARAM_NUM>
			(new ShapeCostFunctorEx2(&shapepose, tpParam, isValidNN, scanCache));
		problem.AddResidualBlock(cost_functionEx2, NULL, shape);
	}
	else
	{
		auto cost_functionEx = new ceres::NumericDiffCostFunction<ShapeCostFunctorEx, ceres::CENTRAL, EVALUATE_POINTS_NUM * 3, SHAPEPARAM_NUM>
			(new ShapeCostFunctorEx(&shapepose, tpParam, isValidNN, scanCache));
		problem.AddResidualBlock(cost_functionEx, NULL, shape);
	}
	if (curFrame > 0)
	{
		auto *soft_function = new ceres::NumericDiffCostFunction<ShapeSofter, ceres::CENTRAL, SHAPEPARAM_NUM, SHAPEPARAM_NUM>
			(new ShapeSofter(modelParams.back(), sqrt(lastErr / SHAPEPARAM_NUM)));
		problem.AddResidualBlock(soft_function, NULL, shape);
	}
//    ceres::CostFunction* reg_function = new ceres::AutoDiffCostFunction<ShapeRegularizer,SHAPEPARAM_NUM,SHAPEPARAM_NUM>
//            (new ShapeRegularizer(1.0/tempbody.nPoints,SHAPEPARAM_NUM));
//    problem.AddResidualBlock(reg_function,NULL,shape);

    Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.linear_solver_type = ceres::DENSE_QR;
	options.num_threads = 2;
	options.num_linear_solver_threads = 2;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;

	cout << "solving...\n";
	runcnt.store(0);
	runtime.store(0);
    ceres::Solve(options, &problem, &summary);
    
	cout << summary.BriefReport();
	tSShape += runtime; cSShape += runcnt;
	const double rt = runtime; const uint32_t rc = runcnt;
	logger.log(summary.FullReport()).log(true, "\nshapeCost invoked %d times, avg %f ms\n\n", rc, rt / (rc * 1000));
}

void fitMesh::raytraceCut(miniBLAS::NNResult& res) const
{
	for (uint32_t idx = 0; idx < tempbody.nPoints; ++idx)
	{
		const auto& p = tempbody.vPts[idx];
		const miniBLAS::VertexI vidx(idx, idx, idx, idx);
		Vertex dir = p - camPos;
		const float dist = dir.length();
		dir /= dist;
		for (const auto& f : tempbody.vFaces)
		{
			if (f.intersect(camPos, dir, vidx, dist))//covered
			{
				res.idxs[idx] = 65536;
				break;
			}
		}
	}
}
void fitMesh::nnFilter(const miniBLAS::NNResult& res, arColIS& result, const VertexVec& scNorms, const double angLim)
{
	const float mincos = cos(3.1415926 * angLim / 180);
	auto *__restrict pRes = result.memptr();
	for (uint32_t i = 0; i < tempbody.nPoints; i++)
	{
		const int idx = res.idxs[i];
		if (idx > 65530)//unavailable already
			continue;
		if (res.mthcnts[idx] > 3)//consider cut some link
			if (res.mdists[idx] < res.dists[i])//not mininum
				continue;
		if ((scNorms[idx] % tempbody.vNorms[i]) >= mincos)//satisfy angle limit
			pRes[i] = 1;
	}
}
uint32_t fitMesh::updatePoints(const CScan& scan, const double angLim, vector<uint32_t> &idxs, arColIS &isValidNN_rtn, double &err)
{
	tempbody.updPoints(shapepose.getModelFast(curMParam.shape, curMParam.pose));
	tempbody.calcFaces();
	tempbody.calcNormals();

	uint64_t t1, t2;
	/*
	cv::Mat idxsNNOLD(1, 1, CV_32S); cv::Mat distNNOLD(1, 1, CV_32FC1);
	if(useFLANN)
	{
		t1 = getCurTime();
		cv::Mat cvPointsSM = armaTOcv(tempbody.points);
		scan.kdtree->knnSearch(cvPointsSM, idxsNNOLD, distNNOLD, 1);//the distance is L2 which is |D|^2
		t2 = getCurTime();
		logger.log(true, "cvFLANN uses %lld ms.\n", t2 - t1);
		if (idxsNNOLD.rows != tempbody.nPoints)
		{
			cout << "error of the result of knn search \n";
			getchar();
			return 0;
		}
	}
	*/
	idxs.resize(tempbody.nPoints); uint32_t *__restrict pidxNN = &idxs[0];
	//dist^2 in fact
	arColIS isValidNN(tempbody.nPoints, arma::fill::zeros);
	miniBLAS::NNResult nnres(tempbody.nPoints, scan.nntree.PTCount());
	{
		if(curFrame > 0 && isRayTrace)
			raytraceCut(nnres);

		t1 = getCurTime();
		
		scan.nntree.searchOnAnglePan(nnres, tempbody.vPts, tempbody.vNorms, angLim*1.1f, angleLimit / 2);

		t2 = getCurTime();
		cMatchNN++; tMatchNN += t2 - t1;
		logger.log(true, "avxNN uses %lld ms.\n", t2 - t1);
		memcpy(pidxNN, nnres.idxs, sizeof(int32_t) * tempbody.nPoints);
		nnFilter(nnres, isValidNN, scan.vNorms, angLim);
	}

	uint32_t sumVNN = 0;
	{//caculate total error
		float distAll = 0;
		const auto pValid = isValidNN.memptr();
		for (uint32_t i = 0; i < tempbody.nPoints; ++i)
		{
			if (pValid[i])
			{
				sumVNN++;
				distAll += nnres.dists[i];
			}
		}
		err = distAll;
		printf("valid nn number: %d , total error: %f\n", sumVNN, err);
	}

	FILE *fp = fopen("output.data", "wb");
	if (fp != nullptr)
	{
		uint32_t cnt;
		uint8_t type;
		char name[16] = { 0 };
		{
			fwrite(&camPos, sizeof(Vertex), 1, fp);
		}
		{
			type = 0; fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "scan"); fwrite(name, sizeof(name), 1, fp);
			cnt = scan.nPoints; fwrite(&cnt, sizeof(cnt), 1, fp);
			fwrite(&scan.vPts[0], sizeof(Vertex), cnt, fp);
		}
		{
			type = 0; fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "temp"); fwrite(name, sizeof(name), 1, fp);
			cnt = tempbody.nPoints; fwrite(&cnt, sizeof(cnt), 1, fp);
			fwrite(&tempbody.vPts[0], sizeof(Vertex), cnt, fp);
		}
		/*
		if (useFLANN)
		{
			type = 1; fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "cvFLANN"); fwrite(name, sizeof(name), 1, fp);
			cnt = tempbody.nPoints; fwrite(&cnt, sizeof(cnt), 1, fp);
			fwrite(idxsNNOLD.ptr<int>(0), sizeof(int), cnt, fp);
		}
		*/
		{
			type = 1; fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "avxNN"); fwrite(name, sizeof(name), 1, fp);
			cnt = tempbody.nPoints; fwrite(&cnt, sizeof(cnt), 1, fp);
			fwrite(pidxNN, sizeof(int), cnt, fp);
		}
		{
			type = 1; fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "validKNN"); fwrite(name, sizeof(name), 1, fp);
			int *tmp = new int[tempbody.nPoints];
			auto pValid = isValidNN.memptr();
			for (cnt = 0; cnt < tempbody.nPoints; ++cnt)
				tmp[cnt] = (pValid[cnt] ? idxs[cnt] : 65536);
			fwrite(&cnt, sizeof(cnt), 1, fp);
			fwrite(tmp, sizeof(int), cnt, fp);
			delete[] tmp;
		}
        fclose(fp);
		printf("save KNN data to file successfully.\n");
    }
	isValidNN_rtn = isValidNN;
	return sumVNN;
}
void fitMesh::showResult(const CScan& scan, const bool isNN, const vector<uint32_t>* const idxs)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	//show scan points
	showPoints(cloud, scan.vPts, scan.vNorms, pcl::PointXYZRGB(192, 0, 0));
    if(isNN)
    {
		const auto colors_rd = arma::randi<arma::Mat<uint8_t>>(scan.nPoints, 3, arma::distr_param(0, 255));
        for(uint32_t i=0; i<tempbody.nPoints; i++)
        {
            const auto& cl = colors_rd.row(i);
            pcl::PointXYZRGB tmp_pt(cl(0), cl(1), cl(2));

			const auto& tpP = tempbody.vPts[i];
            tmp_pt.x = tpP.x; tmp_pt.y = tpP.y; tmp_pt.z = tpP.z;
            cloud->push_back(tmp_pt);

            //change the color of closed NN point in scan
			const auto& scP = scan.vPts[(*idxs)[i]];
			tmp_pt.x = scP.x; tmp_pt.y = scP.y; tmp_pt.z = scP.z;
            cloud->push_back(tmp_pt);
        }
    }
    else
    {
		//show templete points
		showPoints(cloud, tempbody.vPts, tempbody.vNorms, pcl::PointXYZRGB(0, 192, 0));
    }
	viewer.showCloud(cloud);
}

void fitMesh::init(const std::string& baseName, const bool isOnce)
{
	mode = isOnce;
	baseFName = baseName;
	isFastCost = yesORno("use fast cost func?");
	angleLimit = isVtune ? 30 : inputNumber("angle limit");
	isRayTrace = yesORno("use ray trace to cut?");
}
void fitMesh::mainProcess()
{
	scanFrames.clear();
	{
		scanFrames.push_back(CScan());
		CScan& firstScan = scanFrames.back();
		loadScan(firstScan);
		rigidAlignTemplate2ScanPCA(firstScan);
		{
			pcl::ModelCoefficients coef;
			coef.values.resize(4);
			coef.values[0] = camPos.x;
			coef.values[1] = camPos.y;
			coef.values[2] = camPos.z;
			coef.values[3] = 30;
			viewer.runOnVisualizationThreadOnce([=](pcl::visualization::PCLVisualizer& v)
			{
				v.addSphere(coef);
			});
		}
		fitShapePose(firstScan, 10, [](const uint32_t cur, const uint32_t iter, const double aLim) 
		{
			/*log(0.65) = -0.431 ===> ratio of angle range: 1.431-->1.0*/
			const double angLim = aLim * (1 - std::log(0.65 + 0.35 * cur / iter));
			return make_tuple(angLim, true, cur > 0);
		});
		modelParams.push_back(curMParam);
		bakMParam = curMParam;
	}
	if (!isVtune)
		getchar();
	if (mode)//only once
		return;
	while (yesORno("fit next frame?"))
	{
		curFrame += 1;
		scanFrames.push_back(CScan());
		CScan& curScan = scanFrames.back();
		if (!loadScan(curScan))
			break;
		//rigid align the scan
		DirectRigidAlign(curScan);
		if (curFrame > 3)
		{
			const auto& p0 = modelParams[curFrame - 4].pose, &p1 = modelParams[curFrame - 3].pose,
				&p2 = modelParams[curFrame - 2].pose, &p3 = modelParams[curFrame - 1].pose;
			for (uint32_t a = 0; a < 3; ++a)
				curMParam.pose[a] += (p2[a] + p3[a] - p0[a] - p1[a]) / (4 + 1);
		}
		fitShapePose(curScan, 4, [](const uint32_t cur, const uint32_t iter, const double aLim)
		{
			/*log(0.65) = -0.431 ===> ratio of angle range: 1.431-->1.0*/
			const double angLim = aLim * (1 - std::log(0.65 + 0.35 * cur / iter));
			const uint32_t obj_iter = std::min(iter - 1, (iter + 3) / 2);
			if (cur == obj_iter)
				return make_tuple(angLim, true, true);
			else
				return make_tuple(angLim, true, false);
		});
		modelParams.push_back(curMParam);
		bakMParam = curMParam;
	}
	{
		FILE *fp = fopen("params.csv", "w");
		fprintf(fp, "MODEL_PARAMS,%d frames,\n,\n", modelParams.size());

		fprintf(fp, "POSE_PARAM,\n,");
		for (uint32_t a = 0; a < POSPARAM_NUM;)
			fprintf(fp, "p%d,", a++);
		fprintf(fp, "\n");
		uint32_t idx = 0;
		for (const auto& par : modelParams)
		{
			fprintf(fp, "f%d,", idx++);
			for (uint32_t a = 0; a < POSPARAM_NUM; ++a)
				fprintf(fp, "%f,", par.pose[a]);
			fprintf(fp, "\n");
		}

		fprintf(fp, ",\nSHAPE_PARAM,\n,");
		for (uint32_t a = 0; a < SHAPEPARAM_NUM;)
			fprintf(fp, "p%d,", a++);
		fprintf(fp, "\n");
		idx = 0;
		for (const auto& par : modelParams)
		{
			fprintf(fp, "f%d,", idx++);
			for (uint32_t a = 0; a < SHAPEPARAM_NUM; ++a)
				fprintf(fp, "%f,", par.shape[a]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	getchar();
}