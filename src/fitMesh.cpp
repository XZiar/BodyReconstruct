#include "fitMesh.h"

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::isnan;
using std::move;
using std::shared_ptr;
using std::unique_ptr;
using std::make_shared;
using atomic_uint32_t = std::atomic_uint_least32_t;

using arma::zeros;
using arma::ones;
using arma::mean;
using ceres::Problem;
using ceres::Solver;
using miniBLAS::Vertex;

using PointT = pcl::PointXYZRGBNormal;
using PointCloudT = pcl::PointCloud<PointT>;

static const auto deletor = [=](Vertex* ptr) { free_align(ptr); };
static unique_ptr<Vertex[], decltype(deletor)> armaTOcache(const arma::mat in)
{
	const uint32_t cnt = in.n_rows;

	Vertex *__restrict pVert = (Vertex*)malloc_align(sizeof(Vertex) * cnt, 32);
	unique_ptr<Vertex[], decltype(deletor)> cache(pVert, deletor);

	const double *__restrict px = in.memptr(), *__restrict py = px + cnt, *__restrict pz = py + cnt;
	for (uint32_t i = 0; i < cnt; ++i)
		*pVert++ = Vertex(*px++, *py++, *pz++);

	return cache;
}
static unique_ptr<Vertex[], decltype(deletor)> armaTOcache(const arma::mat in, const uint32_t *idx, const uint32_t cnt)
{
	Vertex *__restrict pVert = (Vertex*)malloc_align(sizeof(Vertex) * cnt, 32);
	unique_ptr<Vertex[], decltype(deletor)> cache(pVert, deletor);

	const double *__restrict px = in.memptr(), *__restrict py = in.memptr() + in.n_rows, *__restrict pz = in.memptr() + 2 * in.n_rows;
	for (uint32_t i = 0; i < cnt; ++i)
	{
		const uint32_t off = idx[i];
		pVert[i] = Vertex(px[off], py[off], pz[off]);
	}
	return cache;
}

static pcl::visualization::CloudViewer viewer("viewer");

static atomic_uint32_t runcnt(0), runtime(0);
//Definition of optimization functions
struct PoseCostFunctor
{
private:
	// Observations for a sample.
	const arma::mat shapeParam_;
	const arColIS isValidNN_;
	const cv::Mat idxNN_;
	const arma::mat scanPoints_;
	CShapePose *shapepose_;

public:
	PoseCostFunctor(CShapePose *shapepose, const arma::mat shapeParam, const arColIS isValidNN, const cv::Mat idxNN, const arma::mat scanPoints)
		: shapepose_(shapepose), shapeParam_(shapeParam), isValidNN_(isValidNN), idxNN_(idxNN), scanPoints_(scanPoints)
	{
	}
	//pose is the parameters to be estimated, b is the bias, residual is to return
	bool operator()(const double* pose/*, const double * scale*/, double* residual) const
	{
		uint64_t t1, t2;
		t1 = getCurTimeNS();

		arma::mat pointsSM, jointsSM;
		shapepose_->getModel(shapeParam_.memptr(), pose, pointsSM, jointsSM);
		// pointsSM *= scale[0];

		auto *__restrict pValid = isValidNN_.memptr();
		const uint32_t *__restrict pIdx = idxNN_.ptr<uint32_t>(0);
		for (int j = 0, i = 0; j < idxNN_.rows; ++j, ++pIdx)
		{
			if (*pValid++)
			{
				arma::rowvec delta = scanPoints_.row(*pIdx) - pointsSM.row(j);
				residual[i++] = delta(0);
				residual[i++] = delta(1);
				residual[i++] = delta(2);
			}
			else
			{
				residual[i++] = 0;
				residual[i++] = 0;
				residual[i++] = 0;
			}
		}
		
		runcnt++;
		t2 = getCurTimeNS();
		runtime += (uint32_t)((t2 - t1) / 1000);
		return true;
	}
};

struct PoseCostFunctorEx
{
private:
	// Observations for a sample.
	const arma::mat shapeParam_;
	const arColIS isValidNN_;
	const unique_ptr<Vertex[], decltype(deletor)> scanCache;
	vector<Vertex> basePts;
	CShapePose *shapepose_;

public:
	PoseCostFunctorEx(CShapePose *shapepose, const arma::mat shapeParam, const arColIS isValidNN, const cv::Mat idxNN, const arma::mat scanPoints)
		: shapepose_(shapepose), shapeParam_(shapeParam), isValidNN_(isValidNN), scanCache(armaTOcache(scanPoints, idxNN.ptr<uint32_t>(), idxNN.rows))
	{
		basePts = shapepose_->getBaseModel(shapeParam_.memptr());
	}
	//pose is the parameters to be estimated, b is the bias, residual is to return
	bool operator()(const double* pose/*, const double * scale*/, double* residual) const
	{
		uint64_t t1, t2;
		t1 = getCurTimeNS();

		//const auto pts = shapepose_->getModelFast(shapeParam_.memptr(), pose);
		const auto pts = shapepose_->getModelByPose(basePts, pose);
		auto *__restrict pValid = isValidNN_.memptr();
		for (int j = 0, i = 0; j < isValidNN_.n_elem; ++j)
		{
			if (pValid[j])
			{
				const Vertex delta = scanCache[j] - pts[j];
				residual[i++] = delta.x;
				residual[i++] = delta.y;
				residual[i++] = delta.z;
			}
			else
			{
				residual[i++] = 0;
				residual[i++] = 0;
				residual[i++] = 0;
			}
		}

		runcnt++;
		t2 = getCurTimeNS();
		runtime += (uint32_t)((t2 - t1) / 1000);
		return true;
	}
};

struct ShapeCostFunctorEx
{
private:
	// Observations for a sample.
	const arma::mat poseParam_;
	const arColIS isValidNN_;
	const cv::Mat idxNN_;
	const unique_ptr<Vertex[], decltype(deletor)> scanCache;
	CShapePose *shapepose_;
	double scale_;

public:
	ShapeCostFunctorEx(CShapePose *shapepose, const arma::mat poseParam, const arColIS isValidNN, const cv::Mat idxNN, const arma::mat scanPoints,
		const double scale) : shapepose_(shapepose), poseParam_(poseParam), isValidNN_(isValidNN), idxNN_(idxNN),
		scanCache(armaTOcache(scanPoints, idxNN.ptr<uint32_t>(), idxNN.rows)), scale_(scale)
	{
	}
	//w is the parameters to be estimated, b is the bias, residual is to return
	bool operator() (const double* shape, double* residual) const
	{
		uint64_t t1, t2;
		t1 = getCurTimeNS();

		const auto pts = shapepose_->getModelFast(shape, poseParam_.memptr());
		auto *__restrict pValid = isValidNN_.memptr();
		//float sumerr = 0;
		for (int j = 0, i = 0; j < isValidNN_.n_elem; ++j)
		{
			if (pValid[j])
			{
				const Vertex delta = (scanCache[j] - pts[j]) * scale_;
				residual[i++] = delta.x;
				residual[i++] = delta.y;
				residual[i++] = delta.z;
				//sumerr += delta.length_sqr();
			}
			else
			{
				residual[i++] = 0;
				residual[i++] = 0;
				residual[i++] = 0;
			}
		}
		//printf("\nsumerr:%f\n\n", sumerr);
		runcnt++;
		t2 = getCurTimeNS();
		runtime += (uint32_t)((t2 - t1) / 1000);
		return true;
	}
};

struct ShapeCostFunctor
{
private:
	// Observations for a sample.
	const arma::mat poseParam_;
	const arColIS isValidNN_;
	const cv::Mat idxNN_;
	const arma::mat scanPoints_;
	CShapePose *shapepose_;
	double scale_;

public:
	ShapeCostFunctor(CShapePose *shapepose, const arma::mat poseParam, const arColIS isValidNN, const cv::Mat idxNN, const arma::mat scanPoints,
		const double scale) : shapepose_(shapepose), poseParam_(poseParam), isValidNN_(isValidNN), idxNN_(idxNN), scanPoints_(scanPoints), scale_(scale)
	{
	}
	//w is the parameters to be estimated, b is the bias, residual is to return
	bool operator() (const double* shape,/* const double* scale,*/ double* residual) const
	{
		uint64_t t1, t2;
		t1 = getCurTimeNS();

		arma::mat pointsSM, jointsSM;
		shapepose_->getModel(shape, poseParam_.memptr(), pointsSM, jointsSM);
		pointsSM *= scale_;
		// double s = 0.0;
		//float sumerr = 0;
		for (int j = 0, i = 0; j<idxNN_.rows; j++)
		{
			if (isValidNN_(j))
			{
				arma::rowvec delta = scanPoints_.row(idxNN_.at<int>(j, 0)) - pointsSM.row(j);
				residual[i++] = delta(0);
				residual[i++] = delta(1);
				residual[i++] = delta(2);
				//sumerr += delta(0) + delta(1) + delta(2);
			}
			else
			{
				residual[i++] = 0;
				residual[i++] = 0;
				residual[i++] = 0;
			}
		}
		//printf("\nsumerr:%f\n\n", sumerr);
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
//===============================================================

static bool yesORno(const char *str)
{
    printf("%s(y/n): ", str);
    int key = getchar();
    getchar();
    return key == 'y';
}
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
    if(false)
    {
        for(uint32_t c = 0; c < in.n_cols; ++c)
            for(uint32_t r = 0; r < in.n_rows; ++r)
                if(abs(in(r,c) - out.at<float>(r,c)) > 0.001)
                    printf("error at %d,%d\n",r,c);
    }
    return out;
}

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

/*
double fitMesh::posecost_dlib(dlib::matrix<double, POSPARAM_NUM, 1> pose)
{
    arma::mat pointsSM,jointsSM;
    arma::mat poseParam(1,POSPARAM_NUM);
    for(int i=0;i<POSPARAM_NUM;i++)
    {
        poseParam(0,i)=pose(i);
    }
    shapepose.getModel(tempbody.shape_params, poseParam,pointsSM,jointsSM);

//        pointsSM = pointsSM * scale[0];
    arma::mat delta;
    double residual = 0;
    for(int j=0; j<idxsNN_.rows; j++)
    {
        if(isValidNN_(j))
        {
            delta = scanbody.points.row(idxsNN_.at<int>(j,0))-pointsSM.row(j);
            residual += delta(0,0)*delta(0,0)+delta(0,1)*delta(0,1)+delta(0,2)*delta(0,2);
        }
    }
    return residual;
}

double fitMesh::shapecost_dlib(dlib::matrix<double, SHAPEPARAM_NUM, 1> shape)
{
    arma::mat pointsSM,jointsSM;
    arma::mat shapeParam(1,SHAPEPARAM_NUM);
    for(int i=0;i<SHAPEPARAM_NUM;i++)
    {
        shapeParam(0,i)=shapeParam(i);
    }
    shapepose.getModel(shapeParam, tempbody.pose_params,pointsSM,jointsSM);

//        pointsSM = pointsSM * scale[0];
    arma::mat delta;
    double residual = 0;
    for(int j=0; j<idxsNN_.rows; j++)
    {
        if(isValidNN_(j))
        {
            delta = scanbody.points.row(idxsNN_.at<int>(j,0))-pointsSM.row(j);
            residual += delta(0,0)*delta(0,0)+delta(0,1)*delta(0,1)+delta(0,2)*delta(0,2);
        }
    }
    return residual;
}
*/

CShapePose fitMesh::shapepose;
CTemplate fitMesh::tempbody;

fitMesh::fitMesh()
{
    //ctor
    params.nPCA = SHAPEPARAM_NUM;
    params.nPose = POSPARAM_NUM;
    dataDir = "../BodyReconstruct/data/";
    params.nSamplePoints = 60000;
}

fitMesh::~fitMesh()
{
    //dtor
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
void fitMesh::loadScan()
{
    PointCloudT::Ptr cloud_tmp(new PointCloudT);
    pcl::PolygonMesh mesh;

    string fname = dataDir+"scan";
    printf("###wait for input scan num: ");
    int ret = getchar();
    if(!(ret == 13 || ret == 10))
        getchar();
    if(ret >= '0' && ret <= '9')
        fname += (char)ret;
    fname += ".ply";

    cout<<"loading "<<fname<<endl;
    int read_status = pcl::io::loadPLYFile(fname, *cloud_tmp);
//    pcl::io::loadPLYFile(dataDir+"scan.ply",mesh);
//    int nrows = mesh.polygons.size();
//    pcl::Vertices tmpv = mesh.polygons[1];
//    int ncols = 3;
//    cout<<nrows<<","<<ncols<<endl;
//    arma::mat faces(nrows,ncols);
//    for(int i=0;i<mesh.polygons.size();i++)
//    {
//        pcl::Vertices tmp = mesh.polygons[i];
//        for(int j=0;j<3;j++)
//        {
//            faces(i,j)=tmp.vertices[j];
//        }

//    }
//    cin.ignore();

	printf("read status: %d, load %d points AND normals.\n", read_status, cloud_tmp->points.size());

    PointT tmpPt;
    arma::mat points(cloud_tmp->points.size(),3);
    arma::mat normals(cloud_tmp->points.size(),3);

    uint32_t idx = 0;
    for(const PointT& pt : cloud_tmp->points)//m 2 mm
    {
        points.row(idx) = arma::rowvec({pt.x, -pt.y, pt.z})*1000;
        normals.row(idx) = arma::rowvec({pt.normal_x, -pt.normal_y, pt.normal_z});
        idx++;
    }
    //normalize the normals
    normals = arma::normalise(normals,2,1);

    scanbody.points_orig = points;
    scanbody.normals_orig = normals;
    scanbody.nPoints = points.n_rows;
    //no faces will be loaded, and the normals are calculated when sampling the scan datas
    ///sample the scan points if necessary;
    if(scanbody.points_orig.n_rows <= params.nSamplePoints)//Not sample the scan points
    {
        params.nSamplePoints = scanbody.points_orig.n_rows;
        scanbody.sample_point_idxes.clear();
		for (uint32_t i = 0; i < scanbody.nPoints; i++)
        {
            scanbody.sample_point_idxes.push_back(i);
        }
        scanbody.points = scanbody.points_orig;
        scanbody.normals = scanbody.normals_orig;
    }
    else
    {
		scanbody.sample_point_idxes = tools.randperm(scanbody.nPoints, params.nSamplePoints);
        scanbody.points = arma::zeros(params.nSamplePoints,3);
        scanbody.normals = arma::zeros(params.nSamplePoints,3);
		for (int i = 0; i < params.nSamplePoints; i++)
        {
            scanbody.points.row(i) = scanbody.points_orig.row(scanbody.sample_point_idxes[i]);
            scanbody.normals.row(i) = scanbody.normals_orig.row(scanbody.sample_point_idxes[i]);
        }
        scanbody.nPoints = params.nSamplePoints;
    }

   // arma::mat absn = arma::sqrt(arma::sum(scanbody.normals%scanbody.normals,1));
}

/** @brief loadTemplate
  *
  * @todo: document this function
  */
void fitMesh::loadTemplate()
{
    arma::mat faces;
    faces.load(dataDir+"faces.mat");
    cout<<"Template faces loaded: "<<faces.n_rows<<","<<faces.n_cols<<endl;
    tempbody.faces = faces-1;
	
    arma::mat landmarksIdxes;
    landmarksIdxes.load(dataDir+"landmarksIdxs73.mat");
//    cout<<"Landmark indexes loaded: "<<landmarksIdxes.n_rows<<endl;
    tempbody.landmarksIdx = landmarksIdxes;
//    cout<<"loading mean shape..."<<endl;
    tempbody.meanShape.load(dataDir+"meanShape.mat");
//    cout<<"mean shape loaded, size is: "<<tempbody.meanShape.n_rows<<", "<<tempbody.meanShape.n_cols<<endl;
	tempbody.points = tempbody.meanShape.each_row() - mean(tempbody.meanShape);
	tempbody.shape_params = zeros(1, params.nPCA);
	tempbody.pose_params = zeros(1, params.nPose);
    tempbody.points_idxes.clear();
    tempbody.nPoints = tempbody.points.n_rows;
	//useless, just use an increasing idx
	//for (uint32_t i = 0; i < tempbody.nPoints; i++)
    //    tempbody.points_idxes.push_back(i);
    
	//prepare tpFaceMap
	{
		tpFaceMap.clear();
		tpFaceMap.assign(tempbody.nPoints, UINT32_MAX);
		const double *px = tempbody.faces.memptr(), *py = px + faces.n_rows, *pz = py + faces.n_rows;
		for (uint32_t i = 0; i < faces.n_rows; ++i)
		{
			auto& tx = tpFaceMap[uint32_t(px[i])];
			if (tx == UINT32_MAX)
				tx = i;
			auto& ty = tpFaceMap[uint32_t(py[i])];
			if (ty == UINT32_MAX)
				ty = i; 
			auto& tz = tpFaceMap[uint32_t(pz[i])];
			if (tz == UINT32_MAX)
				tz = i;
		}
	}
    isVisible = ones<arColIS>(tempbody.points.n_rows);
//    arma::mat mshape = mean(tempbody.points);
//    for(int i=0;i<tempbody.nPoints;i++)
//    {
//        if(tempbody.points(i,1)>mshape(0,2)+10)
//        {
//            isVisible(i,0) = 1;
//        }
//    }
}

/** @brief loadModel
  *
  * @todo: document this function
  */
void fitMesh::loadModel()
{
    evectors.load(dataDir+"evectors_bin.mat");
    //evectors.save(dataDir+"evectors_bin.mat",arma::arma_binary);
    cout<<"size of the eigen vectors: "<<evectors.n_rows<<", "<<evectors.n_cols<<endl;
    evectors = evectors.rows(0,params.nPCA-1);
    evectors.save(dataDir+"reduced_evectors.mat",arma::arma_binary);

    cout<<"loading eigen vectors..."<<endl;
    evectors.load(dataDir+"reduced_evectors.mat");
    cout<<"eigen vectors loaded, size is: "<<evectors.n_rows<<","<<evectors.n_cols<<endl;

    cout<<"loading eigen values..."<<endl;
    evalues.load(dataDir+"evalues.mat");
    evalues = evalues.cols(0,params.nPCA-1);
    cout<<"eigen values loaded, size is: "<<evalues.n_rows<<","<<evalues.n_cols<<endl;
    shapepose.setEvectors(evectors);
	shapepose.setEvalues(evalues);
	//printMatAll("evalues", evalues);
}
arma::mat fitMesh::test()
{
    arma::mat points,joints;
    arma::mat shapeParam,poseParam;
    shapeParam = shapeParam.zeros(1,params.nPCA);
    poseParam = poseParam.zeros(1,POSPARAM_NUM);
    shapepose.getModel(shapeParam,poseParam,points,joints);
    return points;
    //cout<<"points:"<<points<<endl;
}
void fitMesh::calculateNormals(const vector<uint32_t> &points_idxes, arma::mat &faces, arma::mat &normals, arma::mat &normals_faces)
{
    //points_idxes is the indexes of the points that sampled frome the scan,how ever the normals are computed from all scan
    //points with triangle faces, so the sampled indexes is needed
	if (faces.n_rows <= 0)
    {
        normals = normals.zeros(tempbody.nPoints,3);
        return;
    }

    normals = normals.zeros(tempbody.nPoints,3);

	uint32_t idx = 0;
    normals.each_row([&](arma::rowvec& row)
    {
		const auto ret = tpFaceMap[idx++];
		if (ret != UINT32_MAX)
			row = normals_faces.row(ret);
		else
			;//row.zeros(1,3);//row = normal_tmp.zeros(1,3);
		/*
        const auto& facesPointIdx = getVertexFacesIdx(points_idxes[idx++], faces);
        if (facesPointIdx.empty())
            ;//row = normal_tmp.zeros(1,3);
        else
            row = normals_faces.row(facesPointIdx[0]);
		*/
    });
}

void fitMesh::calculateNormalsFaces(arma::mat &points, arma::mat &faces, arma::mat &normals_faces)
{
	uint32_t nFaces = faces.n_rows;
    //cout<<"faces number: "<<nFaces<<", "<<points.n_rows<<endl;
	normals_faces.zeros(nFaces, 3);

    //cout<<"normals_faces size: "<<normals_faces.n_rows<<", "<<normals_faces.n_cols<<endl;
    //points.save("pointssm.txt",raw_ascii);
	for (uint32_t i = 0; i < nFaces; i++)
    {
        const arma::rowvec base = points.row(faces(i,0));
        const arma::rowvec edge1 = base - points.row(faces(i,1));
        const arma::rowvec edge2 = base - points.row(faces(i,2));
		normals_faces.row(i) = arma::cross(edge1, edge2);
    }
    //normals_faces.save("notnormals.txt",raw_ascii);
    /*
    arma::mat l = sqrt(sum(normals_faces%normals_faces,1));
    l = arma::repmat(l,1,3);
    normals_faces = normals_faces/l;
    */
	normals_faces = arma::normalise(normals_faces, 2, 1);
}

vector<uint32_t> fitMesh::getVertexFacesIdx(int point_idx, arma::mat &faces)
{
   vector<uint32_t> pointFacesIdx;
   for (uint32_t i = 0; i < faces.n_rows; i++)
   {
       if(faces(i,0)==point_idx)
       {
           pointFacesIdx.push_back(i);
       }
       if(faces(i,1)==point_idx)
       {
           pointFacesIdx.push_back(i);
       }
       if(faces(i,2)==point_idx)
       {
           pointFacesIdx.push_back(i);
       }
   }
   return pointFacesIdx;
}

arma::vec searchShoulder(arma::mat model, const unsigned int lv, vector<double> &widAvg, vector<double> &depAvg, vector<double> &depMax)
{
    arma::mat max_m = max(model,0);
    arma::mat min_m = min(model,0);
    double height = max_m(2) - min_m(2);
    double width = max_m(0) - min_m(0);
    double top = max_m(2);
    double step = height / lv;
    vector<int> lvCnt(lv + 1, 0);
    model.each_row([&](const arma::rowvec& col)
    {
       unsigned int level = (unsigned int)((top - col(2)) / step);
       if(col(1) < depMax[level])//compare front(toward negtive)
           depMax[level] = col(1);
       widAvg[level] += abs(col(0));
       depAvg[level] += col(1);
       lvCnt[level]++;
    });
    for(unsigned int a=0; a <= lv; ++a)
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
	arma::vec res(5, arma::fill::ones);
    {
        const int lv = 64;
        vector<double> sAvgWid(lv+1,0), sAvgDep(lv+1,0), sMaxDep(lv+1,0);
        vector<double> tAvgWid(lv+1,0), tAvgDep(lv+1,0), tMaxDep(lv+1,0);
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
            //printf("^mismatch lv %d : %f AND %f\n", a, sAvgWid[a], tAvgWid[a]);
        }
    }

    printMat("#Shoulder-res:",res);
    dDepth += (res(3) - res(4))/2;
    double tanOri = res(2) / res(0), tanObj = (res(1) + res(3) - res(4)) / res(0);

    double cosx = (1 + tanOri*tanObj)/( sqrt(1+tanOri*tanOri) * sqrt(1+tanObj*tanObj) );
    double sinx = sqrt(1- cosx*cosx);
    printf("rotate: sinx=%7f, cosx=%7f\n",sinx,cosx);
    arma::mat Rmat(3, 3, arma::fill::zeros);
    Rmat(0,0) = 1;
    Rmat(1,1) = Rmat(2,2) = cosx;
    Rmat(1,2) = sinx; Rmat(2,1) = -sinx;

    return Rmat * R;
}

void fitMesh::rigidAlignTemplate2ScanPCA()
{
    //align the scan points to the template based on PCA, note the coordinates direction
    cout<<"align with pca\n";
    arma::rowvec meanpoints;
    arma::mat Rsc;
    arma::mat eig_vec;
    arma::vec eig_val;
    arma::mat scpoint;

    {
        meanpoints = mean(scanbody.points,0);
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
    if(yesORno("do fix"))
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
    printMat("###after YZfix, shift-meanpoints\n",meanpoints);

    scpoint /= s;
    scanbody.points = scpoint.each_row() + meanpoints;

    //rigid align the scan points to the template
    arma::mat T(4, 4, arma::fill::zeros);
    T.submat(0,0,2,2) = R/s;
    T(3,3) = 1;
    T.submat(0,3,2,3) = meanpoints.t();
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
        //saveMat("scan_after.txt", scanbody.normals);
    }
    scanbody.T = T;

	scanbody.nntree.init(scanbody.points);

	isFastCost = yesORno("use fast cost func?");

    //show the initial points
    if(true)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        arma::mat pointsSM,jointsSM;
        shapepose.getModel(tempbody.shape_params,tempbody.pose_params,pointsSM,jointsSM);

        arma::mat normalsSM, normals_faces;
        calculateNormalsFaces(pointsSM,tempbody.faces,normals_faces);
        calculateNormals(tempbody.points_idxes,tempbody.faces,normalsSM,normals_faces);

		showPoints(cloud, tempbody.points, normalsSM, pcl::PointXYZRGB(0, 192, 0));

		showPoints(cloud, scanbody.points, scanbody.normals, pcl::PointXYZRGB(192, 0, 0));
        //pcl::visualization::CloudViewer viewer1("cloud");
        viewer.showCloud(cloud);

    }

}
void fitMesh::rigidAlignTemplate2ScanLandmarks()
{
    //align the scan according to the landmarks
}
void fitMesh::mainProcess()
{
    //The main process of the fitting procedue
    loadScan();
    loadModel();
    loadTemplate();
    rigidAlignTemplate2ScanPCA();
    angleLimit = inputNumber("angle limit");
    fitModel();
}
void fitMesh::fitModel()
{
    //Fit the model to the scan data
    fitShapePose();

}
void fitMesh::fitShapePose()
{
    //Initialization of the optimizer
    vector<int> idxHand;//the index of hands
    scale = 1;// the scale of the template
    double eps_err = 1e-3;
    double errPrev = 0;
    double err=0;

	cv::Mat pointscv = armaTOcv(scanbody.points);
	if (false)
	{
		cv::flann::KDTreeIndexParams indexParams(8);
		scanbody.kdtree = new cv::flann::Index(pointscv, indexParams);
		cout << "cv kd tree build, scan points number: " << pointscv.rows << endl;
	}

	updatePoints(idxsNN_, isValidNN_, scale, err);
    showResult(false);
	errPrev = err + eps_err + 1;
    //Optimization Loop
   // while(fabs(err-errPrev)>eps_err)
    for(int idx=0;idx<10;idx++)
    {
        errPrev = err;//update the error
		cout << "fit pose\n";
        solvePose(idxsNN_,isValidNN_,tempbody.pose_params,tempbody.shape_params,scale);
        //solvePose_dlib();
		cout << "fit shape\n";
        solveShape(idxsNN_,isValidNN_,tempbody.pose_params,tempbody.shape_params,scale);
        //solveShape_dlib();
		cout << "========================================\n";
		updatePoints(idxsNN_, isValidNN_, scale, err);
        showResult(false);
		cout << tempbody.pose_params << endl;
		cout << tempbody.shape_params << endl;
		cout << "----------------------------------------\n";
    }
    //showResult(false);
    //wait until the window is closed
	cout << "optimization finished, close the window to quit\n";
    while(!viewer.wasStopped())
    {
        sleep(1);
    }

//        [poseParams, ~] = fmincon(@PoseFunc,poseParams,[],[],[],[],poseLB,poseUB,[],options);
//        [pointsSM, ~] = shapepose(poseParams(1:end-1),shapeParams(1:end-1),evectors,modelDir);
//        sc = poseParams(end);
//        pointsSM = sc * pointsSM;
//        idxsNN = knnsearch(scan.points,pointsSM);

//        % check the angle between normals
//        normalsScan = normalsScanAll(idxsNN,:);
//        normalsSM = getNormals(pointsSM, template.faces);
//        normalsSM = getNormals1Face(1:nPoints,template.faces,normalsSM);
//        isValidNN = checkAngle(normalsScan,normalsSM,threshNormAngle);

//        % do not register open to closed hands
//        isValidNN(idxHand) = 0;

//        if (~isempty(template.idxsUse))
//            % use only subset of vertices
//            isValidNN = isValidNN.*template.idxsUse;
//        end
//        fprintf('sum(isValidNN): %1.1f\n',sum(isValidNN));
//        if (bFitShape)
//            fprintf('fit shape\n');
//            shapeParams(end) = sc;
//            [shapeParams, ~] = fmincon(@ShapeFunc, shapeParams,[],[],[],[],shapeLB,shapeUB,[],options);
//            % new model code
//            [pointsSM, ~] = shapepose(poseParams(1:end-1),shapeParams(1:end-1),evectors,modelDir);
//            sc = shapeParams(end);
//            pointsSM = sc * pointsSM;
//            [idxsNN, distNN] = knnsearch(scan.points,pointsSM);

//            % check the angle between normals
//            normalsScan = normalsScanAll(idxsNN,:);
//            normalsSM = getNormals(pointsSM,template.faces);
//            normalsSM = getNormals1Face(1:nPoints,template.faces,normalsSM);
//            isValidNN = checkAngle(normalsScan,normalsSM,threshNormAngle);

//            % do not register open to closed hands
//            isValidNN(idxHand) = 0;

//            if (~isempty(template.idxsUse))
//                % use only subset of vertices
//                isValidNN = isValidNN.*template.idxsUse;
//            end
//        end

//        distAll = distNN' * isValidNN;
//        err = distAll / sum(isValidNN);

//        poseParams(end) = sc;
//    end
}

/** @brief checkAngle
  * @param angle_thres max angle(in degree) between two norms
  */
arColIS fitMesh::checkAngle(const arma::mat &normals_knn, const arma::mat &normals_tmp, const double angle_thres)
{
    const uint32_t rowcnt = normals_tmp.n_rows;
    arColIS result(rowcnt, arma::fill::ones);
    if(normals_knn.empty() || normals_tmp.empty())
    {
        return result;
    }

    //saveMat("norm_knn.txt", normals_knn);

    const double mincos = cos(3.1415926 * angle_thres/180);
    arma::vec theta = sum(normals_knn % normals_tmp, 1);
	auto *__restrict pRes = result.memptr();
	const double *__restrict pTheta = theta.memptr();
	for (uint32_t i = 0; i < rowcnt; i++)
    {
		*pRes++ = *pTheta++ >= mincos ? 1 : 0;
    }
    return result;
}

void fitMesh::solvePose(const cv::Mat& idxNN, const arColIS& isValidNN, arma::mat &poseParam, const arma::mat &shapeParam, double &scale)
{
	double *pose = poseParam.memptr();

    cout<<"construct problem: pose\n";
	Problem problem;

	if (isFastCost)
	{
		auto *cost_functionEx = new ceres::NumericDiffCostFunction<PoseCostFunctorEx, ceres::CENTRAL, 6449 * 3, POSPARAM_NUM>
			(new PoseCostFunctorEx(&shapepose, shapeParam, isValidNN, idxNN, scanbody.points));
		problem.AddResidualBlock(cost_functionEx, NULL, pose);
	}
	else
	{
		auto *cost_function = new ceres::NumericDiffCostFunction<PoseCostFunctor, ceres::CENTRAL, 6449 * 3, POSPARAM_NUM>
			(new PoseCostFunctor(&shapepose, shapeParam, isValidNN, idxNN, scanbody.points));
		problem.AddResidualBlock(cost_function, NULL, pose);
	}

	auto *reg_function = new ceres::AutoDiffCostFunction<PoseRegularizer, POSPARAM_NUM, POSPARAM_NUM>
		(new PoseRegularizer(1.0 / tempbody.nPoints, POSPARAM_NUM));
    problem.AddResidualBlock(reg_function,NULL,pose);

    Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.num_linear_solver_threads = 4;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    
	cout << "solving...\n";
	runcnt.store(0);
	runtime.store(0);
    ceres::Solve(options, &problem, &summary);

    cout << summary.BriefReport() << "\n";
	const double rt = runtime; const uint32_t rc = runcnt;
	printf("poseCost invoked %d times, avg %f ms\n", rc, rt / (rc * 1000));
}

void fitMesh::solveShape(const cv::Mat &idxNN, const arColIS &isValidNN, const arma::mat &poseParam, arma::mat &shapeParam,double &scale)
{
	double *shape = shapeParam.memptr();

    Problem problem;
	cout << "construct problem: SHAPE\n";

	if (isFastCost)
	{
		auto cost_functionEx = new ceres::NumericDiffCostFunction<ShapeCostFunctorEx, ceres::CENTRAL, 6449 * 3, SHAPEPARAM_NUM>
			(new ShapeCostFunctorEx(&shapepose, poseParam, isValidNN, idxNN, scanbody.points, scale));
		problem.AddResidualBlock(cost_functionEx, NULL, shape);
	}
	else
	{
		auto cost_function = new ceres::NumericDiffCostFunction<ShapeCostFunctor, ceres::CENTRAL, 6449 * 3, SHAPEPARAM_NUM>
			(new ShapeCostFunctor(&shapepose, poseParam, isValidNN, idxNN, scanbody.points, scale));
		problem.AddResidualBlock(cost_function, NULL, shape);
	}
//    ceres::CostFunction* reg_function = new ceres::AutoDiffCostFunction<ShapeRegularizer,SHAPEPARAM_NUM,SHAPEPARAM_NUM>
//            (new ShapeRegularizer(1.0/tempbody.nPoints,SHAPEPARAM_NUM));
//    problem.AddResidualBlock(reg_function,NULL,shape);

    Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.num_linear_solver_threads = 4;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;

	cout << "solving...\n";
	runcnt.store(0);
	runtime.store(0);
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << "\n";
	const double rt = runtime; const uint32_t rc = runcnt;
	printf("shapeCost invoked %d times, avg %f ms\n", rc, rt / (rc * 1000));

    cout<<"estimated shape params: ";
	for (int i = 0; i < SHAPEPARAM_NUM; i++)
    {
        cout<<shape[i]<<" ";
    }
	printf("scale %f\n", scale);
}

void fitMesh::updatePoints(cv::Mat &idxsNN_rtn, arColIS &isValidNN_rtn, double &scale, double &err)
{
	arma::mat pointsSM;

	auto pts = shapepose.getModelFast(tempbody.shape_params.memptr(), tempbody.pose_params.memptr());
	{
		pointsSM.resize(pts.size(), 3);
		auto *px = pointsSM.memptr(), *py = px + pts.size(), *pz = py + pts.size();
		for (uint32_t a = 0; a < pts.size(); ++a)
		{
			*px++ = pts[a].x;
			*py++ = pts[a].y;
			*pz++ = pts[a].z;
		}
	}
	pointsSM = pointsSM * scale;
	tempbody.points = pointsSM;
	arma::mat normalsSM, normals_faces;
	calculateNormalsFaces(pointsSM, tempbody.faces, normals_faces);
	calculateNormals(tempbody.points_idxes, tempbody.faces, normalsSM, normals_faces);

	uint64_t t1, t2;
	cv::Mat idxsNNOLD(1, 1, CV_32S);
	cv::Mat distNNOLD(1, 1, CV_32FC1);
	if(useFLANN)
	{
		t1 = getCurTime();
		cv::Mat cvPointsSM = armaTOcv(pointsSM);
		scanbody.kdtree->knnSearch(cvPointsSM, idxsNNOLD, distNNOLD, 1);//the distance is L2 which is |D|^2
		cv::sqrt(distNNOLD, distNNOLD);
		t2 = getCurTime();
		printf("cvFLANN uses %lld ms.\n", t2 - t1);
	}

	cv::Mat idxsNN(tempbody.nPoints, 1, CV_32S);
	cv::Mat distNN(tempbody.nPoints, 1, CV_32FC1);
	{
		t1 = getCurTime();

		auto tpPoints = armaTOcache(pointsSM);
		scanbody.nntree.search(tpPoints.get(), tempbody.nPoints, idxsNN.ptr<int>(0), distNN.ptr<float>(0));

		t2 = getCurTime();
		printf("sse4NN uses %lld ms.\n", t2 - t1);
	}

	if (idxsNN.rows != tempbody.nPoints)
	{
		cout << "error of the result of knn search \n";
		return;
	}

	//get the normal of the 1-NN point in scan data
	arma::mat normals_knn(tempbody.nPoints, 3, arma::fill::zeros);
	const int *__restrict pNN = idxsNN.ptr<int>(0);
	for (uint32_t i = 0; i < tempbody.nPoints; i++, ++pNN)
	{
		const int idx = *pNN;
		if (idx < 0 || idx >= scanbody.nPoints)
		{
			cout << "error idx when copy knn normal\n";
			cin.ignore();
			continue;
		}
		normals_knn.row(i) = scanbody.normals.row(idx);//copy the nearest row
	}

	//chech angles between norms of (the nearest point of scan) AND (object point of tbody)
	arColIS isValidNN = checkAngle(normals_knn, normalsSM, angleLimit);
	//    for(uint32_t i=0;i<isValidNN.n_rows;i++)
	//    {
	//        if(distNN.at<float>(i,0)>480000)
	//        {
	//            isValidNN(i,0)=0;
	//        }
	//    }
	isValidNN = isValidNN % isVisible;
	
	{
		uint32_t sum = 0;
		auto *pValid = isValidNN.memptr();
		for (uint32_t a = 0; a++ < isValidNN.n_rows; ++pValid)
			sum += *pValid;
		printf("valid nn number is: %d\n", sum);
	}
	//    do not register open to closed hands
	//    isValidNN(idxHand) = 0;
	
	{//caculate total error
		float distAll = 0;
		auto pValid = isValidNN.memptr();
		const float *pDist = distNN.ptr<float>(0);
		for (uint32_t i = 0; i < tempbody.nPoints; ++i, ++pValid, ++pDist)
		{
			if (*pValid && !isnan(*pDist))
				distAll += *pDist;
		}
		err = distAll;
		cout << "the error is:" << err << endl;
	}

	//if(false)
	{
		FILE *fp = fopen("output.data", "w");
		uint32_t cnt;
		uint8_t type;
		char name[16] = { 0 };
		{
			type = 0;
			fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "scan");
			fwrite(name, sizeof(name), 1, fp);
			cnt = scanbody.nPoints;
			fwrite(&cnt, sizeof(cnt), 1, fp);
			auto sc = armaTOcache(scanbody.points);
			fwrite(sc.get(), sizeof(Vertex), cnt, fp);
		}
		{
			type = 0;
			fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "temp");
			fwrite(name, sizeof(name), 1, fp);
			cnt = tempbody.nPoints;
			fwrite(&cnt, sizeof(cnt), 1, fp);
			auto tp = armaTOcache(tempbody.points);
			fwrite(tp.get(), sizeof(Vertex), cnt, fp);
		}
		if (useFLANN)
		{
			type = 1;
			fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "cvFLANN");
			fwrite(name, sizeof(name), 1, fp);
			cnt = tempbody.nPoints;
			fwrite(&cnt, sizeof(cnt), 1, fp);
			fwrite(idxsNNOLD.ptr<int>(0), sizeof(int), cnt, fp);
		}
		{
			type = 1;
			fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "sse4KNN");
			fwrite(name, sizeof(name), 1, fp);
			cnt = tempbody.nPoints;
			fwrite(&cnt, sizeof(cnt), 1, fp);
			fwrite(idxsNN.ptr<int>(0), sizeof(int), cnt, fp);
		}
		{
			type = 1;
			fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "validKNN");
			fwrite(name, sizeof(name), 1, fp);
			int *tmp = new int[tempbody.nPoints];
			const int *pIdx = idxsNN.ptr<int>(0);
			auto pValid = isValidNN.memptr();
			for (cnt = 0; cnt < tempbody.nPoints; ++pIdx, ++pValid)
				tmp[cnt++] = (*pValid ? *pIdx : 65536);
			fwrite(&cnt, sizeof(cnt), 1, fp);
			fwrite(tmp, sizeof(int), cnt, fp);
			delete[] tmp;
		}
        fclose(fp);
		printf("save KNN data to file successfully.\n");
    }

	idxsNN_rtn = idxsNN;
	isValidNN_rtn = isValidNN;
}
void fitMesh::showResult(bool isNN=false)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	showPoints(cloud, scanbody.points, scanbody.normals, pcl::PointXYZRGB(192, 0, 0));

    arma::mat pointsSM,jointsSM;
	shapepose.getModel(tempbody.shape_params, tempbody.pose_params, pointsSM, jointsSM);
	pointsSM *= scale;

    if(isNN==true)
    {
		arma::Mat<uint8_t> colors_rd = arma::randi<arma::Mat<uint8_t>>(scanbody.points.n_rows, 3, arma::distr_param(0, 255));
        for(uint32_t i=0; i<tempbody.points.n_rows; i++)
        {
            const arma::Row<uint8_t>& cl = colors_rd.row(i);
            pcl::PointXYZRGB tmp_pt(cl(0), cl(1), cl(2));

            const arma::rowvec& tpP = pointsSM.row(i);
            tmp_pt.x = tpP(0);
            tmp_pt.y = tpP(1);
            tmp_pt.z = tpP(2);
            cloud->push_back(tmp_pt);
            //change the color of closed NN point in scan
            int nnidx = idxsNN_.at<int>(i,0);
            const arma::rowvec& scP = scanbody.points.row(nnidx);
            tmp_pt.x = scP(0);
            tmp_pt.y = scP(1);
            tmp_pt.z = scP(2);
            cloud->push_back(tmp_pt);
        }
    //    arma::mat pointsorig,jointsorig;
    //    shapepose.getModel(arma::zeros(1,SHAPEPARAM_NUM),tempbody.pose_params,pointsorig,jointsorig);
    //    for(int i=0;i<tempbody.points.n_rows;i++)
    //    {
    //        tmp_pt.x = pointsorig(i,0);
    //        tmp_pt.y = pointsorig(i,1);
    //        tmp_pt.z = pointsorig(i,2);
    //        tmp_pt.r = 0;
    //        tmp_pt.g = 255;
    //        tmp_pt.b = 128;
    //        cloud->push_back(tmp_pt);
    //    }
        //
        viewer.showCloud(cloud);
    //    cout<<"close the window to continue"<<endl;
    //    while(!viewer.wasStopped())
    //    {
    //        sleep(1);
    //    }
    }
    else
    {
        arma::mat normalsSM, normals_faces;
		calculateNormalsFaces(pointsSM, tempbody.faces, normals_faces);
		calculateNormals(tempbody.points_idxes, tempbody.faces, normalsSM, normals_faces);

		showPoints(cloud, pointsSM, normalsSM, pcl::PointXYZRGB(0, 192, 0));
//        arma::mat pointsorig,jointsorig;
//        shapepose.getModel(arma::zeros(1,SHAPEPARAM_NUM),tempbody.pose_params,pointsorig,jointsorig);
//        arma::mat mshape = mean(pointsorig);

//        for(int i=0;i<tempbody.points.n_rows;i++)
//        {
//            tmp_pt.x = pointsorig(i,0);
//            tmp_pt.y = pointsorig(i,1);
//            tmp_pt.z = pointsorig(i,2);
//            if(tmp_pt.y<mshape(0,2)+10)
//            {
//                tmp_pt.r = 0;
//                tmp_pt.g = 255;
//                tmp_pt.b = 0;
//            }
//            else
//            {
//                tmp_pt.r = 255;
//                tmp_pt.g = 0;
//                tmp_pt.b = 0;
//            }
//            cloud->push_back(tmp_pt);
//        }
        //
        viewer.showCloud(cloud);
        //    cout<<"close the window to continue"<<endl;
        //    while(!viewer.wasStopped())
        //    {
        //        sleep(1);
        //    }
    }
}

/*
void fitMesh::solvePose_dlib()
{
    dlib::matrix<double,POSPARAM_NUM,1> pose;
    for(int i=0;i<POSPARAM_NUM;i++)
    {
        pose(i)=tempbody.pose_params(i);
    }
    dlib::find_min_using_approximate_derivatives(dlib::bfgs_search_strategy(),
                                                 dlib::objective_delta_stop_strategy(1e-7),posecost_dlib,pose,-1);
    for(int i=0;i<POSPARAM_NUM;i++)
    {
        tempbody.pose_params(i)=pose(i);
    }
}

void fitMesh::solveShape_dlib()
{
    dlib::matrix<double,SHAPEPARAM_NUM,1> shape;
    for(int i=0;i<SHAPEPARAM_NUM;i++)
    {
        shape(i)=tempbody.shape_params(i);
    }
    dlib::find_min_using_approximate_derivatives(dlib::bfgs_search_strategy(),
                                                 dlib::objective_delta_stop_strategy(1e-7),shapecost_dlib,shape,-1);
    for(int i=0;i<SHAPEPARAM_NUM;i++)
    {
        tempbody.shape_params(i)=shape(i);
    }
}
*/