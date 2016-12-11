#include "fitMesh.h"
#include "solverModel.hpp"

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

using arma::zeros;
using arma::ones;
using arma::mean;
using ceres::Problem;
using ceres::Solver;
using miniBLAS::Vertex;
using miniBLAS::VertexI;
using miniBLAS::VertexVec;
using miniBLAS::VertexIVec;

using PointT = pcl::PointXYZRGBNormal;
using PointCloudT = pcl::PointCloud<PointT>;


bool FastTriangle::intersect(const Vertex& origin, const Vertex& direction, const VertexI& idx, const float dist) const
{
	if (_mm_movemask_epi8(_mm_cmpeq_epi32(idx, pidx)) != 0)//point is one vertex of the triangle
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


void ModelBase::ShowPC(pcl::PointCloud<pcl::PointXYZRGB>& cloud) const
{
	for (uint32_t i = 0; i < nPoints; ++i)
	{
		const auto& pt = vPts[i];
		const auto& clr = vColors[i];
		pcl::PointXYZRGB obj(clr.x, clr.y, clr.z);
		obj.x = pt.x; obj.y = pt.y; obj.z = pt.z;
		cloud.push_back(obj);
	}
}

void ModelBase::ShowPC(pcl::PointCloud<pcl::PointXYZRGB>& cloud, pcl::PointXYZRGB color, const bool isCalcNorm) const
{
	for (uint32_t i = 0; i < nPoints; ++i)
	{
		const Vertex& pt = vPts[i];
		if (isCalcNorm)
		{
			const auto y = vNorms[i].y;
			color.b = (uint8_t)(y * 32 + (y > 0 ? 255 - 32 : 32));
		}
		color.x = pt.x; color.y = pt.y; color.z = pt.z;
		cloud.push_back(color);
	}
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

vector<pcl::Vertices> CTemplate::ShowMesh(pcl::PointCloud<pcl::PointXYZRGB>& cloud)
{
	ShowPC(cloud);
	vector<pcl::Vertices> vts;
	pcl::Vertices vt;
	vt.vertices.resize(3);
	for (const auto& f : vFaces)
	{
		f.pidx.save<3>(&vt.vertices[0]);
		vts.push_back(vt);
	}
	return vts;
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
static VertexVec shuffleANDfilter(const VertexVec& in, const uint32_t cnt, const uint32_t *idx, const int8_t *mask = nullptr)
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
		const auto maxsize = in.size();
		for (uint32_t i = 0; i < cnt; ++i)
		{
			const auto id = idx[i];
			cache[i] = in[id >= maxsize ? 0 : id];
		}
	}
	return cache;
}

void printMat(const char * str, const arma::mat& v)
{
    printf("%s: ",str);
	v.for_each([](const double& val) { printf("%.4e,", val); });
    printf("\n");
}
void printMatAll(const char * str, const arma::mat& m)
{
    printf("matrix %s :\n", str);
    const uint32_t ccnt = m.n_cols;
    m.each_row([&](const arma::rowvec& row)
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
template<uint32_t N>
static void printArray(const char* str, const std::array<double, N>& array)
{
	printf("%s:\t", str);
	for (uint32_t a = 0; a < N; ++a)
		printf("%f ", array[a]);
	printf("\n");
}


ctools fitMesh::tools;
pcl::visualization::CloudViewer fitMesh::viewer("viewer");

fitMesh::fitMesh(const std::string& dir) : dataDir(dir), shapepose(dir + "model.dat")
{
    nSamplePoints = 60000;
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
	PointCloudT cloud_tmp;
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
	const int read_status = pcl::io::loadPLYFile(fname, cloud_tmp);
	printf("read status: %d, load %lld points AND normals.\n", read_status, cloud_tmp.points.size());

	scan.nPoints = cloud_tmp.points.size();
	scan.points_orig.resize(scan.nPoints, 3);
	scan.normals_orig.resize(scan.nPoints, 3);
	scan.vColors.reserve(scan.nPoints);
		
	for (uint32_t idx = 0; idx < scan.nPoints; ++idx)
	{
		const PointT& pt = cloud_tmp.points[idx];
		scan.vColors.push_back(Vertex(pt.r, pt.b, pt.b));
		if (mode)
		{
			scan.points_orig.row(idx) = arma::rowvec({ pt.x, -pt.y, pt.z }) * 1000;
			scan.normals_orig.row(idx) = arma::rowvec({ pt.normal_x, -pt.normal_y, pt.normal_z });
		}
		else
		{
			scan.points_orig.row(idx) = arma::rowvec({ -pt.y, pt.z, -pt.x }) * 1000;
			scan.normals_orig.row(idx) = arma::rowvec({ -pt.normal_y, pt.normal_z, -pt.normal_x });
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
}

/** @brief loadModel
  *
  * @todo: document this function
  */
void fitMesh::loadModel()
{
	//the eigen vetors of the body shape model
	arma::mat evectors, evalues;

	evectors.load(dataDir + "reduced_evectors.mat");
	cout << "eigen vectors loaded: " << evectors.n_rows << "," << evectors.n_cols << endl;
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
	//res = {tObjHei, tAvgDep[a], sAvgDep[a]/sST, tMaxDep[a], sMaxDep[a]/sST};
	dDepth += (res(3) - res(4)) / 2;
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
	if(isShFix)
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

void fitMesh::fitShapePose(const CScan& scan, const std::vector<FitParam>& fitparams)
{
	//Initialization of the optimizer
	vector<int> idxHand;//the index of hands
	double err = 0;
	uint32_t sumVNN;
	tSPose = tSShape = tMatchNN = 0;
	cSPose = cSShape = cMatchNN = 0;
	bool solvedS = false, solvedP = false;
	vector<uint32_t> idxMapper(tempbody.nPoints);
	bakMParam = predMParam = curMParam;

	//Optimization Loop
	for (uint32_t a = 0; a < fitparams.size(); ++a)
	{
		const auto& param = fitparams[a];
		sumVNN = updatePoints(scan, curMParam, param.anglim, idxMapper, isValidNN_, err);
		showResult(scan);

		const auto scanCache = shuffleANDfilter(scan.vPts, tempbody.nPoints, &idxMapper[0], isFastCost ? isValidNN_.data() : nullptr);
		if (isFastCost)
			msmooth = shapepose.preCompute(isValidNN_.data());
		if (param.isSPose)
		{
			cout << "fit pose\n"; 
			solvedP = true;
			solvePose(param.option, scanCache, isValidNN_, err, a);
			//copy first correction of base 6 param
			for (uint32_t b = 0; b < 6; ++b)
				predMParam.pose[b] = curMParam.pose[b];
			//reset bone angle to stay fixed with last frame but allow some changes
			for (uint32_t b = 6; b < POSPARAM_NUM; ++b)
				predMParam.pose[b] = (bakMParam.pose[b] * 4 + curMParam.pose[b]) / 5;
		}
		if (param.isSShape)
		{
			cout << "fit shape\n";
			solvedS = true;
			solveShape(param.option, scanCache, isValidNN_, err);
			if (curFrame > 0)
			{
				for (uint32_t a = 0; a < SHAPEPARAM_NUM; ++a)
					curMParam.shape[a] = (curMParam.shape[a] * 2 + bakMParam.shape[a]) / 3;
			}
			predMParam.shape = curMParam.shape;
		}
	}
	sumVNN = updatePoints(scan, curMParam, angleLimit, idxMapper, isValidNN_, err);
	showResult(scan);
	//wait until the window is closed
	cout << "optimization finished\n";
	if (solvedP)
		logger.log(true, "POSE : %d times, %f ms each.\n", cSPose, tSPose / (cSPose * 1000));
	if (solvedS)
		logger.log(true, "SHAPE: %d times, %f ms each.\n", cSShape, tSShape / (cSShape * 1000));
	logger.log(true, "KNN : %d times, %f ms each.\nFinally valid nn : %d, total error : %f\n\n", cMatchNN, tMatchNN / cMatchNN, sumVNN, err).flush();
}
void fitMesh::fitShapePoseRe(const CScan & scan, const std::vector<FitParam>& fitparams)
{
	double err = 0;
	uint32_t sumVNN;
	tSPose = tSShape = tMatchNN = 0;
	cSPose = cSShape = cMatchNN = 0;
	bool solvedS = false, solvedP = false;
	vector<uint32_t> idxMapper;
	bakMParam = predMParam = curMParam;

	//Optimization Loop
	for (uint32_t a = 0; a < fitparams.size(); ++a)
	{
		const auto& param = fitparams[a];
		sumVNN = updatePointsRe(scan, curMParam, param.anglim, idxMapper, err);
		showResult(scan);

		if (param.isSPose)
		{
			cout << "fit poseRe\n";
			solvedP = true;
			solvePoseRe(param.option, scan, idxMapper, err, a);
			//copy first correction of base 6 param
			for (uint32_t b = 0; b < 6; ++b)
				predMParam.pose[b] = curMParam.pose[b];
			//reset bone angle to stay fixed with last frame but allow some changes
			for (uint32_t b = 6; b < POSPARAM_NUM; ++b)
				predMParam.pose[b] = (bakMParam.pose[b] * 4 + curMParam.pose[b]) / 5;
		}
		if (param.isSShape)
		{
			cout << "fit shapeRe\n";
			solvedS = true;
			solveShapeRe(param.option, scan, idxMapper, err);
			if (curFrame > 0)
			{
				for (uint32_t a = 0; a < SHAPEPARAM_NUM; ++a)
					curMParam.shape[a] = (curMParam.shape[a] * 2 + bakMParam.shape[a]) / 3;
			}
			predMParam.shape = curMParam.shape;
		}
	}
	sumVNN = updatePointsRe(scan, curMParam, angleLimit, idxMapper, err);
	showResult(scan);
	//wait until the window is closed
	cout << "optimization finished\n";
	if (solvedP)
		logger.log(true, "POSE : %d times, %f ms each.\n", cSPose, tSPose / (cSPose * 1000));
	if (solvedS)
		logger.log(true, "SHAPE: %d times, %f ms each.\n", cSShape, tSShape / (cSShape * 1000));
	logger.log(true, "KNN : %d times, %f ms each.\nFinally valid nn : %d, total error : %f\n\n", cMatchNN, tMatchNN / cMatchNN, sumVNN, err).flush();
}
void fitMesh::solvePose(const ceres::Solver::Options& options, const miniBLAS::VertexVec& scanCache, const arColIS& isValidNN, const double lastErr, const uint32_t curiter)
{
	double *pose = curMParam.pose.data();

    cout<<"construct problem: pose\n";
	Problem problem;

	if (curFrame > 0 && curiter > 0)
	{
		auto *cost_functionPred = new ceres::NumericDiffCostFunction<PoseCostFunctorPred, ceres::CENTRAL, EVALUATE_POINTS_NUM, POSPARAM_NUM>
			(isAngWgt ? new PoseCostFunctorPred(&shapepose, curMParam, predMParam, isValidNN, scanCache, weights) :
				new PoseCostFunctorPred(&shapepose, curMParam, predMParam, isValidNN, scanCache));
		problem.AddResidualBlock(cost_functionPred, NULL, pose);
	}
	else if (isFastCost)
	{
		auto *cost_functionEx2 = new ceres::NumericDiffCostFunction<PoseCostFunctorEx2, ceres::CENTRAL, EVALUATE_POINTS_NUM, POSPARAM_NUM>
			(isAngWgt ? new PoseCostFunctorEx2(&shapepose, curMParam, isValidNN, scanCache, msmooth, weights) :
				new PoseCostFunctorEx2(&shapepose, curMParam, isValidNN, scanCache, msmooth));
		problem.AddResidualBlock(cost_functionEx2, NULL, pose);
	}
	else
	{
		auto *cost_function = new ceres::NumericDiffCostFunction<PoseCostFunctor, ceres::CENTRAL, EVALUATE_POINTS_NUM, POSPARAM_NUM>
			(new PoseCostFunctor(&shapepose, curMParam, isValidNN, scanCache));
		problem.AddResidualBlock(cost_function, NULL, pose);
	}

	if(curFrame > 0)
	{
		auto *soft_function = new ceres::NumericDiffCostFunction<MovementSofter, ceres::CENTRAL, POSPARAM_NUM, POSPARAM_NUM>
			(new MovementSofter(modelParams.back(), lastErr));
		problem.AddResidualBlock(soft_function, NULL, pose);
	}
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
void fitMesh::solveShape(const ceres::Solver::Options& options, const miniBLAS::VertexVec& scanCache, const arColIS& isValidNN, const double lastErr)
{
	double *shape = curMParam.shape.data();

    Problem problem;
	cout << "construct problem: SHAPE\n";

	if (isFastCost)
	{
		auto cost_functionEx2 = new ceres::NumericDiffCostFunction<ShapeCostFunctorEx2, ceres::CENTRAL, EVALUATE_POINTS_NUM, SHAPEPARAM_NUM>
			(new ShapeCostFunctorEx2(&shapepose, curMParam, isValidNN, scanCache, msmooth));
		problem.AddResidualBlock(cost_functionEx2, NULL, shape);
	}
	else
	{
		auto cost_function = new ceres::NumericDiffCostFunction<ShapeCostFunctor, ceres::CENTRAL, EVALUATE_POINTS_NUM, SHAPEPARAM_NUM>
			(new ShapeCostFunctor(&shapepose, curMParam, isValidNN, scanCache));
		problem.AddResidualBlock(cost_function, NULL, shape);
	}

	if (curFrame > 0)
	{
		auto *soft_function = new ceres::NumericDiffCostFunction<ShapeSofter, ceres::CENTRAL, SHAPEPARAM_NUM, SHAPEPARAM_NUM>
			(new ShapeSofter(modelParams.back(), sqrt(lastErr / SHAPEPARAM_NUM)));
		problem.AddResidualBlock(soft_function, NULL, shape);
	}
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

void fitMesh::solvePoseRe(const ceres::Solver::Options & options, const CScan &curScan, const std::vector<uint32_t>& idxs, const double lastErr, const uint32_t curiter)
{
	double *pose = curMParam.pose.data();

	cout << "construct problem: pose\n";
	Problem problem;

	if (isReShift && curiter > 0)
	{
		auto *cost_function = new ceres::NumericDiffCostFunction<PoseCostFunctorReShift, ceres::CENTRAL, EVALUATE_POINTS_NUM, POSPARAM_NUM>
			(new PoseCostFunctorReShift(&shapepose, curMParam, curScan.vPts, weights, idxs, curScan.nPoints));
		problem.AddResidualBlock(cost_function, NULL, pose);
	}
	else
	{
		auto *cost_function = new ceres::NumericDiffCostFunction<PoseCostFunctorRe, ceres::CENTRAL, EVALUATE_POINTS_NUM, POSPARAM_NUM>
			(new PoseCostFunctorRe(&shapepose, curMParam, curScan.vPts, weights, idxs, curScan.nPoints));
		problem.AddResidualBlock(cost_function, NULL, pose);
	}
	if (curFrame > 0)
	{
		auto *soft_function = new ceres::NumericDiffCostFunction<MovementSofter, ceres::CENTRAL, POSPARAM_NUM, POSPARAM_NUM>
			(new MovementSofter(modelParams.back(), lastErr));
		problem.AddResidualBlock(soft_function, NULL, pose);
	}
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
void fitMesh::solveShapeRe(const ceres::Solver::Options & options, const CScan& curScan, const std::vector<uint32_t>& idxs, const double lastErr)
{
	double *shape = curMParam.shape.data();

	Problem problem;
	cout << "construct problem: SHAPE\n";

	if (isReShift)
	{
		auto *cost_function = new ceres::NumericDiffCostFunction<ShapeCostFunctorReShift, ceres::CENTRAL, EVALUATE_POINTS_NUM, SHAPEPARAM_NUM>
			(new ShapeCostFunctorReShift(&shapepose, curMParam, curScan.vPts, weights, idxs, curScan.nPoints));
		problem.AddResidualBlock(cost_function, NULL, shape);
	}
	else
	{
		auto *cost_function = new ceres::NumericDiffCostFunction<ShapeCostFunctorRe, ceres::CENTRAL, EVALUATE_POINTS_NUM, SHAPEPARAM_NUM>
			(new ShapeCostFunctorRe(&shapepose, curMParam, curScan.vPts, weights, idxs, curScan.nPoints));
		problem.AddResidualBlock(cost_function, NULL, shape);
	}
	if (curFrame > 0)
	{
		auto *soft_function = new ceres::NumericDiffCostFunction<ShapeSofter, ceres::CENTRAL, SHAPEPARAM_NUM, SHAPEPARAM_NUM>
			(new ShapeSofter(modelParams.back(), sqrt(lastErr / SHAPEPARAM_NUM)));
		problem.AddResidualBlock(soft_function, NULL, shape);
	}
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

void fitMesh::fitFinal(const uint32_t iter, std::function<std::tuple<double, bool, bool>(const uint32_t, const uint32_t, const double)> paramer)
{
	isFastCost = true;
	double err = 0;
	uint32_t sumVNN;
	tSPose = tSShape = tMatchNN = 0;
	cSPose = cSShape = cMatchNN = 0;

	bool solvedS = false, solvedP = false;

	Solver::Options options;
	options.minimizer_type = ceres::TRUST_REGION;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.linear_solver_type = ceres::DENSE_QR;
	options.num_threads = 2;
	options.num_linear_solver_threads = 2;
	options.minimizer_progress_to_stdout = true;
	options.max_num_line_search_direction_restarts = 10;
	options.max_num_consecutive_invalid_steps = 10;

	for (auto& sc : scanFrames)
		sc.nntree.MAXDist2 = 3e3f;
	tempbody.nntree.MAXDist2 = 3e3f;
	const bool isFitRe = yesORno("use reverse-match to solve shape?");

	for (uint32_t a = 0; a < iter; ++a)
	{
		const auto& pret = paramer(a, iter, angleLimit);
		const auto angle = std::get<0>(pret);
		if (std::get<1>(pret))//solveshape
		{
			if (isFitRe)
				solveAllShapeRe(options, angle);
			else
				solveAllShape(options, angle);
			printArray("\nshape param", curMParam.shape);
			cout << "========================================\n";
		}
		else//solvepose
		{
			solveAllPose(options, angle, std::get<2>(pret));
		}
	}

	vector<uint32_t> idxMapper(tempbody.nPoints);
	sumVNN = updatePoints(scanFrames.back(), curMParam, angleLimit, idxMapper, isValidNN_, err);
	showResult(scanFrames.back());
	//wait until the window is closed
	cout << "optimization finished\n";
	logger.log(true, "POSE : %d times, %f ms each.\n", cSPose, tSPose / (cSPose * 1000));
	logger.log(true, "SHAPE: %d times, %f ms each.\n", cSShape, tSShape / (cSShape * 1000));
	logger.log(true, "KNN : %d times, %f ms each.\nFinally valid nn : %d, total error : %f\n\n", cMatchNN, tMatchNN / cMatchNN, sumVNN, err).flush();
}
void fitMesh::solveAllPose(const ceres::Solver::Options& options, const double angLim, const bool dopred)
{
	double err = 0;
	uint32_t sumVNN;
	for (curFrame = 0; curFrame < modelParams.size(); ++curFrame)
	{
		printFrame(curFrame);
		curMParam.pose = modelParams[curFrame].pose;
		if (dopred && curFrame > 0 && curFrame < modelParams.size() - 1)
		{
			predSoftPose(false);
		}
		else
			predMParam.pose = curMParam.pose;

		const auto& curScan = scanFrames[curFrame];
		vector<uint32_t> idxMapper(tempbody.nPoints);
		sumVNN = updatePoints(curScan, curMParam, angLim, idxMapper, isValidNN_, err);
		showResult(curScan);

		const auto scanCache = shuffleANDfilter(curScan.vPts, tempbody.nPoints, &idxMapper[0], isFastCost ? isValidNN_.data() : nullptr);
		msmooth = shapepose.preCompute(isValidNN_.data());

		solvePose(options, scanCache, isValidNN_, err, 1);

		modelParams[curFrame].pose = curMParam.pose;
	}
}
void fitMesh::solveAllShape(const ceres::Solver::Options& options, const double angLim)
{
	uint32_t sumVNN;
	double err = 0;
	vector<uint32_t> idxMapper(tempbody.nPoints);
	double *shape = curMParam.shape.data();
	Problem problem;
	cout << "construct problem: SHAPE\n";

	vector<VertexVec> scanPts;
	scanPts.reserve(modelParams.size());
	for (uint32_t b = 0; b < modelParams.size(); ++b)
	{
		printFrame(b);
		const auto& curScan = scanFrames[b];
		curMParam.pose = modelParams[b].pose;
		//prepare nn-data
		arColIS validNN;
		sumVNN = updatePoints(curScan, curMParam, angLim, idxMapper, validNN, err);
		logger.log(true, "NN matched : %d \n", sumVNN);
		showResult(curScan);
		scanPts.push_back(shuffleANDfilter(curScan.vPts, tempbody.nPoints, &idxMapper[0], nullptr));

		auto cost_function = new ceres::NumericDiffCostFunction<ShapeCostFunctor, ceres::CENTRAL, EVALUATE_POINTS_NUM, SHAPEPARAM_NUM>
			(new ShapeCostFunctor(&shapepose, modelParams[b], validNN, scanPts.back()));
		problem.AddResidualBlock(cost_function, NULL, shape);
	}

	Solver::Summary summary;
	cout << "solving...\n";
	runcnt.store(0); runtime.store(0);
	ceres::Solve(options, &problem, &summary);

	cout << summary.BriefReport();
	tSShape += runtime; cSShape += runcnt;
	const double rt = runtime; const uint32_t rc = runcnt;
	logger.log(summary.FullReport()).log(true, "\nshapeCost invoked %d times, avg %f ms\n\n", rc, rt / (rc * 1000));
}
void fitMesh::solveAllShapeRe(const ceres::Solver::Options& options, const double angLim)
{
	double err = 0;
	double *shape = curMParam.shape.data();
	Problem problem;
	cout << "construct problem: SHAPE\n";

	SimpleTimer timer;
	for (uint32_t b = 0; b < modelParams.size(); ++b)
	{
		printFrame(b);
		vector<uint32_t> idxMapper;
		const auto& curScan = scanFrames[b];
		curMParam.pose = modelParams[b].pose;
		updatePointsRe(curScan, curMParam, angLim, idxMapper, err);

		showResult(curScan);

		auto cost_function = new ceres::NumericDiffCostFunction<ShapeCostFunctorRe, ceres::CENTRAL, EVALUATE_POINTS_NUM, SHAPEPARAM_NUM>
			(new ShapeCostFunctorRe(&shapepose, modelParams[b], curScan.vPts, weights, idxMapper, curScan.nPoints));
		problem.AddResidualBlock(cost_function, NULL, shape);
	}

	Solver::Summary summary;
	cout << "solving...\n";
	runcnt.store(0); runtime.store(0);
	ceres::Solve(options, &problem, &summary);

	cout << summary.BriefReport();
	tSShape += runtime; cSShape += runcnt;
	const double rt = runtime; const uint32_t rc = runcnt;
	logger.log(summary.FullReport()).log(true, "\nshapeCost invoked %d times, avg %f ms\n\n", rc, rt / (rc * 1000));
}

void fitMesh::predictPose()
{
	Solver::Options option;
	option.minimizer_type = ceres::TRUST_REGION;
	option.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	option.linear_solver_type = ceres::DENSE_QR;
	option.minimizer_progress_to_stdout = false;
	option.max_num_line_search_direction_restarts = 10;
	option.max_num_consecutive_invalid_steps = 10;
	option.use_nonmonotonic_steps = false;

	Problem probGlobal, probJoint;
	cout << "construct problem: predict - Pose\n";
	const auto maxcnt = PosePredictor::calcCount(modelParams.size());
	for (uint32_t a = 0; a < 6; ++a)
	{
		auto *cost_function = new ceres::AutoDiffCostFunction<PosePredictor, ceres::DYNAMIC, 3>(new PosePredictor(modelParams, a), maxcnt);
		probGlobal.AddResidualBlock(cost_function, NULL, npp[a]);
	}
	for (uint32_t a = 6; a < POSPARAM_NUM; ++a)
	{
		auto *cost_function = new ceres::AutoDiffCostFunction<JointPredictor, ceres::DYNAMIC, 4>(new JointPredictor(modelParams, a), maxcnt);
		probJoint.AddResidualBlock(cost_function, NULL, npp[a]);
	}
	Solver::Summary summary;

	cout << "solving...\n";
	ceres::Solve(option, &probGlobal, &summary);
	printf("solving global: %d iters, %e --> %e\n", summary.iterations.size(), summary.initial_cost, summary.final_cost);
	//cout << summary.FullReport();
	ceres::Solve(option, &probJoint, &summary);
	printf("solving joints: %d iters, %e --> %e\n", summary.iterations.size(), summary.initial_cost, summary.final_cost);
	//cout << summary.FullReport();
	
	//printf("normal predict: %f,%f,%f,%f,%f,%f\n", curMParam.pose[0], curMParam.pose[1], curMParam.pose[2], curMParam.pose[3], curMParam.pose[4], curMParam.pose[5]);
	for (uint32_t a = 0; a < 6; ++a)
		curMParam.pose[a] = PosePredictor::Calc::calcute(npp[a], curFrame);
	for (uint32_t a = 6; a < POSPARAM_NUM; ++a)
		curMParam.pose[a] = (JointPredictor::Calc::calcute(npp[a], curFrame) + curMParam.pose[a]) / 2;
	//printf("poly-n predict: %f,%f,%f,%f,%f,%f\n", curMParam.pose[0], curMParam.pose[1], curMParam.pose[2], curMParam.pose[3], curMParam.pose[4], curMParam.pose[5]);
	//getchar();
}
void fitMesh::predSoftPose(const bool solveGlobal)
{
	Solver::Options option;
	option.minimizer_type = ceres::TRUST_REGION;
	option.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	option.linear_solver_type = ceres::DENSE_QR;
	option.minimizer_progress_to_stdout = true;
	option.max_num_line_search_direction_restarts = 10;
	option.max_num_consecutive_invalid_steps = 10;
	option.use_nonmonotonic_steps = false;

	Problem probGlobal, probJoint;
	cout << "construct problem: predict - Pose\n";
	const auto maxcnt = PoseAniSofter::calcCount(modelParams.size(), curFrame);
	if (solveGlobal)
	{
		for (uint32_t a = 0; a < 6; ++a)
		{
			auto *cost_function = new ceres::AutoDiffCostFunction<PoseAniSofter, ceres::DYNAMIC, 3>(
				new PoseAniSofter(modelParams, curFrame, a), maxcnt);
			probGlobal.AddResidualBlock(cost_function, NULL, npp[a]);
		}
	}
	for (uint32_t a = 6; a < POSPARAM_NUM; ++a)
	{
		auto *cost_function = new ceres::AutoDiffCostFunction<JointAniSofter, ceres::DYNAMIC, 4>(
			new JointAniSofter(modelParams, curFrame, a), maxcnt);
		probJoint.AddResidualBlock(cost_function, NULL, npp[a]);
	}
	Solver::Summary summary;

	cout << "solving...\n";
	if (solveGlobal)
	{
		ceres::Solve(option, &probGlobal, &summary);
		printf("solving global: %d iters, %e --> %e\n", summary.iterations.size(), summary.initial_cost, summary.final_cost);
		//cout << summary.FullReport();
	}
	ceres::Solve(option, &probJoint, &summary);
	printf("solving joints: %d iters, %e --> %e\n", summary.iterations.size(), summary.initial_cost, summary.final_cost);
	//cout << summary.FullReport();

	//printf("normal predict: %f,%f,%f,%f,%f,%f\n", curMParam.pose[0], curMParam.pose[1], curMParam.pose[2], curMParam.pose[3], curMParam.pose[4], curMParam.pose[5]);
	if (solveGlobal)
	{
		for (uint32_t a = 0; a < 6; ++a)
			predMParam.pose[a] = PoseAniSofter::Calc::calcute(npp[a], curFrame);
	}
	else
		predMParam.pose = modelParams[curFrame].pose;
	for (uint32_t a = 6; a < POSPARAM_NUM; ++a)
		predMParam.pose[a] = JointAniSofter::Calc::calcute(npp[a], curFrame);
	//printf("poly-n predict: %f,%f,%f,%f,%f,%f\n", curMParam.pose[0], curMParam.pose[1], curMParam.pose[2], curMParam.pose[3], curMParam.pose[4], curMParam.pose[5]);
	getchar();
}

void fitMesh::raytraceCut(miniBLAS::NNResult& res) const
{
	for (uint32_t idx = 0; idx < tempbody.nPoints; ++idx)
	{
		const auto& p = tempbody.vPts[idx];
		const VertexI vidx(idx, idx, idx, idx);
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
std::vector<float> fitMesh::nnFilter(const miniBLAS::NNResult& res, arColIS& isValid, const VertexVec& scNorms, const double angLim)
{
	vector<float> weights;
	isValid.swap(arColIS(tempbody.nPoints, 0));
	const float mincos = cos(3.1415926 * angLim / 180), limcos = cos(3.1415926 * angleLimit / 360), cosInv = 1.0 / limcos;
	for (uint32_t i = 0; i < tempbody.nPoints; i++)
	{
		const int idx = res.idxs[i];
		if (idx > 65530)//unavailable already
			continue;
		if (res.mthcnts[idx] > 3)//consider cut some link
			if (res.mdists[idx] < res.dists[i])//not mininum
				continue;
		const auto thecos = scNorms[idx] % tempbody.vNorms[i];
		if (thecos >= mincos)//satisfy angle limit
		{
			isValid[i] = 1;
			weights.push_back(thecos > limcos? 1 : thecos * cosInv);
		}
	}
	return weights;
}
uint32_t fitMesh::updatePoints(const CScan& scan, const ModelParam& mPar, const double angLim, vector<uint32_t> &idxs, arColIS &isValidNN_rtn, double &err)
{
	tempbody.updPoints(shapepose.getModelFast(mPar.shape.data(), mPar.pose.data()));
	tempbody.calcFaces();
	tempbody.calcNormals();

	SimpleTimer timer;
	//dist^2 in fact
	arColIS isValidNN;
	miniBLAS::NNResult nnres(tempbody.nPoints, scan.nntree.PTCount());
	{
		if(curFrame > 0 && isRayTrace)
			raytraceCut(nnres);
		timer.Start();
		scan.nntree.searchOnAnglePan(nnres, tempbody.vPts, tempbody.vNorms, angLim*1.1f, angleLimit / 2);
		timer.Stop();
		cMatchNN++; tMatchNN += timer.ElapseMs();
		logger.log(true, "avxNN uses %lld ms.\n", timer.ElapseMs());
		idxs.resize(tempbody.nPoints);
		memcpy(idxs.data(), nnres.idxs, sizeof(int32_t) * tempbody.nPoints);
		weights = nnFilter(nnres, isValidNN, scan.vNorms, angLim);
	}

	uint32_t sumVNN = 0;
	{//caculate total error
		float distAll = 0;
		for (uint32_t i = 0; i < tempbody.nPoints; ++i)
		{
			if (isValidNN[i])
			{
				sumVNN++;
				distAll += nnres.dists[i];
			}
		}
		err = distAll;
		logger.log(true, "valid nn number: %d , total error: %f\n", sumVNN, err);
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
		{
			type = 1; fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "avxNN"); fwrite(name, sizeof(name), 1, fp);
			cnt = tempbody.nPoints; fwrite(&cnt, sizeof(cnt), 1, fp);
			fwrite(idxs.data(), sizeof(int), cnt, fp);
		}
		{
			type = 1; fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "validKNN"); fwrite(name, sizeof(name), 1, fp);
			int32_t *tmp = new int32_t[tempbody.nPoints];
			for (cnt = 0; cnt < tempbody.nPoints; ++cnt)
				tmp[cnt] = (isValidNN[cnt] ? idxs[cnt] : 65536);
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

uint32_t fitMesh::updatePointsRe(const CScan & scan, const ModelParam & mPar, const double angLim, std::vector<uint32_t>& idxs, double & err)
{
	tempbody.updPoints(shapepose.getModelFast(mPar.shape.data(), mPar.pose.data()));
	tempbody.calcFaces();
	tempbody.calcNormals();
	tempbody.nntree.init(tempbody.vPts, tempbody.vNorms, tempbody.nPoints);

	uint32_t sumVNN = 0;
	SimpleTimer timer;
	miniBLAS::NNResult nnres(scan.nPoints, tempbody.nntree.PTCount());
	timer.Start();
	tempbody.nntree.searchOnAnglePan(nnres, scan.vPts, scan.vNorms, angLim, angleLimit / 2);
	timer.Stop();
	cMatchNN++; tMatchNN += timer.ElapseMs();
	logger.log(true, "avxNN uses %lld ms.\n", timer.ElapseMs());

	{
		weights.swap(vector<float>(scan.nPoints, 0));
		float distAll = 0;
		const float mincos = cos(3.1415926 * angLim / 180), limcos = cos(3.1415926 * angleLimit / 360), cosInv = 1.0 / limcos;
		for (uint32_t i = 0; i < scan.nPoints; i++)
		{
			auto& idx = nnres.idxs[i];
			if (idx > 65530)//unavailable already
			{
				idx = 0;
				continue;
			}
			if (nnres.mthcnts[idx] > 6)//consider cut some link
				if (nnres.mdists[idx] < nnres.dists[i])//not mininum
					continue;// leave idx unchanged since it is legal
			const auto thecos = scan.vNorms[i] % tempbody.vNorms[idx];
			if (thecos >= mincos)//satisfy angle limit
			{
				weights[i] = (thecos > limcos ? 1 : thecos * cosInv);//change weight to non-zero
				sumVNN++;
				distAll += nnres.dists[i];
			}
		}
		idxs.resize(scan.nPoints);
		memcpy(idxs.data(), nnres.idxs, sizeof(int32_t) * scan.nPoints);
		err = distAll;
		logger.log(true, "valid nn number: %d , total error: %f\n", sumVNN, distAll);
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
			strcpy(name, "temp"); fwrite(name, sizeof(name), 1, fp);
			cnt = tempbody.nPoints; fwrite(&cnt, sizeof(cnt), 1, fp);
			fwrite(&tempbody.vPts[0], sizeof(Vertex), cnt, fp);
		}
		{
			type = 0; fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "scan"); fwrite(name, sizeof(name), 1, fp);
			cnt = scan.nPoints; fwrite(&cnt, sizeof(cnt), 1, fp);
			fwrite(&scan.vPts[0], sizeof(Vertex), cnt, fp);
		}
		{
			type = 1; fwrite(&type, sizeof(type), 1, fp);
			strcpy(name, "validKNN"); fwrite(name, sizeof(name), 1, fp);
			int32_t *tmp = new int32_t[scan.nPoints];
			for (cnt = 0; cnt < scan.nPoints; ++cnt)
				tmp[cnt] = (weights[cnt] > 0 ? idxs[cnt] : 65536);
			fwrite(&cnt, sizeof(cnt), 1, fp);
			fwrite(tmp, sizeof(int), cnt, fp);
			delete[] tmp;
		}
		fclose(fp);
		printf("save Re-KNN data to file successfully.\n");
	}
	return sumVNN;
}

void fitMesh::buildModelColor()
{
	if (!yesORno("\nbuild color for model from current scans?"))
		tempbody.vColors = VertexVec(tempbody.nPoints, Vertex(0, 192, 0));
	else
	{
		tempbody.vColors.resize(tempbody.nPoints);
		memset(&tempbody.vColors[0], 0x0, sizeof(Vertex) * tempbody.nPoints);
		vector<float> times(tempbody.nPoints, 0);
		for (uint32_t a = 0; a < scanFrames.size(); ++a)
		{
			printf("calculating frame %d...", a);
			const auto& mp = modelParams[a];
			auto& scan = scanFrames[a];
			tempbody.updPoints(shapepose.getModelFast(mp.shape.data(), mp.pose.data()));
			tempbody.calcFaces();
			tempbody.calcNormals();

			miniBLAS::NNResult nnres(tempbody.nPoints, scan.nntree.PTCount());
			scan.nntree.MAXDist2 = 1e3f;
			scan.nntree.searchOnAnglePan(nnres, tempbody.vPts, tempbody.vNorms, angleLimit, angleLimit / 2);
			{
				const float mincos = cos(3.1415926 * angleLimit / 180), limcos = cos(3.1415926 * angleLimit / 360), cosInv = 1.0 / limcos;
				for (uint32_t i = 0; i < tempbody.nPoints; i++)
				{
					const auto idx = nnres.idxs[i];
					if (idx > 65530)//unavailable already
						continue;
					if (nnres.mthcnts[idx] > 3)//consider cut some link
						if (nnres.mdists[idx] < nnres.dists[i])//not mininum
							continue;
					const auto thecos = scan.vNorms[idx] % tempbody.vNorms[i];
					if (thecos >= mincos)//satisfy angle limit
					{
						const float wgt = thecos > limcos ? 1 : thecos * cosInv;
						const auto& src = scan.vColors[idx];
						tempbody.vColors[i] += Vertex(src.x, src.y, src.z)*wgt;
						times[i] += wgt;
					}
				}
			}
			//end of each frame
			printf("ok.\n");
		}
		for (uint32_t i = 0; i < tempbody.nPoints; i++)
			tempbody.vColors[i] /= times[i];
	}
}

void fitMesh::showResult(const CScan& scan, const bool showScan, const vector<uint32_t>* const idxs) const
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	if (idxs != nullptr)
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
		if (showScan)
			scan.ShowPC(*cloud, pcl::PointXYZRGB(192, 0, 0));
		tempbody.ShowPC(*cloud, pcl::PointXYZRGB(0, 192, 0));
    }
	viewer.showCloud(cloud);
}

void fitMesh::init(const std::string& baseName, const bool isOnce)
{
	mode = isOnce;
	baseFName = baseName;
	//isShFix = yesORno("fix shoulder?");
	//isFastCost = yesORno("use fast cost func?");
	angleLimit = inputNumber("angle limit", 30);
	//isRayTrace = yesORno("use ray trace to cut?");
	//isAngWgt = yesORno("use weight on angles?");
}
void fitMesh::mainProcess()
{
	static Solver::Options option;
	vector<FitParam> fitparams;
	{
		option.minimizer_type = ceres::TRUST_REGION;
		option.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		option.linear_solver_type = ceres::DENSE_QR;
		option.num_threads = 2;
		option.num_linear_solver_threads = 2;
		option.minimizer_progress_to_stdout = true;
		option.max_num_line_search_direction_restarts = 10;
		option.max_num_consecutive_invalid_steps = 10;
	}
	scanFrames.clear();
	setTitle("Reconstructing...");
	{
		scanFrames.push_back(CScan());
		CScan& firstScan = scanFrames.back();
		loadScan(firstScan);
		rigidAlignTemplate2ScanPCA(firstScan);
		{// draw camara
			pcl::ModelCoefficients coef;
			coef.values.resize(4);
			camPos.save<3>(&coef.values[0]);
			coef.values[3] = 30;
			viewer.runOnVisualizationThreadOnce([=](pcl::visualization::PCLVisualizer& v)
			{
				v.addSphere(coef); 
				v.addText("frame0", 100, 0, 12, 1, 0, 0, "frame");
			});
		}
		for (uint32_t a = 0; a < 10; ++a)
			fitparams.push_back(FitParam{ true, a > 0, (1 - std::log(0.65 + 0.35 * a / 10)) * angleLimit, option });
		fitShapePose(firstScan, fitparams);
		modelParams.push_back(curMParam);
		bakMParam = curMParam;
	}
	if (!isVtune)
		getchar();
	if (mode)//only once
		return;
	const bool useRE = yesORno("use RE version of fit?");
	if (useRE)
		isReShift = yesORno("use ReMin for aloowing shift?");
	int leastframe = inputNumber("at least fit to which frame?", 0);
	{
		//prepare fit params
		fitparams.clear();
		const uint32_t totaliter = 4;
		for (uint32_t a = 0; a < totaliter; ++a)
		{
			/*log(0.65) = -0.431 ===> ratio of angle range: 1.431-->1.0*/
			const double angLim = angleLimit * (1 - std::log(0.65 + 0.35 * a / totaliter));
			const uint32_t obj_iter = totaliter - 2;
			fitparams.push_back(FitParam{ true, a == obj_iter, angLim, option });
		}
	}
	memset(npp, 0x0, sizeof(npp));
	while (curFrame < leastframe || yesORno("fit next frame?"))
	{
		printFrame(++curFrame);
		scanFrames.push_back(CScan());
		CScan& curScan = scanFrames.back();
		if (!loadScan(curScan))
			break;
		//rigid align the scan
		DirectRigidAlign(curScan);
		curScan.nntree.MAXDist2 = tempbody.nntree.MAXDist2 = 5e3f;
		if (curFrame > 3)
			predictPose();
		if (useRE)
			fitShapePoseRe(curScan, fitparams);
		else
			fitShapePose(curScan, fitparams);
		modelParams.push_back(curMParam);
	}
	if (curFrame == 0 && yesORno("load previous 9999 params?"))
		readMParamScan(buildName(9999));
	else
		saveMParam(buildName(9999));
	if (yesORno("run final optimization?"))
	{
		memset(npp, 0x0, sizeof(npp));
		setTitle("Final Optimazing...");
		fitFinal(9, [](const uint32_t cur, const uint32_t iter, const double aLim)
		{
			const bool isSolveShape = (cur % 3 != 0);
			const double ang = aLim * (isSolveShape ? (1.2 - cur*0.05) : (1.1 - cur*0.025));
			return make_tuple(ang, isSolveShape, cur == 3);
		});
		printf("final optimization finished.\n");
		modelParams.push_back(curMParam);
		getchar();
		saveMParam(buildName(curFrame));
		for (auto& mp : modelParams)
			mp.shape = curMParam.shape;
	}
	getchar();
}
std::string fitMesh::buildName(const uint32_t frame)
{
	string fname("params_");
	fname += baseFName.back();
	fname += "_f" + std::to_string(frame);
	return fname;
}
bool fitMesh::saveMParam(const std::string& fname)
{
	printf("writing final csv...");
	string csvname = fname + ".csv";
	FILE *fp = fopen(csvname.c_str(), "w");
	if (fp != nullptr)
	{
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
		printf("done\n");
	}
	else
	{
		printf("fail\n");
	}

	printf("writing final dat...");
	string datname = fname + ".dat";
	fp = fopen(datname.c_str(), "wb");
	if (fp != nullptr)
	{
		uint32_t count = scanFrames.size();
		fwrite(&count, sizeof(uint32_t), 1, fp);
		count = modelParams.size();
		fwrite(&count, sizeof(uint32_t), 1, fp);
		fwrite(&modelParams[0], sizeof(ModelParam), count, fp);
		fclose(fp);
		printf("done\n");
		return true;
	}
	else
	{
		printf("fail\n");
		return false;
	}
}
bool fitMesh::readMParamScan(const std::string& fname)
{
	printf("reading final dat...");
	string datname = fname + ".dat";
	FILE *fp = fopen(datname.c_str(), "rb");
	if (fp != nullptr)
	{
		uint32_t scancount;
		fread(&scancount, sizeof(uint32_t), 1, fp);
		uint32_t mpcount;
		fread(&mpcount, sizeof(uint32_t), 1, fp);
		modelParams.resize(mpcount);
		fread(&modelParams[0], sizeof(ModelParam), mpcount, fp);
		fclose(fp);
		printf("done\n");

		while (curFrame + 1 < scancount)
		{
			curFrame++;
			scanFrames.push_back(CScan());
			CScan& curScan = scanFrames.back();
			if (!loadScan(curScan))
				break;
			DirectRigidAlign(curScan);
		}
		return true;
	}
	else
	{
		printf("fail\n");
		return false;
	}
}

void fitMesh::watch(const uint32_t frameCount)
{
	const string fname = buildName(frameCount);
	readMParamScan(fname);
	watch();
}

void fitMesh::watch()
{
	buildModelColor();
	viewer.runOnVisualizationThreadOnce([=](pcl::visualization::PCLVisualizer& v)
	{
		v.setBackgroundColor(0.25, 0.25, 0.25); 
	});
	setTitle("Watching Mode");
	isEnd = false; isAnimate = false; isShowScan = true;
	printf("========= Enter Watch Mode. =========\n");
	printf("space  -animate switch\n");
	printf("return -scan switch\n");
	printf("ESC    -quit\n");
	printf("L/R    -previous/next frame\n");
	printf("\n");
	curFrame = 0;

	auto keyreg = viewer.registerKeyboardCallback([](const pcl::visualization::KeyboardEvent& event, void* pthis) 
	{
		fitMesh& cthis = *(fitMesh*)pthis;
		if (!cthis.isEnd && event.keyUp())
		{
			//printf("press key: %s\n", event.getKeySym().c_str());
			if (!cthis.isAnimate)
			{
				int32_t f = cthis.curFrame;
				if (event.getKeySym() == "Left")
					f = std::max(f - 1, 0);
				else if (event.getKeySym() == "Right")
					f = std::min(int32_t(cthis.scanFrames.size()), f + 1);
				if(cthis.curFrame != f)
					cthis.isRefresh = true;
				cthis.curFrame = f;
			}
			switch (hash_(event.getKeySym()))
			{
			case "Escape"_hash :
				cthis.isEnd = true; break;
			case "space"_hash:
				cthis.isAnimate = !cthis.isAnimate;
				cthis.isRefresh = true; break;
			case "Return"_hash:
				cthis.isShowScan = !cthis.isShowScan;
				cthis.isRefresh = true; break;
			case "Up"_hash:
				cthis.showTMode = ShowMode((uint8_t(cthis.showTMode) + 1) % 4);
				cthis.isRefresh = true; break;
			case "Down"_hash:
				cthis.showTMode = ShowMode((uint8_t(cthis.showTMode) + 3) % 4);
				cthis.isRefresh = true; break;
			}
		}
	}, this);

	bool gap = true;
	while (!isEnd)
	{
		if (isAnimate && gap)
		{
			curFrame = (curFrame + 1) % scanFrames.size();
			isRefresh = true;
		}
		gap = !gap;
		if (isRefresh)
		{
			isRefresh = false;
			showFrame(curFrame);
			sleepMS(18);
		}
		else
			sleepMS(20);
	}
	keyreg.disconnect();
	return;
}
void fitMesh::printFrame(const uint32_t frame)
{
	viewer.runOnVisualizationThreadOnce([=](pcl::visualization::PCLVisualizer& v)
	{
		v.updateText("frame" + std::to_string(frame), 100, 0, 12, 0, 1, 0, "frame");
	});
}
void fitMesh::setTitle(const std::string& title)
{
	viewer.runOnVisualizationThreadOnce([=](pcl::visualization::PCLVisualizer& v)
	{
		v.setWindowName(title);
	});
}

void fitMesh::showFrame(const uint32_t frame)
{
	if (frame >= scanFrames.size())
		return;
	tempbody.updPoints(shapepose.getModelFast(modelParams[frame].shape.data(), modelParams[frame].pose.data()));
	tempbody.calcFaces();
	tempbody.calcNormals();
	printFrame(frame);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	if (isShowScan)
		scanFrames[frame].ShowPC(*cloud);
	switch (showTMode)
	{
	case PointCloud:
		tempbody.ShowPC(*cloud, pcl::PointXYZRGB(0, 192, 0));
		break;
	case ColorCloud:
		tempbody.ShowPC(*cloud);
		break;
	case Mesh:
		viewer.runOnVisualizationThreadOnce([&](pcl::visualization::PCLVisualizer& v)
		{
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cld(new pcl::PointCloud<pcl::PointXYZRGB>);
			auto vts = tempbody.ShowMesh(*cld);
			if (v.contains("tpbody"))
				v.updatePolygonMesh<pcl::PointXYZRGB>(cld, vts, "tpbody");
			else
				v.addPolygonMesh<pcl::PointXYZRGB>(cld, vts, "tpbody");
		});
		break;
	case None:
	default:
		break;
	}
	if(showTMode != Mesh)
		viewer.runOnVisualizationThreadOnce([&](pcl::visualization::PCLVisualizer& v)
	{
		if (v.contains("tpbody"))
			v.removePolygonMesh("tpbody");
	});
	viewer.showCloud(cloud);
}

