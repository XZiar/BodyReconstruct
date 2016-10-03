#include "tools.h"

using std::vector;
using std::cout;
using std::endl;

ctools::ctools()
{
    //ctor
}

ctools::~ctools()
{
    //dtor
}
///估计平面内刚性变换
arma::mat ctools::calcTrans2D(arma::mat src, arma::mat dst)
{
    //src and dst are mat of size 1*numpt*2
    arma::mat lambda;
    lambda=0.00001*lambda.eye(4,4);//shrinkage matrix
    arma::mat H;
    H=H.zeros(src.n_cols,4);
    dst = dst.t();
	for (uint32_t i = 0; i < src.n_cols / 2; i++)
    {
        H(2*i,0)=src(0,2*i);
        H(2*i,1)=src(0,2*i+1);
        H(2*i,2)=1;

        H(2*i+1,0)=src(0,2*i+1);
        H(2*i+1,1)=-src(0,2*i);
        H(2*i+1,3)=1;
    }
    arma::mat x=inv(trans(H)*H+lambda)*(trans(H)*dst);
    arma::mat rt(2,3);
    rt(0,0)=x(0,0);
    rt(0,1)=x(1,0);
    rt(1,0)=-rt(0,1);
    rt(1,1)=rt(0,0);
    rt(0,2)=x(2,0);
    rt(1,2)=x(3,0);
    return rt;
}
///平面内刚性变换
arma::mat ctools::trans2D(arma::mat rt, arma::mat src)
{
    src.reshape(2,src.n_cols/2);
    arma::mat dst;
    src = join_cols(src,arma::ones(1,src.n_cols));
//    cout<<"test: "<<src.n_rows<<","<<src.n_cols<<endl;
    dst = rt*src;
    dst.reshape(1,dst.n_cols*2);
    return dst;
}
///估计二维平面上的仿射变换
arma::mat ctools::calcTransAffine2D(arma::mat src, arma::mat dst)
{
    arma::mat lambda;
    lambda=0.00001*lambda.eye(6,6);//shrinkage matrix
    arma::mat H;
    H=H.zeros(src.n_cols,6);
    dst = dst.t();
	for (uint32_t i = 0; i < src.n_cols / 2; i++)
    {
        H(2*i,0)=src(0,2*i);
        H(2*i,1)=src(0,2*i+1);
        H(2*i,2)=1;

        H(2*i+1,3) = src(0,2*i);
        H(2*i+1,4) = src(0,2*i+1);
        H(2*i+1,5) = 1;
    }
    arma::mat x=inv(trans(H)*H+lambda)*(trans(H)*dst);
    x.reshape(3,2);
    x=x.t();
    return x;
}
///对二维平面上的点进行仿射变换

arma::mat ctools::transAffine2D(arma::mat trmat, arma::mat src)
{
    src.reshape(2,src.n_cols/2);
    arma::mat dst;
    src = join_cols(src,arma::ones(1,src.n_cols));
//    cout<<"test: "<<src.n_rows<<","<<src.n_cols<<endl;
    dst = trmat*src;
    dst.reshape(1,dst.n_cols*2);
    return dst;
}
///随机游走算法，在0-range之间随机选择n个整数
vector<uint32_t> ctools::randperm(const uint32_t range, uint32_t n)
{
    vector<uint32_t> rtRdNum;
	if (range <= 0)
    {
        cout<<"the range must > 0"<<endl;
        return rtRdNum;
    }
	if (n > range)
    {
//        cout<<"the number must < range: "<<range<<endl;
//        return rtRdNum;
        //cout<<"the number is large than range, set it to range"<<endl;
		n = range;
		for (uint32_t k = 0; k < n; k++)
        {
            rtRdNum.push_back(k);
        }
        return rtRdNum;
    }
    srand((unsigned int)(time(NULL)));
    while (rtRdNum.size() < n)
    {
		uint32_t rdn = rand() % range;
        if(uniqueVector(rtRdNum,rdn))
        {
            rtRdNum.push_back(rdn);
        }
    }
    return rtRdNum;
}
/** @brief uniqueVector
  *
  * @todo: document this function
  */
bool ctools::uniqueVector(const vector<uint32_t>& rdidx, const uint32_t selidx)
{
	for (const auto& ele : rdidx)
		if (ele == selidx)
			return false;
	
	return true;
}

///获取给定人脸形状的bounding box
arma::mat ctools::getBoundingBox(arma::mat faceShape)
{
    faceShape.reshape(2,faceShape.n_cols/2);
    arma::mat max_x, max_y,min_x,min_y;
    max_x = arma::max(faceShape.row(0));
    min_x = arma::min(faceShape.row(0));
    max_y = arma::max(faceShape.row(1));
    min_y = arma::min(faceShape.row(1));
    ///calc the bouding box;
    arma::mat bdbox(1,4);
    bdbox(0,0)=min_x(0,0);
    bdbox(0,1)=min_y(0,0);
    bdbox(0,2)=max_x(0,0)-min_x(0,0);
    bdbox(0,3)=max_y(0,0)-min_y(0,0);
    return bdbox;
}
void ctools::starttimer()
{
    start = clock();
}
void ctools::stoptimer()
{
    stop = 1000*(clock()-start)/CLOCKS_PER_SEC;
    cout<<"Elapsed time is: "<<stop<<" ms."<<endl;
}
/** @brief calcHist
  *
  * @todo: 计算直方图
  */
vector<int> ctools::calcHist(arma::mat data, int binNum)
{
    vector<int> hist(0,binNum);
    hist.resize(binNum);
    arma::mat min_v(1,1);
    arma::mat max_v(1,1);
//    min_v = arma::min(arma::min(data,0),1);
//    max_v = arma::max(arma::max(data,0),1);
    max_v(0,0) = -255;
    min_v(0,0) = 255;

	const float step = (max_v(0, 0) - min_v(0, 0)) / binNum - 1;
	for (uint32_t r = 0; r < data.n_rows; r++)
    {
		for (uint32_t c = 0; c < data.n_cols; c++)
        {
            int idx = (data(r,c)-min_v(0,0))/step;
            hist[idx]++;
        }
    }
    return hist;
}
/** @brief calRotate2D
  *
  * @todo: calculat the rotation within the plane
  */

arma::mat ctools::calRotate2D(arma::mat src, arma::mat dst)
{
    src.reshape(2,src.n_cols/2);
    dst.reshape(2,dst.n_cols/2);
    ///centerized
    arma::mat msrc = mean(src,1);
    arma::mat mdst = mean(dst,1);
    src = src-repmat(msrc,1,src.n_cols);
    dst = dst-repmat(mdst,1,dst.n_cols);
    //src and dst are mat of size 1*numpt*2
    arma::mat lambda;
    lambda=0.00001*lambda.eye(2,2);//shrinkage matrix
    arma::mat H;
    H=H.zeros(src.n_cols*2,2);
	for (uint32_t i = 0; i < src.n_cols; i++)
    {
        H(2*i,0)=src(0,i);
        H(2*i,1)=src(1,i);
        H(2*i+1,0)=src(1,i);
        H(2*i+1,1)=-src(0,i);
    }
    dst.reshape(1,dst.n_cols*2);
    dst = dst.t();
    arma::mat x=inv(trans(H)*H+lambda)*(trans(H)*dst);
    arma::mat rt(2,2);
    rt(0,0)=x(0,0);
    rt(0,1)=x(1,0);
    rt(1,0)=-rt(0,1);
    rt(1,1)=rt(0,0);
    return rt;
}

/** @brief rotate2D
  *
  * @todo: rotate the shape src according to rmat: rmat*src
  */
arma::mat ctools::rotate2D(arma::mat rmat, arma::mat src)
{
    arma::mat src_rotated;
    src.reshape(2,src.n_cols/2);
    src_rotated = rmat * src;
    src_rotated.reshape(1,src_rotated.n_cols*2);
    return src_rotated;
}

