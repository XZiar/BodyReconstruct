///class of tools ,like geometry transformations tools etc.
#pragma once
#ifndef TOOLS_H
#define TOOLS_H

#include "main.h"

class ctools
{
    public:
        ctools();
        virtual ~ctools();
        arma::mat calcTrans2D(arma::mat src, arma::mat dst);    ///2D 刚体变换的
        arma::mat trans2D(arma::mat rt, arma::mat src);
        arma::mat calcTransAffine2D(arma::mat src, arma::mat dst); ///2D 仿射变换
        arma::mat transAffine2D(arma::mat trmat, arma::mat src);
        std::vector<uint32_t> randperm(const uint32_t range, uint32_t n);
        arma::mat getBoundingBox(arma::mat faceShape);
        void starttimer();
        void stoptimer();
		std::vector<int> calcHist(arma::mat data, int binNum);
        bool uniqueVector(const std::vector<uint32_t> &rdidx, const uint32_t selidx);
        arma::mat calRotate2D(arma::mat src, arma::mat dst);
        arma::mat rotate2D(arma::mat rmat, arma::mat src);
    private:
        clock_t start,stop;
};

#endif // TOOLS_H
