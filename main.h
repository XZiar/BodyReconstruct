#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
#include <array>
#include <vector>
#include <memory>
#include <time.h>
#include <chrono>
#include <atomic>
#include <thread>
#include <functional>

#include <armadillo>
//#include <opencv2/core/core.hpp>
//#include <opencv2/flann/flann.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <ceres/ceres.h>

using atomic_uint32_t = std::atomic_uint_least32_t;

constexpr uint32_t POSPARAM_NUM = 31;
constexpr uint32_t SHAPEPARAM_NUM = 20;
constexpr uint32_t EVALUATE_POINTS_NUM = 6449;


uint64_t getCurTime();
uint64_t getCurTimeNS();
void sleepMS(uint32_t ms);
bool yesORno(const char *str, const bool expect = true);
int inputNumber(const char *str, const int32_t expect);
extern bool isVtune;

struct SimpleTimer
{
private:
	uint64_t t1, t2;
public:
	SimpleTimer() { t2 = Start(); };
	uint64_t Start() { return t1 = getCurTimeNS(); };
	uint64_t Stop() { return t2 = getCurTimeNS(); };
	uint64_t ElapseNs() { return t2 - t1; };
	uint64_t ElapseUs() { return ElapseNs() / 1000; };
	uint64_t ElapseMs() { return ElapseNs() / 1000000; };
};