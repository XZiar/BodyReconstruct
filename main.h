#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
#include <vector>
#include <memory>
#include <time.h>
#include <chrono>
#include <atomic>
#include <thread>

#include <armadillo>

#include <opencv2/core/core.hpp>
#include <opencv2/flann/flann.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <ceres/ceres.h>

static const uint32_t POSPARAM_NUM = 31;
static const uint32_t SHAPEPARAM_NUM = 20;
static const uint32_t EVALUATE_POINTS_NUM = 6449;


uint64_t getCurTime();
uint64_t getCurTimeNS();
void sleepMS(uint32_t ms);
bool yesORno(const char *str);
extern bool isVtune;