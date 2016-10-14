#ifdef _DEBUG

#pragma comment(lib, R"(zlibd.lib)")
#pragma comment(lib, R"(ceresd.lib)")
#pragma comment(lib, R"(armadillod.lib)")
#pragma comment(lib, R"(superlu\cblasd.lib)")
#pragma comment(lib, R"(superlu\superlud.lib)")
#pragma comment(lib, R"(blas_win64_MTd.lib)")
#pragma comment(lib, R"(lapack_win64_MTd.lib)")
#pragma comment(lib, R"(glog\glogd.lib)")
#pragma comment(lib, R"(gflags\gflags_staticd.lib)")
#pragma comment(lib, R"(opencv\opencv_core310d.lib)")
#pragma comment(lib, R"(opencv\opencv_flann310d.lib)")

#else

#pragma comment(lib, R"(zlib.lib)")
#pragma comment(lib, R"(ceres.lib)")
#pragma comment(lib, R"(armadillo.lib)")
#pragma comment(lib, R"(superlu\cblas.lib)")
#pragma comment(lib, R"(superlu\superlu.lib)")
#pragma comment(lib, R"(blas_win64_MT.lib)")
#pragma comment(lib, R"(lapack_win64_MT.lib)")
#pragma comment(lib, R"(glog\glog.lib)")
#pragma comment(lib, R"(gflags\gflags_static.lib)")
#pragma comment(lib, R"(opencv\opencv_core310.lib)")
#pragma comment(lib, R"(opencv\opencv_flann310.lib)")

#endif


#pragma comment(lib, R"(opengl32.lib)")
#pragma comment(lib, R"(ippicvmt.lib)")
#pragma comment(lib, R"(pcl\pcl_common_release.lib)")
#pragma comment(lib, R"(pcl\pcl_io_ply_release.lib)")
#pragma comment(lib, R"(pcl\pcl_io_release.lib)")
#pragma comment(lib, R"(pcl\pcl_visualization_release.lib)")