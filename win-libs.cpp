/*
 * This cpp add dependency of other libraries. 
 * It's only used in VS project.
 * However, boost libraries are controlled by project settings(VC directory)
 * blas and lapack is optional for ceres solver.
 * I choose an unofficial dynamic library just because it does not depend on gfortran and gcc library
 **/

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
//#pragma comment(lib, R"(opencv\opencv_core310d.lib)")
//#pragma comment(lib, R"(opencv\opencv_flann310d.lib)")

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
//#pragma comment(lib, R"(opencv\opencv_core310.lib)")
//#pragma comment(lib, R"(opencv\opencv_flann310.lib)")

#endif


#pragma comment(lib, R"(opengl32.lib)")
//#pragma comment(lib, R"(ippicvmt.lib)")
#pragma comment(lib, R"(vtk\vtkCommonCore-7.0.lib)")
#pragma comment(lib, R"(vtk\vtkCommonDataModel-7.0.lib)")
#pragma comment(lib, R"(vtk\vtkCommonMath-7.0.lib)")
#pragma comment(lib, R"(vtk\vtkRenderingCore-7.0.lib)")
#pragma comment(lib, R"(pcl\pcl_common_release.lib)")
#pragma comment(lib, R"(pcl\pcl_io_ply_release.lib)")
#pragma comment(lib, R"(pcl\pcl_io_release.lib)")
#pragma comment(lib, R"(pcl\pcl_visualization_release.lib)")