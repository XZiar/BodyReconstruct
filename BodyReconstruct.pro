TEMPLATE = app
CONFIG += console
CONFIG += c++14
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += link_pkgconfig

OBJECTS_DIR = ./obj

SOURCES += main.cpp \
    src/tools.cpp \
    src/fitMesh.cpp \
    shapemodel/cshapepose.cpp \
    shapemodel/paramMap.cpp \
    shapemodel/Show.cpp \
    shapemodel/NRBM.cpp \
    shapemodel/NMath.cpp \
    shapemodel/CTMesh-30DOF.cpp

HEADERS += main.h \
	include/miniBLAS.hpp \
	include/kdNNTree.hpp \
	include/solverModel.hpp \
    include/tools.h \
    include/fitMesh.h \
    shapemodel/cshapepose.h \
	shapemodel/CVector.h \
	shapemodel/CMatrix.h \
	shapemodel/CTMesh.h

INCLUDEPATH += \
    include \
    shapemodel/lib/include \
    shapemodel/lib/nr \
    shapemodel \
    /usr/local/include/vtk-7.0 \
    /usr/local/include/pcl-1.8 \
    /usr/include

LIBS += /usr/local/lib/libboost_system.so \
        /usr/local/lib/libarmadillo.so \
        /usr/local/lib/libglog.so \
        /usr/local/lib/libcblas.a \
        /usr/local/lib/libceres.a \
        /usr/lib/gcc/x86_64-linux-gnu/5/libgfortran.so \
        /usr/local/lib/libvtkCommonCore-7.0.so \
        /usr/local/lib/libvtkCommonDataModel-7.0.so \
        /usr/local/lib/libvtkCommonMath-7.0.so \
        /usr/local/lib/libvtkRenderingCore-7.0.so \
        -lgomp \
        -lpthread

QMAKE_CXXFLAGS += -mavx
QMAKE_CXXFLAGS += -g

PKGCONFIG += pcl_io-1.8
PKGCONFIG += pcl_visualization-1.8
PKGCONFIG += lapack
PKGCONFIG += blas
