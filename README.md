# BodyReconstruct
3D Body Reconstruct from scan data of RGBD camara

##Requirement
eigen  
ceres solver  
gtk  
pcl  
armadillo  
opencv(currently removed)  

cpu : support AVX  
compiler : support C++14  
x64 only  

##Structure
###BodyReconstruct.pro
qt5.7 project
###BodyReconstruct.sln
vs2015 solution
###BodyReconstruct.vcxproj
vs2015 peoject for linux
###BodyReconstructWin.vcxproj
vs2015 project for windows
###win-libs.cpp
requierd by windows msvc, coded library dependency
