cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

project(VocabTree)


INCLUDE_DIRECTORIES(ThirdParty/ann_1.1/include)
INCLUDE_DIRECTORIES(ThirdParty/ann_1.1_char/include)		
link_directories(ThirdParty/ann_1.1/MS_Win32/dll/Release)
link_directories(ThirdParty/ann_1.1_char/MS_Win32/dll/Release)


add_executable(VocabTree main.cpp
keys2.cpp
kmeans.cpp 
kmeans_kd.cpp 
VocabFlatNode.cpp
VocabTree.cpp
VocabTreeBuild.cpp
VocabTreeIO.cpp
VocabTreeUtil.cpp
qsort.c
util.c
)

IF(WIN32) 
target_link_libraries( VocabTree ../ThirdParty/ann_1.1/MS_Win32/dll/Release/ANN ../ThirdParty/ann_1.1_char/x64/Release/ANN_char)
ENDIF()
