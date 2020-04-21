cmake_minimum_required(VERSION 3.5)

project(video-detector LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check http://doc.qt.io/qt-5/deployment-android.html for more information.
# They need to be set before the find_package(Qt5 ...) call.

#if(ANDROID)
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
#    if (ANDROID_ABI STREQUAL "armeabi-v7a")
#        set(ANDROID_EXTRA_LIBS
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libcrypto.so
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libssl.so)
#    endif()
#endif()

set(CMAKE_PREFIX_PATH /Users/home/Desktop/libtorch)
set(OpenCV_DIR /Users/home/build-opencv)

find_package(Qt5 COMPONENTS Widgets Multimedia MultimediaWidgets REQUIRED)
find_package(Torch REQUIRED NO_CMAKE_FIND_ROOT_PATH)
find_package(OpenCV REQUIRED NO_CMAKE_FIND_ROOT_PATH)

include_directories(  ${OpenCV_INCLUDE_DIRS}  ${TORCH_INCLUDE_DIRS} include)

if(ANDROID)
  add_library(video-detector SHARED
    main.cpp
    videoplayer.cpp
    videoplayer.h
    videosurface.cpp
    videosurface.h
    detector.cpp
    detector.h
  )
else()
  add_executable(video-detector
    main.cpp
    videoplayer.cpp
    videoplayer.h
    videosurface.cpp
    videosurface.h
    detector.cpp
    detector.h
  )
endif()

target_link_libraries(video-detector PRIVATE Qt5::Widgets Qt5::Multimedia Qt5::MultimediaWidgets ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET video-detector PROPERTY CXX_STANDARD 14)
