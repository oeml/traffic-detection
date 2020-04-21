#ifndef DETECTOR_H
#define DETECTOR_H

#include <QObject>
#undef slots
#include "torch/script.h"
#define slots Q_SLOTS
#include <opencv2/opencv.hpp>
#include <string>

class Detector : public QObject
{
    Q_OBJECT
public:
    Detector(std::string modulePath, std::string classNamesPath);
    ~Detector();
    cv::Mat detect(cv::Mat originalImage);

private:
    torch::jit::script::Module m_module;
    std::vector< std::string > m_classNames;

    torch::Tensor extractResults(torch::Tensor rawPredictions, float confidenceThreshold, float nmsThreshold);
};

#endif // DETECTOR_H
