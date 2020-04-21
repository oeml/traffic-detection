#include "detector.h"

#define IMAGE_SIZE 416
#define NR_CLASSES 80

static cv::Scalar colors[6] = {
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0),
    cv::Scalar(255, 0, 255),
    cv::Scalar(0, 255, 255)
};


static inline torch::Tensor getBoundingBoxIOU(torch::Tensor box1, torch::Tensor box2)
{
    torch::Tensor box1x1, box1y1, box1x2, box1y2;
    box1x1 = box1.select(1, 0);
    box1y1 = box1.select(1, 1);
    box1x2 = box1.select(1, 2);
    box1y2 = box1.select(1, 3);
    torch::Tensor box2x1, box2y1, box2x2, box2y2;
    box2x1 = box2.select(1, 0);
    box2y1 = box2.select(1, 1);
    box2x2 = box2.select(1, 2);
    box2y2 = box2.select(1, 3);

    torch::Tensor intersectionX1 =  torch::max(box1x1, box2x1);
    torch::Tensor intersectionY1 =  torch::max(box1y1, box2y1);
    torch::Tensor intersectionX2 =  torch::min(box1x2, box2x2);
    torch::Tensor intersectionY2 =  torch::min(box1y2, box2y2);
    torch::Tensor intersectionArea =
        torch::max(intersectionX2 - intersectionX1 + 1, torch::zeros(intersectionX2.sizes()))
        * torch::max(intersectionY2 - intersectionY1 + 1, torch::zeros(intersectionX2.sizes()));

    torch::Tensor box1Area = (box1x2 - box1x1 + 1)*(box1y2 - box1y1 + 1);
    torch::Tensor box2Area = (box2x2 - box2x1 + 1)*(box2y2 - box2y1 + 1);
    torch::Tensor unionArea = box1Area + box2Area - intersectionArea;

    return intersectionArea / unionArea;
}


Detector::Detector(std::string modulePath, std::string classNamesPath)
{
    m_module = torch::jit::load(modulePath);

    m_classNames.resize(NR_CLASSES);
    std::ifstream classFile(classNamesPath);
    if (classFile.is_open()) {
        int i = 0;
        while (std::getline(classFile, m_classNames[i])) {
            ++i;
        }
        classFile.close();
    }
}


Detector::~Detector()
{

}


torch::Tensor Detector::extractResults(torch::Tensor rawPredictions, float confidenceThreshold, float nmsThreshold)
{
    auto confidenceMask = (rawPredictions.select(2, 4) > confidenceThreshold).to(torch::kF32).unsqueeze(2);
    rawPredictions.mul_(confidenceMask);
    auto nonzeroIndices = torch::nonzero(rawPredictions.select(2,4)).transpose(0,1).contiguous();

    if (nonzeroIndices.size(0) == 0) {
        return torch::zeros({0});
    }

    // convert from center notation to vertex notation
    torch::Tensor box = torch::ones(rawPredictions.sizes(), rawPredictions.options());
    box.select(2,0) = rawPredictions.select(2,0) - rawPredictions.select(2,2).div(2);
    box.select(2,1) = rawPredictions.select(2,1) - rawPredictions.select(2,3).div(2);
    box.select(2,2) = rawPredictions.select(2,0) + rawPredictions.select(2,2).div(2);
    box.select(2,3) = rawPredictions.select(2,1) + rawPredictions.select(2,3).div(2);
    rawPredictions.slice(2,0,4) = box.slice(2,0,4);

    int batchSize = rawPredictions.size(0);
    int attrLength = 5;

    torch::Tensor output = torch::ones({1, rawPredictions.size(2) + 1});
    bool doConcat = false;
    int num = 0, i;
    for (i = 0; i < batchSize; ++i) {
        auto currImagePrediction = rawPredictions[i];
        std::tuple< torch::Tensor, torch::Tensor > maxClasses = torch::max(
            currImagePrediction.slice(1, attrLength, attrLength + NR_CLASSES), 1);

        auto maxObjectness = std::get< 0 >(maxClasses).to(torch::kF32).unsqueeze(1);
        auto maxClassScore = std::get< 1 >(maxClasses).to(torch::kF32).unsqueeze(1);

        // result is n x 7: left x, left y, right x, right y, objectness, class score, class id
        currImagePrediction = torch::cat({currImagePrediction.slice(1,0,attrLength), maxObjectness, maxClassScore}, 1 );

        auto nonzeroes = torch::nonzero(currImagePrediction.select(1,4));
        auto imagePredictionView = currImagePrediction.index_select(0, nonzeroes.squeeze()).view({-1, 7});

        std::vector< torch::Tensor > imageClasses;
        size_t len = imagePredictionView.size(0);
        for (size_t j = 0; j < len; ++j) {
            bool found = false;
            for (size_t k = 0; k < imageClasses.size(); ++k) {
                auto matchingClasses = (imagePredictionView[j][6] == imageClasses[k]);
                if (torch::nonzero(matchingClasses).size(0) > 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                imageClasses.push_back(imagePredictionView[j][6]);
            }
        }

        for (size_t k = 0; k < imageClasses.size(); ++k) {
            auto cls = imageClasses[k];
            auto clsMask = imagePredictionView * (imagePredictionView.select(1,6) == cls).to(torch::kF32).unsqueeze(1);
            auto clsMaskNonzeroIndex = torch::nonzero(clsMask.select(1,5)).squeeze();
            auto imageClassPredictions = imagePredictionView.index_select(0, clsMaskNonzeroIndex).view({-1,7});

            std::tuple< torch::Tensor, torch::Tensor > sortedByConfidence = torch::sort(imageClassPredictions.select(1,4));
            auto sortedByConfidenceIndex = std::get< 1 >(sortedByConfidence);
            imageClassPredictions = imageClassPredictions.index_select(0, sortedByConfidenceIndex.squeeze()).cpu();

            for (long long step = 0; step < imageClassPredictions.size(0) - 1; ++step) {
                int curr = imageClassPredictions.size(0) - 1 - step;
                if (curr <= 0) break;  // this is a workaround bc I modify inside the loop

                auto ious = getBoundingBoxIOU(imageClassPredictions[curr].unsqueeze(0),
                                              imageClassPredictions.slice(0, 0, curr));
                auto iouMask = (ious < nmsThreshold).to(torch::kF32).unsqueeze(1);
                imageClassPredictions.slice(0, 0, curr) = imageClassPredictions.slice(0, 0, curr) * iouMask;

                auto nonzeroes = torch::nonzero(imageClassPredictions.select(1, 4)).squeeze();
                imageClassPredictions = imageClassPredictions.index_select(0, nonzeroes).view({-1, 7});
            }

            torch::Tensor batchIndex = torch::ones({imageClassPredictions.size(0), 1}).fill_(i);
            if (!doConcat) {
                output = torch::cat({batchIndex, imageClassPredictions}, 1);
                doConcat = true;
            } else {
                auto toConcat = torch::cat({batchIndex, imageClassPredictions}, 1);
                output = torch::cat({output, toConcat}, 0);
            }

            ++num;
        }
    }

    if (num == 0) {
        return torch::zeros({0});
    }
    return output;
}


cv::Mat Detector::detect(cv::Mat originalImage)
{
    cv::Mat resizedImage, floatImage;

    cv::cvtColor(originalImage, resizedImage,  cv::COLOR_BGR2RGB);
    cv::resize(resizedImage, resizedImage, cv::Size(IMAGE_SIZE, IMAGE_SIZE));

    resizedImage.convertTo(floatImage, CV_32F, 1.0/255);

    torch::Device device(torch::kCPU);
    auto imageTensor = torch::from_blob(floatImage.data, {1, IMAGE_SIZE, IMAGE_SIZE, 3}).to(device);
    imageTensor = imageTensor.permute({0,3,1,2});

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(imageTensor);

    auto start = std::chrono::high_resolution_clock::now();
    at::Tensor output = m_module.forward(inputs).toTensor();
    auto result = extractResults(output, 0.6, 0.4);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "inference taken : " << duration.count() << " ms" << std::endl;

    if (result.dim() == 1) {
        std::cout << "no object found" << std::endl;
    } else {
        int obj_num = result.size(0);

        std::cout << obj_num << " objects found" << std::endl;

        float wScale = float(originalImage.cols) / IMAGE_SIZE;
        float hScale = float(originalImage.rows) / IMAGE_SIZE;

        result.select(1,1).mul_(wScale);
        result.select(1,2).mul_(hScale);
        result.select(1,3).mul_(wScale);
        result.select(1,4).mul_(hScale);

        auto resultAccessor = result.accessor<float, 2>();

        for (int i = 0; i < result.size(0) ; i++)
        {
            int cls = resultAccessor[i][7];
            cv::rectangle(originalImage,
                          cv::Point(resultAccessor[i][1], resultAccessor[i][2]),
                          cv::Point(resultAccessor[i][3], resultAccessor[i][4]),
                          colors[cls % 6],
                          2);

            std::string className = m_classNames[cls];
            cv::rectangle(originalImage,
                          cv::Point(resultAccessor[i][1], resultAccessor[i][2] - 35),
                          cv::Point(resultAccessor[i][1] + className.length() * 19,
                                    resultAccessor[i][2]),
                          colors[cls % 6],
                          cv::FILLED);
            cv::putText(originalImage,
                        className,
                        cv::Point(resultAccessor[i][1], resultAccessor[i][2] - 10),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1,
                        cv::Scalar(255, 255, 255));
        }
    }

    std::cout << "Done" << std::endl;
    return originalImage;
}
