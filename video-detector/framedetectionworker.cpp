#include "framedetectionworker.h"

#include <QImage>
#include <QPixmap>
#include <QDebug>
#include <QThread>
#include <QAbstractVideoBuffer>

FrameDetectionWorker::FrameDetectionWorker(QObject *parent) : QObject(parent)
{
    detector = new Detector("/Users/home/Desktop/traffic-detection/traced_model.pt",
                            "/Users/home/Desktop/traffic-detection/model-python/config/coco.names");
}

void FrameDetectionWorker::doDetection(QVideoFrame frame, int sequenceNumber)
{
    // if (sequenceNumber % total != id) return;

    qDebug() << sequenceNumber;

    frame.map(QAbstractVideoBuffer::ReadOnly);

    QImage::Format imageFormat = QVideoFrame::imageFormatFromPixelFormat(frame.pixelFormat());
    const QImage image(frame.bits(),
                       frame.width(),
                       frame.height(),
                       imageFormat);

    cv::Mat src(image.height(), image.width(),
                CV_8UC4,
                const_cast<uchar*>(image.bits()),
                static_cast<size_t>(image.bytesPerLine()));
    cv::Mat mat, dst;
    cv::cvtColor(src, mat, cv::COLOR_BGRA2BGR);
    mat = detector->detect(mat);
    cv::cvtColor(mat, dst, cv::COLOR_BGR2BGRA);
    const QImage detection(dst.data,
                           dst.cols, dst.rows,
                           static_cast<int>(dst.step),
                           QImage::Format_ARGB32);

    // QThread::msleep(1000);
    qDebug() << sequenceNumber << "done";
    frame.unmap();
    emit frameReady(QPixmap::fromImage(detection), sequenceNumber);
}
