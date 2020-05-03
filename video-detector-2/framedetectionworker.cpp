#include "framedetectionworker.h"

#include <QImage>
#include <QPixmap>
#include <QDebug>
#include <QThread>
#include <QAbstractVideoBuffer>

FrameDetectionWorker::FrameDetectionWorker(Detector *detector, int id, int total, QObject *parent) : QObject(parent)
{
    this->detector = detector;
    this->id = id;
    this->total = total;
}

void FrameDetectionWorker::doDetection(QVideoFrame frame, int sequenceNumber)
{
    if (sequenceNumber % total != id) return;

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

    mat = detector->detect(mat, sequenceNumber, id);

    cv::cvtColor(mat, dst, cv::COLOR_BGR2BGRA);
    const QImage detection(dst.data,
                           dst.cols, dst.rows,
                           static_cast<int>(dst.step),
                           QImage::Format_ARGB32);

    frame.unmap();
    emit frameReady(detection.copy(), sequenceNumber);
}
