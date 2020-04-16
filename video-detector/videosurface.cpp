#include "videosurface.h"
#include <opencv2/opencv.hpp>
#include <QDebug>

VideoSurface::VideoSurface(QObject *parent) :
    QAbstractVideoSurface(parent)
{
    detector = new Detector("../traced_model.pt",
                            "../coco.names");
}

QList<QVideoFrame::PixelFormat> VideoSurface::supportedPixelFormats(
            QAbstractVideoBuffer::HandleType type) const
{
    Q_UNUSED(type);
    return QList<QVideoFrame::PixelFormat>()
        << QVideoFrame::Format_RGB32
        << QVideoFrame::Format_ARGB32
        << QVideoFrame::Format_ARGB32_Premultiplied
        << QVideoFrame::Format_RGB565
        << QVideoFrame::Format_RGB555;
}

bool VideoSurface::present(const QVideoFrame &frame)
{
    if (frame.isValid()) {
        QVideoFrame clonedFrame(frame);
        clonedFrame.map(QAbstractVideoBuffer::ReadOnly);

        QImage::Format imageFormat = QVideoFrame::imageFormatFromPixelFormat(clonedFrame.pixelFormat());
        const QImage image(clonedFrame.bits(),
                           clonedFrame.width(),
                           clonedFrame.height(),
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

        emit frameReady(QPixmap::fromImage(detection));

        clonedFrame.unmap();
        return true;
    }

    return false;
}
