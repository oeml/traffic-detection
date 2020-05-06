#include "videosurface.h"
#include "framedetectionworker.h"

#include <opencv2/opencv.hpp>
#include <QDebug>
#include <QVideoSurfaceFormat>
#include <QGraphicsPixmapItem>

bool operator<(const SequencedFrame& lhs, const SequencedFrame& rhs)
{
    return lhs.seqNum > rhs.seqNum;
}

VideoSurface::VideoSurface(QGraphicsView *view, QGraphicsPixmapItem *pixmap, QObject *parent)
    : QAbstractVideoSurface(parent)
    , view(view)
    , pixmap(pixmap)
    , imageFormat(QImage::Format_Invalid)
{
    int nrThreads = QThread::idealThreadCount() / 2;
    for (int i = 0; i < nrThreads; ++i) {
        QThread *thread = new QThread;
        Detector *detector = new Detector("../traced-models/adam-100e-16l.pt",
                                          "../yolo-custom/config/custom.names");
        detector->moveToThread(thread);
        FrameDetectionWorker *worker = new FrameDetectionWorker(detector, i, nrThreads);
        worker->moveToThread(thread);

        connect(thread, &QThread::finished, worker, &QObject::deleteLater);
        connect(this, &VideoSurface::frameAvailable, worker, &FrameDetectionWorker::doDetection);
        connect(worker, &FrameDetectionWorker::frameReady, this, &VideoSurface::receiveFrame);
        threads.append(thread);
        thread->start();
    }

    connect(this, &VideoSurface::frameReady, this, &VideoSurface::displayFrame);
}

VideoSurface::~VideoSurface()
{
    foreach (QThread *thread, threads) {
        thread->quit();
        thread->wait();
    }
}

QList<QVideoFrame::PixelFormat> VideoSurface::supportedPixelFormats(
            QAbstractVideoBuffer::HandleType type) const
{
    Q_UNUSED(type);
    return QList<QVideoFrame::PixelFormat>()
        << QVideoFrame::Format_ARGB32
        << QVideoFrame::Format_ARGB32_Premultiplied;
}

bool VideoSurface::start(const QVideoSurfaceFormat &format)
{
    const QImage::Format imageFormat = QVideoFrame::imageFormatFromPixelFormat(format.pixelFormat());
    const QSize size = format.frameSize();

    if (imageFormat != QImage::Format_Invalid && !size.isEmpty()) {
        view->setMinimumSize(size);
        this->imageFormat = imageFormat;
        QAbstractVideoSurface::start(format);
        return true;
    }
    else{
        return false;
    }
}

bool VideoSurface::present(const QVideoFrame &frame)
{
    if (frame.isValid() && !paused) {
        QVideoFrame clonedFrame(frame);
        emit frameAvailable(clonedFrame, frameCounter);
        ++frameCounter;
        return true;
    }

    return false;
}


void VideoSurface::displayFrame(QPixmap frame)
{
    pixmap->setPixmap(frame);
    view->fitInView(QRectF(0,0,frame.width(),frame.height()),Qt::KeepAspectRatio);
}


void VideoSurface::receiveFrame(QImage frame, int seqNum)
{
    if (seqNum == currSeqNum + 1) {
        emit frameReady(QPixmap::fromImage(frame));
        ++currSeqNum;

        while (!q.empty()) {
            SequencedFrame nextFrame = q.top();
            if (nextFrame.seqNum == currSeqNum + 1) {
                emit frameReady(QPixmap::fromImage(nextFrame.frame));
                ++currSeqNum;
                q.pop();
            } else {
                break;
            }
        }
    } else {
        q.push(SequencedFrame(frame.copy(), seqNum));
    }
}
