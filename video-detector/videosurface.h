#ifndef VIDEOSURFACE_H
#define VIDEOSURFACE_H

#include "detector.h"

#include <QAbstractVideoSurface>
#include <QList>
#include <QPixmap>
#include <QGraphicsView>
#include <QThread>

class VideoSurface : public QAbstractVideoSurface
{
    Q_OBJECT
public:
    VideoSurface(QGraphicsView *view, QGraphicsPixmapItem *item, QObject *parent = 0);
    ~VideoSurface();

    QList<QVideoFrame::PixelFormat> supportedPixelFormats(QAbstractVideoBuffer::HandleType type = QAbstractVideoBuffer::NoHandle) const override;

    bool start(const QVideoSurfaceFormat &format) override;
    bool present(const QVideoFrame &frame) override;

public slots:
    void displayFrame(QPixmap frame, int seqNum);

signals:
    void frameAvailable(QVideoFrame frame, int sequenceNumber);
    // void frameReady(QPixmap frame);

private:
    // Detector *detector;
    QGraphicsView *view;
    QGraphicsPixmapItem *pixmap;
    QImage::Format imageFormat;

    int frameCounter = 0;
    QThread thread;

};

#endif // VIDEOSURFACE_H
