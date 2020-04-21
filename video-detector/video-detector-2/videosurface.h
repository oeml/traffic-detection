#ifndef VIDEOSURFACE_H
#define VIDEOSURFACE_H

#include "detector.h"

#include <QAbstractVideoSurface>
#include <QList>
#include <QPixmap>
#include <QGraphicsView>

class VideoSurface : public QAbstractVideoSurface
{
    Q_OBJECT
public:
    VideoSurface(QGraphicsView *view, QGraphicsPixmapItem *item, QObject *parent = 0);

    QList<QVideoFrame::PixelFormat> supportedPixelFormats(QAbstractVideoBuffer::HandleType type = QAbstractVideoBuffer::NoHandle) const override;

    bool start(const QVideoSurfaceFormat &format) override;
    bool present(const QVideoFrame &frame) override;

signals:
    // void frameAvailable(QImage frame);
    void frameReady(QPixmap frame);

private:
    Detector *detector;
    QGraphicsView *view;
    QGraphicsPixmapItem *pixmap;
    QImage::Format imageFormat;

    void displayFrame(QPixmap frame);
};

#endif // VIDEOSURFACE_H
