#ifndef VIDEOSURFACE_H
#define VIDEOSURFACE_H

#include "detector.h"

#include <QAbstractVideoSurface>
#include <QList>
#include <QPixmap>

class VideoSurface : public QAbstractVideoSurface
{
    Q_OBJECT
public:
    VideoSurface(QObject *parent = 0);

    QList<QVideoFrame::PixelFormat> supportedPixelFormats(QAbstractVideoBuffer::HandleType type = QAbstractVideoBuffer::NoHandle) const override;

    bool present(const QVideoFrame &frame) override;

signals:
    // void frameAvailable(QImage frame);
    void frameReady(QPixmap frame);

private:
    Detector *detector;
};

#endif // VIDEOSURFACE_H
