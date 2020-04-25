#ifndef VIDEOSURFACE_H
#define VIDEOSURFACE_H

#include "detector.h"

#include <queue>

#include <QAbstractVideoSurface>
#include <QList>
#include <QPixmap>
#include <QGraphicsView>
#include <QThread>

class SequencedFrame {
public:
    QImage frame;
    int seqNum;

    SequencedFrame(QImage frame, int seqNum) : frame(frame), seqNum(seqNum) {};
};

class VideoSurface : public QAbstractVideoSurface
{
    Q_OBJECT
public:
    VideoSurface(QGraphicsView *view, QGraphicsPixmapItem *item, QObject *parent = 0);
    ~VideoSurface();

    QList<QVideoFrame::PixelFormat> supportedPixelFormats(QAbstractVideoBuffer::HandleType type = QAbstractVideoBuffer::NoHandle) const override;

    bool start(const QVideoSurfaceFormat &format) override;
    bool present(const QVideoFrame &frame) override;

    void setPaused(bool newVal) { this->paused = newVal; }

public slots:
    void receiveFrame(QImage frame, int seqNum);
    void displayFrame(QPixmap frame);

signals:
    void frameAvailable(QVideoFrame frame, int sequenceNumber);
    void frameReady(QPixmap frame);

private:
    // Detector *detector;
    QGraphicsView *view;
    QGraphicsPixmapItem *pixmap;
    QImage::Format imageFormat;

    int frameCounter = 0;
    QList<QThread*> threads;

    bool paused = false;

    int currSeqNum = -1;
    std::priority_queue<SequencedFrame> q;

};

#endif // VIDEOSURFACE_H
