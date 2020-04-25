#ifndef FRAMEDETECTIONWORKER_H
#define FRAMEDETECTIONWORKER_H

#include "detector.h"

#include <QObject>
#include <QImage>
#include <QPixmap>
#include <QVideoFrame>

class FrameDetectionWorker : public QObject
{
    Q_OBJECT
public:
    explicit FrameDetectionWorker(Detector *detector, int id = 0, int total = 1, QObject *parent = nullptr);

public slots:
    void doDetection(QVideoFrame frame, int sequenceNumber);

signals:
    void frameReady(QImage frame, int sequenceNumber);

private:
    Detector *detector;
    int id, total;

};

#endif // FRAMEDETECTIONWORKER_H
