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
    explicit FrameDetectionWorker(QObject *parent = nullptr);
    Detector* getDetector() { return this->detector; }

public slots:
    void doDetection(QVideoFrame frame, int sequenceNumber);

signals:
    void frameReady(QPixmap frame, int sequenceNumber);

private:
    Detector *detector;

};

#endif // FRAMEDETECTIONWORKER_H
