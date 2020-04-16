#include "videoplayer.h"

#include "videosurface.h"

#include <QVBoxLayout>

VideoPlayer::VideoPlayer(QWidget *parent)
    : QWidget(parent)
{
    // this->resize(600, 450);

    VideoSurface *surface = new VideoSurface;

    m_mediaPlayer = new QMediaPlayer(this);
    label = new QLabel;
    m_mediaPlayer->setMedia(QUrl::fromLocalFile("../sample.mp4"));
    m_mediaPlayer->setVideoOutput(surface);

    connect(surface, SIGNAL(frameReady(QPixmap)), this, SLOT(forwardVideo(QPixmap)));

    QBoxLayout *layout = new QVBoxLayout;
    layout->addWidget(label);
    setLayout(layout);

    m_mediaPlayer->play();
}

VideoPlayer::~VideoPlayer()
{
}

void VideoPlayer::forwardVideo(QPixmap frame)
{
    label->setPixmap(frame);
}

