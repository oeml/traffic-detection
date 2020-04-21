#include "videoplayer.h"

#include "videosurface.h"

#include <QVBoxLayout>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QAbstractButton>
#include <QPushButton>
#include <QFileDialog>
#include <QStyle>

VideoPlayer::VideoPlayer(QWidget *parent)
    : QWidget(parent)
{
    this->resize(600, 450);

    m_mediaPlayer = new QMediaPlayer(this);
    m_graphicsScene = new QGraphicsScene();

    QGraphicsPixmapItem *pixmapItem = new QGraphicsPixmapItem();
    m_graphicsScene->addItem(pixmapItem);
    QGraphicsView *view = new QGraphicsView();
    view->setScene(m_graphicsScene);

    surface = new VideoSurface(view, pixmapItem);

    m_mediaPlayer->setVideoOutput(surface);
    connect(m_mediaPlayer, &QMediaPlayer::stateChanged, this, &VideoPlayer::mediaPlayerStateChanged);

    connect(surface, SIGNAL(frameReady(QPixmap)), this, SLOT(forwardVideo(QPixmap)));

    QAbstractButton *openButton = new QPushButton(tr("Open..."));
    connect(openButton, &QPushButton::clicked, this, &VideoPlayer::openFile);

    m_playButton = new QPushButton;
    m_playButton->setEnabled(false);
    m_playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    connect(m_playButton, &QPushButton::clicked, this, &VideoPlayer::play);

    QBoxLayout *layout = new QVBoxLayout;
    layout->addWidget(openButton);
    layout->addWidget(view);
    layout->addWidget(m_playButton);
    setLayout(layout);
}

VideoPlayer::~VideoPlayer()
{
}

void VideoPlayer::openFile()
{
    QUrl fileURL = QFileDialog::getOpenFileUrl(this, tr("Open Video"), QDir::homePath(), tr("Video Files (*.mp3 *mp4)"));
    if (!fileURL.isEmpty()) {
        surface->stop();
        m_mediaPlayer->setMedia(fileURL);
        m_playButton->setEnabled(true);
    }
}

void VideoPlayer::play()
{
    switch (m_mediaPlayer->state()) {
    case QMediaPlayer::StoppedState:
    case QMediaPlayer::PausedState:
        m_mediaPlayer->play();
        break;
    case QMediaPlayer::PlayingState:
        m_mediaPlayer->pause();
        break;
    }
}

void VideoPlayer::mediaPlayerStateChanged(QMediaPlayer::State state)
{
    switch (state) {
    case QMediaPlayer::StoppedState:
    case QMediaPlayer::PausedState:
        m_playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
        break;
    case QMediaPlayer::PlayingState:
        m_playButton->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
        break;
    }
}

