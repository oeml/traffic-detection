#ifndef VIDEOPLAYER_H
#define VIDEOPLAYER_H

#include <QWidget>
#include <QMediaPlayer>
#include <QGraphicsScene>
#include <QLabel>
#include <QPixmap>

class QAbstractButton;
class VideoSurface;

class VideoPlayer : public QWidget
{
    Q_OBJECT

public:
    VideoPlayer(QWidget *parent = nullptr);
    ~VideoPlayer();

public slots:
    void forwardVideo(QPixmap frame);
    void openFile();
    void play();

private slots:
    void mediaPlayerStateChanged(QMediaPlayer::State state);

private:
    QMediaPlayer *m_mediaPlayer;
    QGraphicsScene *m_graphicsScene;
    VideoSurface *surface;
    QAbstractButton *m_playButton;
    // QLabel *label;
};
#endif // VIDEOPLAYER_H
