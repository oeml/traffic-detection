#ifndef VIDEOPLAYER_H
#define VIDEOPLAYER_H

#include <QWidget>
#include <QMediaPlayer>
#include <QLabel>
#include <QPixmap>

class VideoPlayer : public QWidget
{
    Q_OBJECT

public:
    VideoPlayer(QWidget *parent = nullptr);
    ~VideoPlayer();

public slots:
    void forwardVideo(QPixmap frame);

private:
    QMediaPlayer* m_mediaPlayer;
    QLabel *label;
};
#endif // VIDEOPLAYER_H
