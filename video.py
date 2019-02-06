import numpy as np
import cv2

class vid:
    def __init__(self, video):
        vid = cv2.VideoCapture(video)
        ##total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = 39
        vid_lst = []
        for i in range(total_frames):
            frame = vid.read()[1]
            vid_lst.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    
        self.video = np.array(vid_lst)
        self.linear = self.video.reshape(self.video.size)
        
        self.pic3D = vid_into_pic3D(self)
        self.longpic = self.video.reshape(3900, 100)


def vid_into_pic3D(vid):
    pic3D = np.zeros((100, 100, 39), dtype = np.uint8)
    for frame in range(39):
        for row in  range(100):
            for pixel in range(100):
                pic3D[row, pixel, frame] = vid.video[frame, row, pixel]
    return pic3D