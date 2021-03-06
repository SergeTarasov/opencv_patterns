import cv2 as cv
import numpy as np

from utils.figures import draw_dashed_line, draw_rect


class Frame:
    def __init__(self, x, y, width, height, blue=False):
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.frame = [x, y, width + x, height + y]
        self.blue = blue

    def draw(self, image):
        if self.blue:
            draw_rect(image, (self.x, self.y), (self.width+self.x, self.height+self.y), (192, 0, 0), thickness=2)
            draw_rect(image, (self.x, self.y), (self.width+self.x, self.height+self.y), (0, 128, 0), thickness=1)

        else:
            draw_rect(image, (self.x, self.y), (self.width+self.x, self.height+self.y), (0, 128, 0), thickness=2)
            draw_rect(image, (self.x, self.y), (self.width+self.x, self.height+self.y), (0, 192, 0), thickness=1)

    def move_to(self, x, y):
        self.x, self.y = x, y
        self.frame = [self.x, self.y, self.width + self.x, self.height + self.y]


class ProcessImage:

    threshold_distance = 3
    window_capture_name = 'Video Capture'

    MEDIA_ROOT = '../media/'
    # video_name = MEDIA_ROOT + "transient_instability_in_dusty_plasma_crystal.mp4"
    video_name = MEDIA_ROOT + "10JUN34.AVI"
    # video_name = MEDIA_ROOT + "Airbus_A320_75_105km.avi"

    ref_pt = []
    cropping = False
    temp_updated = False
    frames = []
    frame = np.array((10, 10), dtype=np.uint8)

    def click_and_crop(self, event, x, y, flags, param):

        if self.cropping:
            draw_rect(self.frame, self.ref_pt[0], (x, y), (128, 0, 0))
            self.frames[-1] = Frame(*self.ref_pt[0], x - self.ref_pt[0][0], y - self.ref_pt[0][1], blue=True)

        if event == cv.EVENT_LBUTTONDOWN:
            self.ref_pt = [(x, y)]
            self.cropping = True
            self.frames.append(Frame(*self.ref_pt[0], 0, 0))

        elif event == cv.EVENT_LBUTTONUP:
            width, height = x - self.ref_pt[0][0], y - self.ref_pt[0][1]
            if width < 20 or height < 20:
                del self.frames[-1]
                self.cropping = False
                return

            self.ref_pt.append((x, y))
            self.cropping = False
            self.temp_updated = True
            self.frames[-1] = Frame(*self.ref_pt[0], width, height)

    def loop(self, filename=video_name):

        vid = cv.VideoCapture(filename, 0)
        vid.set(cv.CAP_PROP_POS_FRAMES, 0)
        _, frame = vid.read()

        prev_frame = frame

        cv.namedWindow('frame')
        cv.setMouseCallback('frame', self.click_and_crop)

        paused = False
        while True:

            key = cv.waitKey(10)
            if key == ord('q'):
                break
            elif key == ord('c'):
                paused = True
            elif key == ord('x'):
                paused = False

            if not self.cropping and not paused:
                _, self.frame = vid.read()
                prev_frame = self.frame
            else:
                self.frame = np.copy(prev_frame)

            for frame_object in self.frames:
                frame_object.draw(self.frame)

            # repeat
            if not _:
                vid.release()
                vid = cv.VideoCapture(filename, 0)
                _, self.frame = vid.read()

            if self.temp_updated:
                self.temp_updated = False
                x, y = self.ref_pt[0]
                width, height = self.ref_pt[1][0] - x, self.ref_pt[1][1] - y

            self.process(self.frame)

            cv.imshow('frame', self.frame)

        vid.release()
        cv.destroyAllWindows()

    def process(self, frame):
        # processing
        pass


def start(filename):
    process = ProcessImage()
    process.loop(filename)


if __name__ == '__main__':
    # start("../media/seagull.mp4")
    start(0)
