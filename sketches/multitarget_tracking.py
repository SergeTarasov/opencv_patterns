import cv2 as cv
import numpy as np

from utils.kalman import Kalman


class Filter(Kalman):
    def __init__(self, dynamic_parameters, measure_parameters, control_parameters):
        super().__init__(dynamic_parameters, measure_parameters, control_parameters)

        self.measurement_matrix = np.eye(measure_parameters, dynamic_parameters, dtype=np.float32)
        self.transition_matrix = np.eye(dynamic_parameters, dynamic_parameters, dtype=np.float32)
        self.transition_matrix[0, dynamic_parameters-2] = 5
        self.transition_matrix[1, dynamic_parameters-1] = 5
        self.process_noise_cov = np.eye(dynamic_parameters, dynamic_parameters, dtype=np.float32) * 1e-3
        self.measurement_noise_cov = np.eye(measure_parameters, measure_parameters, dtype=np.float32) * 10
        self.error_cov_post = np.eye(dynamic_parameters, dynamic_parameters, dtype=np.float32)


class Target:
    def __init__(self, position, ID):
        self.tracker = Filter(
            dynamic_parameters=position.shape[0] * 2,
            measure_parameters=position.shape[0],
            control_parameters=position.shape[0]
        )
        self.ID = ID
        self.position = np.array(position)
        self.estimation = np.array([])
        self.track = np.array([[0, 0]])

        self.tracker.state_post[:2] = self.position[:, np.newaxis]

    def update(self):
        self.estimation = self.tracker.predict()

    def upgrade(self, measurement):
        self.track = np.append(self.track, self.estimation.T, axis=0)
        self.tracker.correct(measurement[:, np.newaxis])


def draw_dashed_line(image, pt1, pt2, color, thickness):
    length = np.linalg.norm(np.array(pt1) - np.array(pt2))
    dash_len = 10
    number_of_dashes = int(length / dash_len)
    if number_of_dashes % 2 != 0:
        number_of_dashes += 1

    x_dots = np.linspace(pt1[0], pt2[0], number_of_dashes, dtype=int)
    y_dots = np.linspace(pt1[1], pt2[1], number_of_dashes, dtype=int)

    var = 0
    for i, (y, x) in enumerate(zip(y_dots, x_dots)):
        if i == 0:
            continue

        if var == 0:
            cv.line(image, (y_dots[i-1], x_dots[i-1]), (y, x), color, thickness)
            var = 1
        elif var == 1:
            var = 0


def draw_rect(image, pt1, pt2, color, thickness=1):
    top, bottom = pt1[1], pt2[1]
    left, right = pt1[0], pt2[0]

    draw_dashed_line(image, (top, left),    (top, right),    color, thickness)
    draw_dashed_line(image, (bottom, left), (bottom, right), color, thickness)
    draw_dashed_line(image, (top, left),    (bottom, left),  color, thickness)
    draw_dashed_line(image, (top, right),   (bottom, right), color, thickness)


class Tracker:
    def __init__(self, position, width, height, image):
        self.y, self.x = position

        self.width, self.height = width, height
        self.frame = Frame(*position, width, height)
        self.margin = 40

        self.y_slice = slice(self.y, self.y + self.height)
        self.x_slice = slice(self.x, self.x + self.width)

        self.template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)[self.y_slice, self.x_slice]
        self.buffer = np.zeros(self.template.shape, dtype=np.uint16)

        self.target = Target(np.array([self.y, self.x]), 1)

        self.smooth = 15
        self.counter = self.smooth
        self.kalman_filter_timeout = 100

    def reinit(self, position, width, height, image):
        self.frame.describe = False
        self.__init__(position, width, height, image)

    def update(self, image):

        top = max(self.y - self.margin, 0)
        left = max(self.x - self.margin, 0)
        cropped = image[top:self.y + self.height + self.margin,
                        left:self.x + self.width + self.margin]

        try:
            template_match = cv.matchTemplate(
                cv.cvtColor(cropped, cv.COLOR_BGR2GRAY),
                self.template,
                cv.TM_CCOEFF_NORMED
            )
        except cv.error:
            return

        min_val, max_val, min_loc, [x, y] = cv.minMaxLoc(template_match)

        max_x = left + x
        max_y = top + y

        # tracker_y, tracker_x = self.target.tracker.state_post[:2].T[0]
        tracker_y, tracker_x = self.target.tracker.state_pre[:2].T[0]

        dist_loc = np.linalg.norm(np.array([max_x, max_y]) - np.array([self.x, self.y]))
        dist_tracker = np.linalg.norm(np.array([tracker_x, tracker_y]) - np.array([self.x, self.y]))

        self.kalman_filter_timeout -= 1

        self.target.update()
        self.target.upgrade(np.array([max_y, max_x]))

        if self.kalman_filter_timeout < 5 and dist_loc - dist_tracker > 20:
            self.x = int(tracker_x)
            self.y = int(tracker_y)
        else:
            self.x = max_x
            self.y = max_y

        self.frame.move_to(self.x, self.y)

        self.buffer += cv.cvtColor(
                cropped[y:y+self.height, x:x+self.width].astype(dtype=np.uint8),
                cv.COLOR_BGR2GRAY
            )

        if self.counter == 0:
            self.counter = self.smooth
            self.buffer = self.buffer / (self.smooth + 1)

            self.template = self.buffer.astype(dtype=np.uint8)

        self.counter -= 1

        if self.kalman_filter_timeout > 0:
            self.kalman_filter_timeout -= 1
            cv.drawMarker(image, (int(self.x + 10), int(self.y + 10)), (0, 0, 128),
                          cv.MARKER_TRIANGLE_DOWN, markerSize=10, thickness=1)
        else:
            cv.drawMarker(image, (int(self.x + 10), int(self.y + 10)), (0, 128, 0),
                          cv.MARKER_TRIANGLE_UP, markerSize=10)

        cv.drawMarker(image, (int(self.x + self.width / 2), int(self.y + self.height / 2)), (128, 0, 0))
        cv.drawMarker(image, (int(max_x + self.width / 2), int(max_y + self.height / 2)), (0, 0, 128))

    def set_frame(self, frame_object):
        self.frame.describe = False
        self.frame = frame_object
        self.frame.describe = True


class Frame:
    def __init__(self, x, y, width, height, blue=False, describe=False):
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.frame = [x, y, width + x, height + y]
        self.blue = blue
        self.describe = describe

    def draw(self, image):

        if self.describe or self.blue:
            cv.putText(image, str((self.x, self.y, self.width, self.height)),
                       (self.x, self.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 0), 2)
            cv.putText(image, str((self.x, self.y, self.width, self.height)),
                       (self.x, self.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

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

    ref_pt = []
    cropping = False
    temp_updated = False
    frames = []
    image = np.array((10, 10), dtype=np.uint8)

    def __init__(self, filename=0):
        self.filename = filename
        self.vid = cv.VideoCapture(filename, 0)
        self.vid.set(cv.CAP_PROP_POS_FRAMES, 0)

        _, self.image = self.vid.read()
        self.prev_frame = self.image

        fourcc = cv.VideoWriter_fourcc(*'XVID')
        height, width = self.image.shape[:2]
        self.vid_out = cv.VideoWriter(
            '../media/output/test_multiple_targets_from_frame/' +
            filename.split('/')[-1].split('.')[0] if type(filename) != int else 'out' + '_tracking.avi',
            0,
            fourcc,
            30,
            (width, height)
        )

        self.left, self.top, self.width, self.height = [50, 50, 50, 50]
        self.right, self.bottom = self.left + self.width, self.top + self.height

        self.new_trackers = []

        cv.namedWindow('image')
        cv.setMouseCallback('image', self.click_and_crop)

    def click_and_crop(self, event, x, y, flags, param):

        if self.cropping:
            draw_rect(self.image, self.ref_pt[0], (x, y), (128, 0, 0))
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

    def loop(self, write=False):
        paused = True
        frame_counter = 0
        while True:
            timer = cv.getTickCount()
            key = cv.waitKey(1)

            if key == ord('q'):
                break

            elif key == ord('z'):
                paused = True

            elif key == ord('x'):
                paused = False

            elif key == ord('d'):
                del self.new_trackers[-1]
                del self.frames[-1]

            elif key == ord('c'):
                self.frames.clear()

                del self.new_trackers
                self.new_trackers = []

            if not self.cropping and not paused:
                frame_counter += 1
                _, self.image = self.vid.read()

                # repeat
                if not _:
                    self.vid.release()
                    frame_counter = 0
                    self.vid = cv.VideoCapture(self.filename, 0)
                    _, self.image = self.vid.read()

                self.prev_frame = np.copy(self.image)
                # slightly faster
                # self.prev_frame = self.image

            else:
                self.image = np.copy(self.prev_frame)

            if self.temp_updated:
                self.temp_updated = False

                x, y = self.ref_pt[0]
                self.width, self.height = self.ref_pt[1][0] - x, self.ref_pt[1][1] - y
                self.new_trackers.append(Tracker((y, x), self.width, self.height, self.prev_frame))
                self.new_trackers[-1].set_frame(self.frames[-1])
                continue

            if not paused and not self.cropping:
                self.process(self.image)

            for frame_object in self.frames:
                frame_object.draw(self.image)

            fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

            cv.putText(self.image, str([frame_counter, int(fps)]),
                       (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv.putText(self.image, str([frame_counter, int(fps)]),
                       (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv.imshow('image', self.image)
            if write:
                self.vid_out.write(self.image)

        self.vid_out.release()
        self.vid.release()
        cv.destroyAllWindows()

    def process(self, image):
        for tracker in self.new_trackers:
            tracker.update(image)


def start(filename):
    print('Push left mouse button, select a frame, then hit X key. \n'
          'Reselect live with left mouse button, '
          'hit Z to pause the video, '
          'X to continue, '
          'C to remove all previous targets and '
          'D to remove last target.')
    process = ProcessImage(filename)
    process.loop()
    # process.loop(write=True)


if __name__ == '__main__':
    # start with webcam
    start(0)
