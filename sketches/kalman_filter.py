import cv2 as cv
import numpy as np


def opencv_kalman(filename):
    stream = cv.VideoCapture(filename)

    __filter__ = cv.KalmanFilter(4, 2, 0)

    __filter__.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)

    __filter__.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)

    __filter__.processNoiseCov = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32) * 1e-4

    __filter__.measurementNoiseCov = 1 * np.array([[1, 0],
                                                   [0, 1]], np.float32)

    __filter__.errorCovPost = 1. * np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)

    __filter__.predict()
    while True:

        if cv.waitKey(1) == ord('q'):
            break

        try:
            _, frame = stream.read()

            features = cv.goodFeaturesToTrack(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), 1, 0.1, 2)
            feature_location = features[0].reshape((2, 1)).astype(dtype=np.float32)

            __filter__.correct(feature_location)
            __filter__.predict()

            filter_location = __filter__.statePost[:2]

            cv.drawMarker(frame, tuple(feature_location.astype(dtype=int).T.tolist()[0]), [0, 128, 0])
            cv.drawMarker(frame, tuple(filter_location.astype(dtype=int).T.tolist()[0]), [0, 0, 200])
            cv.imshow('stream ' + str(filename), frame)

        except ValueError:
            print('ValueError occurred')
            break

    print('release')
    stream.release()


def main():
    filename = 0
    opencv_kalman(filename)


if __name__ == '__main__':
    main()
