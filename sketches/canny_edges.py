import numpy as np
import cv2 as cv


def main():

    # open webcam
    stream = cv.VideoCapture(0)

    grayscale = False

    while True:
        try:

            _, frame = stream.read()

            if grayscale:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                canny_edges = cv.Canny(frame, 100, 100)

            else:

                blue_channel = np.zeros(frame.shape, dtype=np.uint8)
                red_channel = np.zeros(frame.shape, dtype=np.uint8)
                green_channel = np.zeros(frame.shape, dtype=np.uint8)

                blue_channel[:, :, 0] = frame[:, :, 0]
                green_channel[:, :, 1] = frame[:, :, 1]
                red_channel[:, :, 2] = frame[:, :, 2]

                canny_edges = np.zeros(frame.shape, dtype=np.uint8)
                canny_edges[:, :, 0] = cv.Canny(blue_channel, 100, 100)
                canny_edges[:, :, 1] = cv.Canny(green_channel, 100, 100)
                canny_edges[:, :, 2] = cv.Canny(red_channel, 100, 100)

            stack_frame = np.concatenate((frame, canny_edges), axis=1)
            cv.imshow('stack', stack_frame)

            if cv.waitKey(1) == ord('q'):
                print('release')
                break

        except ValueError:
            print('an exception occurred. (ValueError)')
            break

    cv.destroyAllWindows()
    pass


if __name__ == '__main__':
    main()
