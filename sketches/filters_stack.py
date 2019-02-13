import numpy as np
import cv2 as cv


# BGR stack for now


def main():
    # open webcam
    stream = cv.VideoCapture(0)

    while True:
        try:
            _, frame = stream.read()

            blue_channel = np.zeros(frame.shape, dtype=np.uint8)
            red_channel = np.zeros(frame.shape, dtype=np.uint8)
            green_channel = np.zeros(frame.shape, dtype=np.uint8)

            blue_channel[:, :, 0] = frame[:, :, 0]
            green_channel[:, :, 1] = frame[:, :, 1]
            red_channel[:, :, 2] = frame[:, :, 2]

            upper_stack_frame = np.concatenate((frame, red_channel), axis=1)
            lower_stack_frame = np.concatenate((blue_channel, green_channel), axis=1)
            stack_frame = np.concatenate((upper_stack_frame, lower_stack_frame), axis=0)
            stack_frame = cv.resize(stack_frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)



            cv.imshow('stack', stack_frame)

            if cv.waitKey(1) == ord('q'):
                print('release')
                break

        except ValueError:
            print('an exception occurred. (ValueError)')
            print('looks like one of the sources is missing. Make sure you typed in the right path.')
            break

    cv.destroyAllWindows()

    return 0


if __name__ == '__main__':
    main()
