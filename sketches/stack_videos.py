import cv2 as cv
import numpy as np


# Gets and concatenates two videos/streams


def get_sources(filename_1, filename_2):

    stream_1 = cv.VideoCapture(filename_1)
    stream_2 = cv.VideoCapture(filename_2)

    return stream_1, stream_2


def show_stack(stream_1, stream_2, axis=1):

    while True:
        try:

            if cv.waitKey(10) == ord('q'):
                print('release')
                break

            _, frame_1 = stream_1.read()
            _, frame_2 = stream_2.read()

            print('frame_1', frame_1.shape)
            print('frame_2', frame_2.shape)

            cv.imshow('frame_1', frame_1)
            cv.imshow('frame_2', frame_2)

            # stack_frame = np.concatenate((frame_1, frame_2), axis=axis)
            # cv.imshow('stack', stack_frame)

        except ValueError:
            print('an exception occurred. (ValueError)')
            print('looks like one of the sources is missing. Make sure you typed in the right path.')
            break

    stream_1.release()
    stream_2.release()
    cv.destroyAllWindows()

    return 0


def main():

    # webcam
    filename_1 = 1

    # a video file
    filename_2 = 'vid.mp4'
    # filename_2 = 1

    print(
        'file_1: ' + ('' if type(filename_1) is not int else 'webcam ') + str(filename_1) + '; ' +
        'file_2: ' + ('' if type(filename_2) is not int else 'webcam ') + str(filename_2) + '; print "q" to exit.'
    )

    align = {
        'horizontally': 1,
        'vertically': 0
    }

    stream_1, stream_2 = get_sources(filename_1, filename_2)
    show_stack(stream_1, stream_2, axis=align['horizontally'])


if __name__ == '__main__':
    main()
