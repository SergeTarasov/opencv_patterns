import cv2 as cv


def fast_feature_detection(filename):
    stream = cv.VideoCapture(filename)

    fast = cv.FastFeatureDetector_create(threshold=70)

    while True:

        if cv.waitKey(1) == ord('q'):
            break

        try:
            _, frame = stream.read()

            features = fast.detect(frame)
            cv.drawKeypoints(frame, features, frame, color=[0, 250, 0])

            cv.imshow('stream ' + str(filename), frame)

        except ValueError:
            print('ValueError occurred')
            break

    print('release')
    stream.release()


def main():

    # webcam
    filename = 0

    fast_corners_detection(filename)


if __name__ == '__main__':
    main()
