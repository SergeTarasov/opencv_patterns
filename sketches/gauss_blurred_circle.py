import cv2 as cv
import numpy as np


class ProcessImage:

    def __init__(self):
        self.x, self.y = 200, 200

        self.image = cv.imread('../media/sky.jpg')
        # self.image = cv.resize(self.image, None, fx=0.2, fy=0.2, interpolation=cv.INTER_CUBIC)
        self.down = False
        
        self.margin = 10
        self.radius = 100
        cv.namedWindow('image')
        cv.setMouseCallback('image', self.click_and_crop)

    def click_and_crop(self, event, x, y, flags, param):

        if event == cv.EVENT_LBUTTONDOWN or self.down:
            self.down = True
            self.x, self.y = x, y

        if event == cv.EVENT_LBUTTONUP:
            self.down = False

        if event == cv.EVENT_MOUSEWHEEL:
            if flags < 0:
                self.radius += 2
            else:
                self.radius = max(10, self.radius - 2)

    def loop(self):

        while True:
            res_image = np.copy(self.image)
            center = (self.x, self.y)

            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            
            cv.circle(mask, center, self.radius, (255, 255, 255), -1)
#             cv.rectangle(mask,
#                          (max(0, self.x - self.radius), self.y - 20),
#                          (max(0, self.x + self.radius), self.y + 20),
#                          (255, 255, 255), -1)
            
            gauss = cv.GaussianBlur(
                res_image[
                    max(0, self.y-self.radius-1):self.y+self.radius+self.margin,
                    max(0, self.x-self.radius-1):self.x+self.radius+self.margin
                ],
                (77, 77),
                19,
                19
            )
            np.clip(gauss.astype(dtype=np.uint16)+20, 0, 255, out=gauss)
            gauss_res = np.zeros(self.image.shape, dtype=np.uint8)
            gauss_res[
                max(0, self.y-self.radius-1):self.y+self.radius+self.margin,
                max(0, self.x-self.radius-1):self.x+self.radius+self.margin
            ] = gauss

            res_image = cv.bitwise_not(-gauss_res, res_image, mask=mask)

            cv.imshow('image', res_image)

            if cv.waitKey(1) == ord('q'):
                break

        cv.destroyAllWindows()


def main():
    print('drag blurred area with the mouse, vary the radius with the mousewheel. "q" to quit.')
    
    process = ProcessImage()
    process.loop()


if __name__ == '__main__':
    main()
