import numpy as np
import cv2 as cv


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
