# we are going to use open cv2
# we also need numpy
import cv2
import numpy as np

def draw_the_lines(image, lines):
    # create the distinct image for the lines
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lines_image, (x1, y1 ), (x2, y2), (255, 0, 0), thickness=3)


    # finally we have to merge the images with lines
    image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.0)
    return image_with_lines

def region_of_interest(image, region_points):
    # we are going to replace pixels with 0(black) - the regions we are not interested
    mask = np.zeros_like(image)
    # the region we are interested is in lower triangle - 255 white pixel values
    # we are going to fill the triangle within the given points with 255 values
    cv2.fillPoly(mask, region_points, 255)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image





# def the get detected function
def get_detected_lanes(images):
    # height and width of the image
    (height, width) = (images.shape[0], images.shape[1])

    # as cv process the image in gar so convert them to grayscale
    gray_image = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

    # now we have to do edge detection using Canny's Algorithm
    canny_image = cv2.Canny(gray_image, 100, 120)
    # two thresholds --- lower and upper threshold

    # we are interested in the lower part of footage beacuse there are the driving lanes
    region_of_interest_vertices = [
        (0, height),   #bottom left corner of image
        (width/2, height*0.65),   #more than the center of image
        (width, height)   #bottom right corner of image
    ]

    # we can get unrelevant part of image
    # we just keep the lower triangle region
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    # use line detection algorithm
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]),
                           minLineLength=40, maxLineGap=150)

    # draw the lines on image
    image_with_lines = draw_the_lines(images, lines)
    return image_with_lines

# video file that we like to open or test
# video = several frames (images shown after one another)
video = cv2.VideoCapture('lane_detection_video.mp4')

# when the video is open 1st thing is to read the frames on one by one basis
while video.isOpened():

    is_grabbed, frame = video.read()

    # now the video is at end of the video
    if not is_grabbed:
        break       # break the while loop

    frame = get_detected_lanes(frame)
    # otherwise show the image
                   #title              using frames on one by one basis
    cv2.imshow('Lane Detection Video', frame)
    # video is too fast
    # application of frames is too fast that's why we need to delay the application
    cv2.waitKey(20)



# after coming out of while loop or when the video is ended
video.release()
cv2.destroyAllWindows()