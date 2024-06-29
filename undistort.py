import numpy as np
import cv2
import argparse

def do_undistort1(calib_file: str, input_file: str, output_file: str):
    with np.load(calib_file) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
        img = cv2.imread(input_file)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(output_file,dst)

def do_undistort2(calib_file: str, input_file: str, output_file: str):
    with np.load(calib_file) as X:
        mtx, dist = [X[i] for i in ('mtx', 'dist')]
        img = cv2.imread(input_file)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # undistort
        mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
        dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(output_file,dst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibfile', help='output calibration filename', type=str, required=True)
    parser.add_argument('--input', help='input image', type=str)
    parser.add_argument('--output', help='output ilename', type=str)
    parser.add_argument('--method', help='method', type=int, default=1)
    args = parser.parse_args()
    if args.method == 1:
        do_undistort1(args.calibfile, args.input, args.output)
    else:
        do_undistort2(args.calibfile, args.input, args.output)

if __name__ == '__main__':
    main()
