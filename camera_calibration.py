import numpy as np
import cv2
import glob
import pandas as pd
import os
import argparse

def print_calibfile(calib_filename: str):
    print(f'calib_filename: {calib_filename}')
    with np.load(calib_filename) as X:
        mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
        print('mtx:')
        print(pd.DataFrame(mtx))
        print('dist:')
        print(pd.DataFrame(dist))
        print(f'rvecs: {len(rvecs)} images')
        for rvec in rvecs:
            print(pd.DataFrame(rvec))
        print(f'tvecs: {len(tvecs)} images')
        for tvec in tvecs:
            print(pd.DataFrame(tvec))

def do_calibration(image_dir: str, calib_filename: str, saveall: bool):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 縦7横10
    # checkerboard_4x4_a3.aiをA3に印刷し、横持ちで撮影する
    objp = np.zeros((10*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:10].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    images = glob.glob(os.path.join(image_dir, 'input', '*.jpg'))
    if len(images) == 0:
        print('No images found')
        return
    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7,10),None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, (7,10), corners2,ret)
            cv2.imshow('img',img)
            _, filename = os.path.split(fname)
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_marked{ext}"
            new_dir = os.path.join(image_dir, 'output')
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            new_fname = os.path.join(new_dir, new_filename)
            print(new_fname)
            cv2.imwrite(new_fname, img)
            cv2.waitKey(500)
        else:
            print('not found')
    # calibration
    ret2, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(f'ret2: {ret2}')
    print('mtx:')
    print(pd.DataFrame(mtx))
    print('dist:')
    print(pd.DataFrame(dist))
    print(f'rvecs: {len(rvecs)} images')
    for rvec in rvecs:
        print(pd.DataFrame(rvec))
    print(f'tvecs: {len(tvecs)} images')
    for tvec in tvecs:
        print(pd.DataFrame(tvec))
    print('----------')
    if saveall:
        np.savez(calib_filename, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    else:
        np.savez(calib_filename, mtx=mtx, dist=dist, rvecs=[], tvecs=[])
    # calc reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("total error: ", mean_error/len(objpoints))
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', help='image directory', type=str)
    parser.add_argument('--calibfile', help='output calibration filename', type=str, required=True)
    parser.add_argument('--saveall', help='save all images\' tvecs and rvecs', action='store_true', default=False)
    args = parser.parse_args()
    if args.image_dir is None:
        print_calibfile(args.calibfile)
    else:
        do_calibration(args.image_dir, args.calibfile, args.saveall)

if __name__ == '__main__':
    main()
