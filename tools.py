import numpy as np
import time
import cv2
import copy
import os
import os.path as path
import imageio
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import mpl_toolkits.mplot3d as plt3d
from numpy.linalg import inv
from utils import transformations as tfms
from scipy.interpolate import interp1d
import math
import random
# import train_network
from networks import generators
from networks import mynet
from networks import p3d
from networks import densenet
from networks import resnet
from networks import resnext
import torch
import torch.nn as nn
import sys
import SimpleITK as sitk

mask_img = cv2.imread('data/US_mask.png', 0)

frames_folder = '/home/guoh9/tmp/US_vid_frames'
pos_folder = '/home/guoh9/tmp/US_vid_pos'


def load_frames_as_volume(frames_dir):
    # case_path = path.join('/home/guoh9/tmp/US_new_frames', case_id)
    frames_list = os.listdir(frames_dir)
    frames_list.sort()

    volume = []
    for frame_name in frames_list:
        frame_path = path.join(frames_dir, frame_name)
        frame_img = cv2.imread(frame_path, 0)
        # frame_img = cv2.resize(frame_img, (820, 616))
        # print(frame_img.shape)
        volume.append(frame_img)
    # volume.append(frame_img)
    volume = np.asarray(volume)
    print('volume shape {}'.format(volume.shape))
    # time.sleep(30)
    # return volume[:10, :, :]
    return volume

def read_list(list_path):
    mylist = []
    with open(list_path, 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            mylist.append(currentPlace)
    return mylist

def read_aurora(file_path):
    """
    Read the Aurora position file and formatly reorganize the shape
    :param file_path: path of Aurora position file
    :return: (frame_number * 9) matrix, each row is a positioning vector
    """
    file = open(file_path, 'r')
    lines = file.readlines()
    pos_np = []
    for line_index in range(1, len(lines) - 1):  # exclude the first line and last line
        line = lines[line_index]
        values = line.split()
        values_np = np.asarray(values[1:]).astype(np.float32)
        pos_np.append(values_np)
    pos_np = np.asarray(pos_np)
    return pos_np


def save_all_aurora_pos():
    """
    This function uses read_aurora function to convert Aurora.pos file into (N x 9) matrix
    Save such txt files for all 640 cases
    """
    check_folder = '/home/guoh9/tmp/US_vid_frames'
    project_folder = '/zion/common/data/uronav_data'
    dst_folder = '/home/guoh9/tmp/US_vid_pos'
    case_list = os.listdir(check_folder)
    case_list.sort()

    for case_index in range(len(case_list)):
        case_id = case_list[case_index]

        pos_path = path.join(project_folder, case_id, '{}_Aurora.pos'.format(case_id))
        pos_np = read_aurora(file_path=pos_path)
        # print(pos_np.shape)

        dst_path = path.join(dst_folder, '{}.txt'.format(case_id))
        np.savetxt(dst_path, pos_np)
        print('{} {} saved'.format(case_id, pos_np.shape))
    print('ALL FINISHED')


def save_vid_gifs():
    """
    Convert the frames of video to a gif
    """
    project_folder = '/home/guoh9/tmp/US_vid_frames'
    dst_folder = '/home/guoh9/tmp/US_vid_gif'
    case_list = os.listdir(project_folder)
    case_list.sort()
    kargs = {'duration': 0.05}

    for case in case_list:
        case_folder = os.path.join(project_folder, case)
        frames_list = os.listdir(case_folder)
        frames_list.sort()

        imgs = []
        for frame in frames_list:
            frame_path = path.join(case_folder, frame)
            frame_img = cv2.imread(frame_path)
            imgs.append(frame_img)
        imageio.mimsave(path.join(dst_folder, '{}.gif'.format(case)), imgs, **kargs)
        print('{}.gif saved'.format(case))
    print('ALL CASES FINISHED!!!')

def save_vid_1frame():
    """
    Convert the frames of video to a gif
    """
    project_folder = '/home/guoh9/tmp/US_vid_frames'
    dst_folder = '/home/guoh9/tmp/US_1frame'


    for status in ['train', 'val', 'test']:
        case_list = os.listdir(path.join(project_folder, status))
        case_list.sort()

        for case in case_list:
            case_folder = os.path.join(project_folder, status, case)
            frames_list = os.listdir(case_folder)
            frames_list.sort()
            # print(frames_list)
            # time.sleep(30)

            frame1 = cv2.imread(path.join(case_folder, frames_list[0]), 0)
            cv2.imwrite(path.join(dst_folder, '{}.jpg'.format(case)), frame1)
            print('{}.gif saved'.format(case))
    print('ALL CASES FINISHED!!!')
    time.sleep(30)


def segmentation_us(input_img):
    # mask_img = cv2.imread('data/US_mask.png', 0)
    # mask_img[mask_img > 50] = 255
    # mask_img[mask_img <= 50] = 0
    #
    # # input_img[mask_img > 50] = 255
    # input_img[mask_img <= 50] = 0
    #
    # cv2.imshow('mask', input_img)
    # cv2.waitKey(0)

    img = np.log2(input_img, dtype=np.float32)
    img = cv2.medianBlur(img, 5)
    ret, thresh = cv2.threshold(img, 0.5, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed_copy = copy.copy(closed)
    cv2.imwrite('closed.jpg', closed)


def mask_us(input_img):
    """
    Use the manually created mask to segment useful US areas
    :param input_img:
    :return: masked US image
    """
    # mask_img[mask_img > 50] = 255
    # mask_img[mask_img <= 50] = 0

    # input_img[mask_img > 50] = 255
    input_img[mask_img <= 20] = 0
    return input_img


def params_to_mat44(trans_params, cam_cali_mat):
    """
    Transform the parameters in Aurora files into 4 x 4 matrix
    :param trans_params: transformation parameters in Aurora.pos. Only the last 7 are useful
    3 are translations, 4 are the quaternion (x, y, z, w) for rotation
    :return: 4 x 4 transformation matrix
    """
    if trans_params.shape[0] == 9:
        trans_params = trans_params[2:]

    translation = trans_params[:3]
    quaternion = trans_params[3:]

    """ Transform quaternion to 3 x 3 rotation matrix, get rid of unstable scipy codes"""
    # r_mat = R.from_quat(quaternion).as_matrix()
    # print('r_mat\n{}'.format(r_mat))

    new_quat = np.zeros((4,))
    new_quat[0] = quaternion[-1]
    new_quat[1:] = quaternion[:3]
    r_mat = tfms.quaternion_matrix(quaternion=new_quat)[:3, :3]
    # print('my_mat\n{}'.format(r_mat))

    trans_mat = np.zeros((4, 4))
    trans_mat[:3, :3] = r_mat
    trans_mat[:3, 3] = translation
    trans_mat[3, 3] = 1

    trans_mat = np.dot(cam_cali_mat, trans_mat)
    trans_mat = inv(trans_mat)

    return trans_mat


def params2corner_pts(params, cam_cali_mat, input_img=np.ones((224, 224)), shrink=1):
    """
    Transform the Aurora params to corner points coordinates of each frame
    :param params: slice_num x 7(or 9) params matrix
    :param input_img: just use for size
    :return: slice_num x 4 x 3. 4 corner points 3d coordinates (x, y, z)
    """
    h, w = input_img.shape
    corner_pts = np.asarray([[-h, 0, 0],
                             [-h, -w, 0],
                             [0, -w, 0],
                             [0, 0, 0]])

    corner_pts = np.asarray([[-h*(1+shrink)/2, -w*(1-shrink)/2, 0],
                             [-h*(1+shrink)/2, -w*(1+shrink)/2, 0],
                             [-h*(1-shrink)/2, -w*(1+shrink)/2, 0],
                             [-h*(1-shrink)/2, -w*(1-shrink)/2, 0]])

    corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
    corner_pts = np.transpose(corner_pts)

    transformed_pts = []
    for frame_id in range(params.shape[0]):
        trans_mat = params_to_mat44(trans_params=params[frame_id, :],
                                    cam_cali_mat=cam_cali_mat)
        transformed_corner_pts = np.dot(trans_mat, corner_pts)
        # print('transformed_corner_pts shape {}'.format(transformed_corner_pts.shape))
        # print(transformed_corner_pts)

        dist1 = np.linalg.norm(transformed_corner_pts[:3, 0] - transformed_corner_pts[:3, 1]) * shrink
        dist2 = np.linalg.norm(transformed_corner_pts[:3, 1] - transformed_corner_pts[:3, 2]) * shrink
        scale_ratio = (dist2 / input_img.shape[0] + dist1 / input_img.shape[1]) / 2
        transformed_corner_pts = transformed_corner_pts / scale_ratio

        # dist3 = np.linalg.norm(transformed_corner_pts[:3, 2] - transformed_corner_pts[:3, 3])
        # dist4 = np.linalg.norm(transformed_corner_pts[:3, 3] - transformed_corner_pts[:3, 0])
        # print(dist1, dist2, dist3, dist4)

        transformed_corner_pts = np.moveaxis(transformed_corner_pts[:3, :], 0, 1)
        transformed_pts.append(transformed_corner_pts)
    transformed_pts = np.asarray(transformed_pts)
    return transformed_pts

def mats2pts(mats, input_img=np.ones((224, 224)), shrink=1):
    """
    Transform the Aurora params to corner points coordinates of each frame
    :param params: slice_num x 7(or 9) params matrix
    :param input_img: just use for size
    :return: slice_num x 4 x 3. 4 corner points 3d coordinates (x, y, z)
    """
    h, w = input_img.shape
    corner_pts = np.asarray([[-h, 0, 0],
                             [-h, -w, 0],
                             [0, -w, 0],
                             [0, 0, 0]])

    corner_pts = np.asarray([[-h*(1+shrink)/2, -w*(1-shrink)/2, 0],
                             [-h*(1+shrink)/2, -w*(1+shrink)/2, 0],
                             [-h*(1-shrink)/2, -w*(1+shrink)/2, 0],
                             [-h*(1-shrink)/2, -w*(1-shrink)/2, 0]])

    corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
    corner_pts = np.transpose(corner_pts)

    transformed_pts = []
    for frame_id in range(mats.shape[0]):
        trans_mat = mats[frame_id, :, :]
        transformed_corner_pts = np.dot(trans_mat, corner_pts)
        # print('transformed_corner_pts shape {}'.format(transformed_corner_pts.shape))
        # print(transformed_corner_pts)

        dist1 = np.linalg.norm(transformed_corner_pts[:3, 0] - transformed_corner_pts[:3, 1]) * shrink
        dist2 = np.linalg.norm(transformed_corner_pts[:3, 1] - transformed_corner_pts[:3, 2]) * shrink
        scale_ratio = (dist2 / input_img.shape[0] + dist1 / input_img.shape[1]) / 2
        transformed_corner_pts = transformed_corner_pts / scale_ratio

        # dist3 = np.linalg.norm(transformed_corner_pts[:3, 2] - transformed_corner_pts[:3, 3])
        # dist4 = np.linalg.norm(transformed_corner_pts[:3, 3] - transformed_corner_pts[:3, 0])
        # print(dist1, dist2, dist3, dist4)

        transformed_corner_pts = np.moveaxis(transformed_corner_pts[:3, :], 0, 1)
        transformed_pts.append(transformed_corner_pts)
    transformed_pts = np.asarray(transformed_pts)
    return transformed_pts

def params2mats(input_params, cam_cali_mat):
    """Covert N*7(9) params into N*4*4 transformation mats

    Args:
        input_params ([type]): [description]
        cam_cali_mat ([type]): [description]

    Returns:
        [type]: [description]
    """
    out_mats = []
    for frame_id in range(input_params.shape[0]):
        trans_mat = params_to_mat44(trans_params=input_params[frame_id, :],
                                    cam_cali_mat=cam_cali_mat)
        out_mats.append(trans_mat)
    out_mats = np.asarray(out_mats)
    return out_mats

def errorMats(gt_dof, pd_dof):
    """ lets just study the error between every two frames

    Args:
        gt_dof ([type]): [description]
        pd_dof ([type]): [description]

    Returns:
        [type]: [description]
    """
    error_mats = []
    for i in range(gt_dof.shape[0]):
        mat_label = dof2mat_np(input_dof=gt_dof[i, :])
        mat_predi = dof2mat_np(input_dof=pd_dof[i, :])
        mat_error = np.dot(np.linalg.inv(mat_label), mat_predi)
        error_mats.append(mat_error)
    error_mats = np.asarray(error_mats)
    # print('error_mats {}'.format(error_mats.shape))
    # print('gt_dof {}, pd_dof {}'.format(gt_dof.shape, pd_dof.shape))
    # sys.exit()
    return error_mats

def frameError(error_mats, spacing, img_size):
    """ Compute the frame error, eliminates the accumulative error

    Args:
        error_mats ([type]): [description]
        spacing ([type]): [description]
        img_size ([type]): [description]

    Returns:
        [type]: [description]
    """
    print('error_mats {}, spacing {}'.format(error_mats.shape, spacing))
    frame_error = []
    for i in range(error_mats.shape[0]):
        error = computeError(mat_error=error_mats[i, :, :], spacing=spacing, 
                             img_size=img_size)
        frame_error.append(error)
    frame_error = np.asarray(frame_error)
    frame_error = np.mean(frame_error)
    return frame_error

def computeError(mat_error, spacing, img_size):
    """[summary]

    Args:
        mat_error ([numpy]): 4x4 numpy mat, difference mat between GT and Prediction
        spacing ([float]): spacing of original usvolume
        img_size ([tuple 2]): tuple of numpy frame size, for defining corner pts

    Returns:
        [float]: error in mm
    """
    # print('mat_error\n{}'.format(mat_error))
    # print('spacing\n{}'.format(spacing))
    # print('img_size\n{}'.format(img_size))

    h, w = img_size
    corner_pts = []
    for x in [-h/2, h/2]:
        for y in [-w/2, w/2]:
            corner_pts.append([x, y, 0, 1])
    corner_pts = np.asarray(corner_pts)
    corner_pts = np.transpose(corner_pts)
    # print('corner_pts\n{}'.format(corner_pts))

    trans_corner_pts = np.dot(mat_error, corner_pts)
    # print('trans_corner_pts\n{}'.format(trans_corner_pts))

    dist = np.linalg.norm(corner_pts - trans_corner_pts, axis=0)
    # print('dist\n{}'.format(dist))

    error_mm = spacing * np.mean(dist)
    # print('error {} mm'.format(error_mm))
    
    # sys.exit()
    return error_mm

def dof2mat_np(input_dof, scale=False):
    """ Transfer degrees to euler """
    dof = input_dof
    # print('deg {}'.format(dof[3:6]))
    dof[3:6] = dof[3:6] * (2 * math.pi) / 360.0
    # print('rad {}'.format(dof[3:6]))


    rot_mat = tfms.euler_matrix(dof[5], dof[4], dof[3], 'rzyx')[:3, :3]

    mat44 = np.identity(4)
    mat44[:3, :3] = rot_mat
    mat44[:3, 3] = dof[:3]

    if scale:
        scales = dof[6:]
        mat_scale = np.diag([scales[1], scales[0], scales[2], 1])
        mat44 = np.dot(mat44, np.linalg.inv(mat_scale))
    # print('mat_scale\n{}'.format(mat_scale))
    # print('recon mat\n{}'.format(mat44))
    # sys.exit()
    return mat44

def evaluate_dist(pts1, pts2, resolution=0.2):
    """
    points input formats are frame_num x 4 (corner_points) x 3 (xyz)
    :param pts1:
    :param pts2:
    :param resolution:
    :return: The average Euclidean distance between all points pairs, times 0.2 is mm
    """
    error = np.square(pts1 - pts2)
    error = np.sum(error, axis=2)
    error = np.sqrt(error)
    error = np.mean(error) * resolution
    return error

def seq_length(pts, resolution=0.2):
    pts = np.mean(pts, axis=1)

    dist = np.square(pts[1:, :] - pts[:-1, :])
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    dist = np.mean(dist) * resolution

    return dist

def drfit_rate(gt_pts, pd_pts, resolution=0.2):
    # print('pts1 {}'.format(pts1.shape))
    # print('pts2 {}'.format(pts2.shape))

    gt_len = seq_length(gt_pts, resolution=resolution)
    pd_len = seq_length(pd_pts, resolution=resolution)

    d_rate = abs(gt_len-pd_len)/gt_len * 100

    # print('gt_len {:.2f}, pd_len {:.2f}'.format(gt_len, pd_len))
    # print('d_rate {:.2f} %'.format(d_rate))
    # sys.exit()
    return d_rate

def drfit_rate2(gt_pts, pd_pts, resolution=0.2):
    # print('pts1 {}'.format(pts1.shape))
    # print('pts2 {}'.format(pts2.shape))

    gt_len = seq_length(gt_pts, resolution=resolution)
    drift = final_drift(pts1=gt_pts[-1, :, :], pts2=pd_pts[-1, :, :], resolution=resolution)
    # pd_len = seq_length(pd_pts, resolution=resolution)

    d_rate = drift/gt_len * 100

    # print('gt_len {:.2f}, pd_len {:.2f}'.format(gt_len, pd_len))
    # print('d_rate {:.2f} %'.format(d_rate))
    # sys.exit()
    return d_rate


def final_drift(pts1, pts2, resolution=0.2):
    # print(pts1.shape, pts2.shape)
    center_pt1 = np.mean(pts1, axis=0)
    center_pt2 = np.mean(pts2, axis=0)
    dist = np.linalg.norm(center_pt1 - center_pt2) * resolution
    return dist



def plot_2d_in_3d(trans_params, frame_color='b', input_img=np.ones((480, 640))):
    """
    Plot a 2D frame into 3D space for sequence visualization
    :param input_img: input image frame
    :param trans_params: Aurora position file line of position
    """
    h, w = input_img.shape
    corner_pts = np.asarray([[0, 0, 0],
                             [0, w, 0],
                             [h, w, 0],
                             [h, 0, 0]])
    corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
    corner_pts = np.transpose(corner_pts)
    print('imgshape {}'.format(input_img.shape))
    print('corner_pts:\n{}'.format(corner_pts))

    trans_mat = params_to_mat44(trans_params=trans_params)
    print('trans_mat:\n{}'.format(trans_mat))

    transformed_corner_pts = np.dot(trans_mat, corner_pts)
    print('transformed_corner_pts:\n{}'.format(transformed_corner_pts))
    # dst = np.linalg.norm(transformed_corner_pts[:, 0] - transformed_corner_pts[:, 2])
    # print(dst)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # w_weights, h_weights = np.meshgrid(np.linspace(0, 1, w),
    #                                    np.linspace(0, 1, h))
    # X = (1 - w_weights - h_weights) * transformed_corner_pts[0, 0] + \
    #     h_weights * transformed_corner_pts[0, 3] + w_weights * transformed_corner_pts[0, 1]
    # Y = (1 - w_weights - h_weights) * transformed_corner_pts[1, 0] + \
    #     h_weights * transformed_corner_pts[1, 3] + w_weights * transformed_corner_pts[1, 1]
    # Z = (1 - w_weights - h_weights) * transformed_corner_pts[2, 0] + \
    #     h_weights * transformed_corner_pts[2, 3] + w_weights * transformed_corner_pts[2, 1]
    # input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
    # input_img = input_img / 255
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    #                 facecolors=input_img)
    # plt.show()
    # time.sleep(30)
    for i in range(-1, 3):
        xs = transformed_corner_pts[0, i], transformed_corner_pts[0, i+1]
        ys = transformed_corner_pts[1, i], transformed_corner_pts[1, i+1]
        zs = transformed_corner_pts[2, i], transformed_corner_pts[2, i+1]
        # line = plt3d.art3d.Line3D(xs, ys, zs)
        # ax.add_line(line)
        ax.plot(xs, ys, zs, color=frame_color)

    # ax.plot(pt1, pt2, color='b')
    # ax.scatter()
    # ax.plot(transformed_corner_pts[:3, 0], transformed_corner_pts[:3, 1], color='b')
    # ax.plot(transformed_corner_pts[:3, 1], transformed_corner_pts[:3, 2], color='b')
    # ax.plot(transformed_corner_pts[:3, 2], transformed_corner_pts[:3, 3], color='b')
    # ax.plot(transformed_corner_pts[:3, 3], transformed_corner_pts[:3, 0], color='b')

    plt.show()

def plot_2d_in_3d_test(trans_params1, trans_params2,
                       frame_color='b', input_img=np.ones((480, 640))):
    """
    Plot a 2D frame into 3D space for sequence visualization
    :param input_img: input image frame
    :param trans_params: Aurora position file line of position
    """
    h, w = input_img.shape
    corner_pts = np.asarray([[0, 0, 0],
                             [0, w, 0],
                             [h, w, 0],
                             [h, 0, 0]])
    corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
    corner_pts = np.transpose(corner_pts)
    print('imgshape {}'.format(input_img.shape))
    print('corner_pts:\n{}'.format(corner_pts))

    trans_mat1 = params_to_mat44(trans_params=trans_params1)
    trans_mat2 = params_to_mat44(trans_params=trans_params2)
    print('trans_mat1 shape {}, trans_mat2 shape {}'.format(trans_mat1.shape, trans_mat2.shape))
    print('trans_mat1 shape\n{}\ntrans_mat2 shape\n{}'.format(trans_mat1, trans_mat2))
    # time.sleep(30)

    relative_mat = np.dot(inv(trans_mat1), trans_mat2)

    original_mat2 = np.dot(trans_mat1, relative_mat)
    print('relative_mat\n{}'.format(relative_mat))
    print('original_mat2\n{}'.format(original_mat2))

    transformed_corner_pts = np.dot(trans_mat1, corner_pts)
    print('transformed_corner_pts:\n{}'.format(transformed_corner_pts))
    # dst = np.linalg.norm(transformed_corner_pts[:, 0] - transformed_corner_pts[:, 2])
    # print(dst)

    fig = plt.figure()
    ax = fig.gca(projection='3d')


    for i in range(-1, 3):
        xs = transformed_corner_pts[0, i], transformed_corner_pts[0, i+1]
        ys = transformed_corner_pts[1, i], transformed_corner_pts[1, i+1]
        zs = transformed_corner_pts[2, i], transformed_corner_pts[2, i+1]
        # line = plt3d.art3d.Line3D(xs, ys, zs)
        # ax.add_line(line)
        ax.plot(xs, ys, zs, color=frame_color)

    # ax.plot(pt1, pt2, color='b')
    # ax.scatter()
    # ax.plot(transformed_corner_pts[:3, 0], transformed_corner_pts[:3, 1], color='b')
    # ax.plot(transformed_corner_pts[:3, 1], transformed_corner_pts[:3, 2], color='b')
    # ax.plot(transformed_corner_pts[:3, 2], transformed_corner_pts[:3, 3], color='b')
    # ax.plot(transformed_corner_pts[:3, 3], transformed_corner_pts[:3, 0], color='b')

    plt.show()

def visualize_frames(case_id):
    case_frames_path = path.join(frames_folder, 'Case{:04}'.format(case_id))
    frames_list = os.listdir(case_frames_path)
    frames_list.sort()

    case_pos_path = path.join(pos_folder, 'Case{:04}.txt'.format(case_id))
    case_pos = np.loadtxt(case_pos_path)
    print('frames_list {}, case_pos {}'.format(len(frames_list), case_pos.shape))

    frames_num = case_pos.shape[0]
    colors_R = np.linspace(0, 255, frames_num).astype(np.int16).reshape((frames_num, 1))
    colors_G = np.zeros((frames_num, 1))
    colors_B = np.linspace(255, 0, frames_num).astype(np.int16).reshape((frames_num, 1))

    colors = np.concatenate((colors_R, colors_G, colors_B), axis=1)

    for frame_id in range(frames_num):
        frame_pos = case_pos[frame_id, :]
        frame_color = tuple(colors[frame_id, :])

        time.sleep(30)


class VisualizeSequence():
    def __init__(self, case_id):
        super(VisualizeSequence, self).__init__()
        self.case_id = case_id
        if 1 <= self.case_id <= 71:
            self.data_part = 'test'
        elif 71 < self.case_id <= 140:
            self.data_part = 'val'
        elif 140 < self.case_id <= 747:
            self.data_part = 'train'
        self.case_frames_path = path.join(frames_folder, self.data_part,
                                          'Case{:04}'.format(self.case_id))
        self.frames_list = os.listdir(self.case_frames_path)
        self.frames_list.sort()

        self.cam_cali_mat = np.loadtxt('/zion/common/data/uronav_data/Case{:04}/'
                                       'Case{:04}_USCalib.txt'.format(self.case_id, self.case_id))

        case_pos_path = path.join(pos_folder, 'Case{:04}.txt'.format(self.case_id))
        self.case_pos = np.loadtxt(case_pos_path)
        print('frames_list {}, case_pos {}'.format(len(self.frames_list), self.case_pos.shape))

        self.frames_num = self.case_pos.shape[0]
        colors_R = np.linspace(0, 1, self.frames_num).reshape((self.frames_num, 1))
        colors_G = np.zeros((self.frames_num, 1))
        colors_B = np.linspace(1, 0, self.frames_num).reshape((self.frames_num, 1))

        self.colors = np.concatenate((colors_R, colors_G, colors_B), axis=1)

        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')

        def plot_frame3d(trans_params, frame_color=(255, 0, 0),
                         input_img=np.ones((480, 640)), plot_img=False):
            """
            Plot a 2D frame into 3D space for sequence visualization
            :param frame_color: color of the initial frame, default to be blue
            :param input_img: input image frame
            :param trans_params: Aurora position file line of position
            """
            h, w = input_img.shape
            # corner_pts = np.asarray([[0, 0, 0],
            #                          [0, w, 0],
            #                          [h, w, 0],
            #                          [h, 0, 0]])
            corner_pts = np.asarray([[-h, 0, 0],
                                     [-h, -w, 0],
                                     [0, -w, 0],
                                     [0, 0, 0]])
            corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
            corner_pts = np.transpose(corner_pts)
            print('imgshape {}'.format(input_img.shape))
            print('corner_pts:\n{}'.format(corner_pts))
            print('h {}, w {}'.format(h, w))

            trans_mat = params_to_mat44(trans_params=trans_params,
                                        cam_cali_mat=self.cam_cali_mat)
            # trans_mat = trans_mat.transpose()
            # trans_mat = np.dot(self.cam_cali_mat, trans_mat)
            # trans_mat = inv(trans_mat)
            # trans_mat = np.dot(trans_mat, inv(self.cam_cali_mat))
            # trans_mat = np.dot(trans_mat, self.cam_cali_mat)

            print('trans_mat:\n{}'.format(trans_mat))

            transformed_corner_pts = np.dot(trans_mat, corner_pts)
            # time.sleep(30)
            print('transformed_corner_pts:\n{}'.format(transformed_corner_pts))
            print('transformed_corner_pts shape {}'.format(transformed_corner_pts.shape))
            time.sleep(30)
            # dst = np.linalg.norm(transformed_corner_pts[:, 0] - transformed_corner_pts[:, 2])
            # print(dst)

            for i in range(-1, 3):
                xs = transformed_corner_pts[0, i], transformed_corner_pts[0, i + 1]
                ys = transformed_corner_pts[1, i], transformed_corner_pts[1, i + 1]
                zs = transformed_corner_pts[2, i], transformed_corner_pts[2, i + 1]
                if i == 0 or i == 2:
                    linewidth = 10
                else:
                    linewidth = 1
                self.ax.plot(xs, ys, zs, color=frame_color, lw=linewidth)

            if plot_img:
                w_weights, h_weights = np.meshgrid(np.linspace(0, 1, w),
                                                   np.linspace(0, 1, h))
                X = (1 - w_weights - h_weights) * transformed_corner_pts[0, 0] + \
                    h_weights * transformed_corner_pts[0, 3] + w_weights * transformed_corner_pts[0, 1]
                Y = (1 - w_weights - h_weights) * transformed_corner_pts[1, 0] + \
                    h_weights * transformed_corner_pts[1, 3] + w_weights * transformed_corner_pts[1, 1]
                Z = (1 - w_weights - h_weights) * transformed_corner_pts[2, 0] + \
                    h_weights * transformed_corner_pts[2, 3] + w_weights * transformed_corner_pts[2, 1]
                input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
                input_img = input_img / 255
                self.ax.plot_surface(X, Y, Z, rstride=10, cstride=10, facecolors=input_img)


        for frame_id in range(self.frames_num):
            frame_pos = self.case_pos[frame_id, :]
            frame_color = tuple(self.colors[frame_id, :])
            frame_img = cv2.imread(path.join(self.case_frames_path, '{:04}.jpg'.format(frame_id)), 0)
            frame_img = train_network.data_transform(frame_img)
            plot_frame3d(trans_params=frame_pos, frame_color=frame_color,
                         input_img=frame_img, plot_img=True)
            print('{} frame'.format(frame_id))
        plt.show()


class DofPlot():
    def __init__(self, case_id):
        super(DofPlot, self).__init__()
        self.case_id = case_id
        self.case_frames_path = path.join(frames_folder, 'Case{:04}'.format(self.case_id))
        self.frames_list = os.listdir(self.case_frames_path)
        self.frames_list.sort()

        self.cam_cali_mat = np.loadtxt('/zion/common/data/uronav_data/Case{:04}/'
                                       'Case{:04}_USCalib.txt'.format(self.case_id, self.case_id))

        case_pos_path = path.join(pos_folder, 'Case{:04}.txt'.format(self.case_id))
        self.case_pos = np.loadtxt(case_pos_path)
        print('frames_list {}, case_pos {}'.format(len(self.frames_list), self.case_pos.shape))

        self.frames_num = self.case_pos.shape[0]

        def plot_dof():
            plt.figure()
            colors = ['lightcoral', 'darkorange', 'palegreen',
                      'aqua', 'royalblue', 'violet']
            names = ['tX', 'tY', 'tZ', 'rX', 'rY', 'rZ']
            for dof_id in range(0, self.extracted_dof.shape[1]):
                plt.plot(self.extracted_dof[:, dof_id], color=colors[dof_id],
                         label=names[dof_id])
            plt.legend(loc='upper left')
            # plt.show()
            plot_path = 'figures/dofs/Case{:04}.jpg'.format(self.case_id)
            plt.savefig(plot_path)


        extracted_dof = []
        for frame_id in range(1, self.frames_num):
            this_params = self.case_pos[frame_id, :]
            this_dof = get_6dof_label(trans_params1=self.case_pos[0, :],
                                      trans_params2=this_params,
                                      cam_cali_mat=self.cam_cali_mat,
                                      use_euler=False)
            extracted_dof.append(this_dof)
        self.extracted_dof = np.asarray(extracted_dof)
        plot_dof()
        print('extracted_dof shape {}'.format(self.extracted_dof.shape))

        # for frame_id in range(self.frames_num):
        #     frame_pos = self.case_pos[frame_id, :]
        #     frame_color = tuple(self.colors[frame_id, :])
        #     frame_img = cv2.imread(path.join(self.case_frames_path, '{:04}.jpg'.format(frame_id)), 0)
        #     plot_frame3d(trans_params=frame_pos, frame_color=frame_color,
        #                  input_img=frame_img, plot_img=False)
        #     print('{} frame'.format(frame_id))
        # plt.show()

def relativeDOF(trans_mat2, trans_mat1):
    relative_mat = np.dot(trans_mat2, inv(trans_mat1))

    translations = relative_mat[:3, 3]
    rotations_eulers = np.asarray(tfms.euler_from_matrix(relative_mat))

    rotations_degrees = (rotations_eulers / (2 * math.pi)) * 360

    dof = np.concatenate((translations, rotations_degrees), axis=0)

    return dof

def get_6dof_label(trans_params1, trans_params2, cam_cali_mat):
    """
    Given two Aurora position lines of two frames, return the relative 6 degrees of freedom label
    Aurora position line gives the transformation from the ultrasound tracker to Aurora
    :param trans_params1: Aurora position line of the first frame
    :param trans_params2: Aurora position line of the second frame
    :param cam_cali_mat: Camera calibration matrix of this case, which is the transformation from
    the ultrasound image upper left corner (in pixel) to the ultrasound tracker (in mm).
    :return: the relative 6 degrees of freedom (3 translations and 3 rotations xyz) as training label
    Note that this dof is based on the position of the first frame
    The angles are in degrees, not euler!
    """
    trans_mat1 = params_to_mat44(trans_params1, cam_cali_mat=cam_cali_mat)
    trans_mat2 = params_to_mat44(trans_params2, cam_cali_mat=cam_cali_mat)

    relative_mat = np.dot(trans_mat2, inv(trans_mat1))

    translations = relative_mat[:3, 3]
    rotations_eulers = np.asarray(tfms.euler_from_matrix(relative_mat))

    rotations_degrees = (rotations_eulers / (2 * math.pi)) * 360

    dof = np.concatenate((translations, rotations_degrees), axis=0)

    return dof

def get_6dof_label2(trans_params1, trans_params2, cam_cali_mat):
    """
    Given two Aurora position lines of two frames, return the relative 6 degrees of freedom label
    Aurora position line gives the transformation from the ultrasound tracker to Aurora
    :param trans_params1: Aurora position line of the first frame
    :param trans_params2: Aurora position line of the second frame
    :param cam_cali_mat: Camera calibration matrix of this case, which is the transformation from
    the ultrasound image upper left corner (in pixel) to the ultrasound tracker (in mm).
    :return: the relative 6 degrees of freedom (3 translations and 3 rotations xyz) as training label
    Note that this dof is based on the position of the first frame
    The angles are in degrees, not euler!
    """
    # trans_mat1 = params_to_mat44(trans_params1, cam_cali_mat=cam_cali_mat)
    # trans_mat2 = params_to_mat44(trans_params2, cam_cali_mat=cam_cali_mat)
    trans_mat1, _, _ = params2mat_new(trans_params=trans_params1,
                                    cam_cali_mat=cam_cali_mat,
                                    calibration=True)
    trans_mat2, _, _ = params2mat_new(trans_params=trans_params2,
                                    cam_cali_mat=cam_cali_mat,
                                    calibration=True)

    print('trans_mat1\n{}'.format(trans_mat1))
    print('trans_mat2\n{}'.format(trans_mat2))



    relative_mat = np.dot(trans_mat2, inv(trans_mat1))

    translations = relative_mat[:3, 3]
    rotations_eulers = np.asarray(tfms.euler_from_matrix(relative_mat))

    rotations_degrees = (rotations_eulers / (2 * math.pi)) * 360

    dof = np.concatenate((translations, rotations_degrees), axis=0)

    return dof

def get_6dof_from_mats(trans_mat1, trans_mat2):
    relative_mat = np.dot(trans_mat2, inv(trans_mat1))

    translations = relative_mat[:3, 3]
    rotations_eulers = np.asarray(tfms.euler_from_matrix(relative_mat))

    rotations_degrees = (rotations_eulers / (2 * math.pi)) * 360

    dof = np.concatenate((translations, rotations_degrees), axis=0)

    return dof


def get_next_pos(trans_params1, dof, cam_cali_mat):
    """
    Given the first frame's Aurora position line and relative 6dof, return second frame's position line
    :param trans_params1: Aurora position line of the first frame
    :param dof: 6 degrees of freedom based on the first frame, rotations should be degrees
    :param cam_cali_mat: Camera calibration matrix of this case
    :return: Aurora position line of the second frame
    """
    trans_mat1 = params_to_mat44(trans_params1, cam_cali_mat=cam_cali_mat)
    # trans_mat1 = params2mat_new(trans_params1, cam_cali_mat=cam_cali_mat, calibration=True)[0]
    # print(trans_mat1)
    # sys.exit()
    """ Transfer degrees to euler """
    dof[3:] = dof[3:] * (2 * math.pi) / 360

    rot_mat = tfms.euler_matrix(dof[5], dof[4], dof[3], 'rzyx')[:3, :3]

    relative_mat = np.identity(4)
    relative_mat[:3, :3] = rot_mat
    relative_mat[:3, 3] = dof[:3]

    next_mat = np.dot(inv(cam_cali_mat), inv(np.dot(relative_mat, trans_mat1)))
    quaternions = tfms.quaternion_from_matrix(next_mat)  # wxyz

    next_params = np.zeros(7)
    next_params[:3] = next_mat[:3, 3]
    next_params[3:6] = quaternions[1:]
    next_params[6] = quaternions[0]
    return next_params


def get_next_mat(trans_mat1, dof):
    """
    Given the first frame's Aurora position line and relative 6dof, return second frame's position line
    :param trans_params1: Aurora position line of the first frame
    :param dof: 6 degrees of freedom based on the first frame, rotations should be degrees
    :param cam_cali_mat: Camera calibration matrix of this case
    :return: Aurora position line of the second frame
    """
    """ Transfer degrees to euler """
    dof[3:] = dof[3:] * (2 * math.pi) / 360

    rot_mat = tfms.euler_matrix(dof[5], dof[4], dof[3], 'rzyx')[:3, :3]

    relative_mat = np.identity(4)
    relative_mat[:3, :3] = rot_mat
    relative_mat[:3, 3] = dof[:3]
    # print('ret {}, tran {}'.format(relative_mat.shape, trans_mat1.shape))

    next_mat = np.dot(relative_mat, trans_mat1)
    return next_mat

def mat2params(mat):
    quaternions = tfms.quaternion_from_matrix(mat)  # wxyz

    params = np.zeros(7)
    params[:3] = mat[:3, 3]
    params[3:6] = quaternions[1:]
    params[6] = quaternions[0]
    return params


def smooth_array(input_array1d, smooth_deg=10):
    ori_x = np.linspace(0, input_array1d.shape[0]-1, input_array1d.shape[0])
    print('ori_x\n{}'.format(ori_x))
    print('ori_x shape {}'.format(ori_x.shape))
    ori_y = input_array1d

    p = np.polyfit(ori_x, ori_y, deg=smooth_deg)
    f = np.poly1d(p)

    smoothed = f(ori_x)
    # print('input_array1d\n{}'.format(input_array1d))
    # print('smoothed\n{}'.format(smoothed))
    # time.sleep(30)
    return smoothed


def sample_ids(slice_num, neighbour_num, sample_option='skip', random_reverse_prob=0,
               self_prob=0):
    """
    This function gives different sampling strategies.
    :param slice_num: Number of total slices of a case
    :param neighbour_num: How many slices to serve as one input
    :param sample_option: skip range or normally consecutive
    :param random_reverse_prob: probability of applying random reverse, 0 to be no random reverse
    :return:
    """
    skip_ratio = 3

    if sample_option in ['skip', 'skip_random'] and neighbour_num * skip_ratio > slice_num:
        sample_option = 'normal'

    if sample_option == 'skip':
        start_range = slice_num - skip_ratio * neighbour_num
        if start_range == 0:
            start_id = 0
        else:
            start_id = np.random.randint(0, start_range, 1)[0]
        end_id = start_id + skip_ratio * neighbour_num - 1
        range = np.linspace(start_id, end_id, skip_ratio * neighbour_num)
        np.random.shuffle(range)
        select_ids = np.sort(range[:neighbour_num])
    elif sample_option == 'skip_random':
        ''' ending sample ID is randomly chose from latter half '''
        ''' This function creates more varieties of sampling range'''
        start_range = slice_num - skip_ratio * neighbour_num
        if start_range == 0:
            start_id = 0
        else:
            start_id = np.random.randint(0, start_range, 1)[0]
        end_id = start_id + skip_ratio * neighbour_num - 1
        central_id = int((start_id + end_id) / 2)

        sample_end_id_pool = np.linspace(central_id, end_id, end_id - central_id + 1)
        sample_end_id = int(np.random.choice(sample_end_id_pool, 1)[0])

        sample_ratio = np.linspace(0, 1, neighbour_num)
        select_ids = (sample_ratio * (sample_end_id - start_id) + start_id).astype(np.uint64)
    elif sample_option == 'skip_random_fixed':
        start_range = slice_num - skip_ratio * neighbour_num
        if start_range == 0:
            start_id = 0
        else:
            start_id = np.random.randint(0, start_range, 1)[0]
        frame_gap_choices = [0, 1, 2, 3]
        frame_gap_probs = [0, 0.5, 0.5, 0]
        frame_gap_random = np.random.choice(frame_gap_choices, 1, p=frame_gap_probs)[0]
        select_ids = np.linspace(start=start_id,
                                 stop=start_id + (neighbour_num - 1) * frame_gap_random,
                                 num=neighbour_num, endpoint=True)
        # print(frame_gap_random)
        # print(select_ids)
        # time.sleep(30)
    else:
        start_range = slice_num - neighbour_num
        if start_range == 0:
            start_id = 0
        else:
            start_id = np.random.randint(0, start_range, 1)[0]
        select_ids = np.linspace(start_id, start_id + neighbour_num - 1, neighbour_num)

    if random.uniform(0, 1) < random_reverse_prob:
        select_ids = np.flip(select_ids)

    if random.uniform(0, 1) < self_prob:
        ''' input the same slice for NS times '''
        slice_id = random.randint(0, slice_num-1)
        select_ids = slice_id * np.ones((neighbour_num,))
        # print(select_ids)

    select_ids = select_ids.astype(np.int64)
    # print('selected ids {}'.format(select_ids))
    # select_ids = np.random.shuffle(select_ids)
    # print('shuffled selected ids {}'.format(select_ids))
    return select_ids


def clean_ids():
    """
    Eliminate weird BK scans from all three portions
    """
    project_folder = '/home/guoh9/tmp/US_vid_frames'
    bk_ids = np.loadtxt('infos/BK.txt')
    clean_case_ids = {'train': [], 'val': [], 'test': []}

    for status in ['train', 'val', 'test']:
        case_list = os.listdir(path.join(project_folder, status))
        case_list.sort()

        for case in case_list:
            case_id = int(case[-4:])
            if case_id not in bk_ids:
                clean_case_ids[status].append(case_id)

        np_id = np.asarray(clean_case_ids[status]).astype(np.int64)
        np.savetxt('infos/{}_ids.txt'.format(status), np_id)

    print('clean cases ids finished')
    time.sleep(30)


def test_avg_dof():
    case_id = 10
    case_pos_np = np.loadtxt('/home/guoh9/tmp/US_vid_pos/Case{:04}.txt'.format(case_id))
    case_calib_mat = np.loadtxt('/zion/common/data/uronav_data/Case{:04}/'
                                'Case{:04}_USCalib.txt'.format(case_id, case_id))

    start_id = 0
    ns = 10

    # print(np.around(case_pos_np, decimals=3))

    all_labels = []
    for id in range(start_id, start_id + ns - 1):
        # print('id {}'.format(id))
        pos1 = case_pos_np[id, :]
        pos2 = case_pos_np[id + 1, :]
        print('{}, {}'.format(id, id + 1))
        label = get_6dof_label(trans_params1=pos1, trans_params2=pos2,
                               cam_cali_mat=case_calib_mat)
        all_labels.append(label)
    all_labels = np.asarray(all_labels)
    print('all_labels shape {}'.format(all_labels.shape))

    sum_labels = np.sum(all_labels, axis=0)

    pos_start = case_pos_np[start_id, :]
    pos_end = case_pos_np[start_id + ns - 1, :]
    print('pos_start\n{}'.format(pos_start))
    print('pos_end\n{}'.format(pos_end))

    pos_end_recon = get_next_pos(trans_params1=pos_start, dof=sum_labels,
                                 cam_cali_mat=case_calib_mat)
    print('pos_end_recon\n{}'.format(pos_end_recon))

    time.sleep(30)
    pos1 = case_pos_np[1, 2:]
    pos2 = case_pos_np[10, 2:]
    label = get_6dof_label(trans_params1=pos1, trans_params2=pos2,
                           cam_cali_mat=case_calib_mat)
    recon_params = get_next_pos(trans_params1=pos1, dof=label,
                                cam_cali_mat=case_calib_mat)
    print('labels\n{}'.format(label))
    print('params2\n{}'.format(pos2))
    print('recon_params\n{}'.format(recon_params))


def center_crop():
    folder = '/home/guoh9/tmp/US_vid_frames/train/Case0347'
    frame_list = os.listdir(folder)
    frame_list.sort()

    for i in frame_list:
        frame_path = path.join(folder, i)
        frame_img = cv2.imread(frame_path, 0)
        crop = train_network.data_transform(input_img=frame_img)
        cv2.imwrite('data/crops/{}.jpg'.format(i), crop)
    print('finished')
    time.sleep(30)


def produce_Aurora(case_id):
    original_pos_path = '/zion/common/shared/uronav_data/test/Case{:04}/Case{:04}_Aurora.pos'.format(case_id, case_id)
    results_pos_path = 'results/pos/Case{:04}_Aurora_result.pos'.format(case_id)
    results_pos = np.loadtxt(results_pos_path)

    # if results_pos.shape[1] == 7:
    #     results_pos = np.concatenate((np.zeros((results_pos.shape[0], 2)), results_pos), axis=1)

    file = open(original_pos_path, 'r')
    file_dst = open('results/results_pos/Case{:04}_Aurora_results.pos'.format(case_id), 'a+')

    lines = file.readlines()
    file_dst.write('{}'.format(lines[0]))

    pos_np = []
    for line_index in range(1, len(lines) - 1):  # exclude the first line and last line
        result_index = line_index - 1
        line = lines[line_index]
        values = line.split()
        values_np = np.asarray(values[1:]).astype(np.float32)
        pos_np.append(values_np)

        for fixed_id in range(3):
            file_dst.write('{} '.format(int(values[fixed_id])))

        for params_id in range(results_pos.shape[1]):
            file_dst.write('{:.6f} '.format(results_pos[result_index, params_id]))
        file_dst.write('\n')
    pos_np = np.asarray(pos_np)
    file_dst.write('{}'.format(lines[-1]))
    file_dst.close()
    print('pos_np.shape {}'.format(pos_np.shape))
    print('results_pos.shape {}'.format(results_pos.shape))
    # time.sleep(30)


def produceAurora(case_id, result_params, target_dir='results'):
    results_pos_path = 'results/pos/Case{:04}_Aurora_result.pos'.format(case_id)
    results_pos = np.loadtxt(results_pos_path)

    ts_start = '30157462'

    # file = open(original_pos_path, 'r')
    file_dst = open('results/results_pos/Case{:04}_Aurora_results.pos'.format(case_id), 'a+')

    # lines = file.readlines()
    # file_dst.write('{}'.format(lines[0]))
    file_dst.write('### Thursday, January 14, 2016 08:22:37 Start logging\n')

    pos_np = []
    for line_index in range(1, len(lines) - 1):  # exclude the first line and last line
        result_index = line_index - 1
        line = lines[line_index]
        values = line.split()
        values_np = np.asarray(values[1:]).astype(np.float32)
        pos_np.append(values_np)

        for fixed_id in range(3):
            file_dst.write('{} '.format(int(values[fixed_id])))

        for params_id in range(results_pos.shape[1]):
            file_dst.write('{:.6f} '.format(results_pos[result_index, params_id]))
        file_dst.write('\n')
    pos_np = np.asarray(pos_np)
    file_dst.write('{}'.format(lines[-1]))
    file_dst.close()
    print('pos_np.shape {}'.format(pos_np.shape))
    print('results_pos.shape {}'.format(results_pos.shape))


def evaluate_correlation(dof1, dof2, abs=False):
    # print(dof1.shape, dof2.shape)
    corrs = []
    for dof_id in range(dof1.shape[1]):
        this_dof1 = dof1[:, dof_id]
        this_dof2 = dof2[:, dof_id]

        cor_coe = np.corrcoef(this_dof1, this_dof2)
        corrs.append(cor_coe[0, 1])
    if abs:
        corr_result = np.mean(np.abs(np.asarray(corrs)))
    else:
        corr_result = np.mean(np.asarray(corrs))
    # time.sleep(30)
    return corr_result


def visualize_attention(case_id, batch_ids, batch_imgs, maps, weights):
    batch_imgs = batch_imgs.data.cpu().numpy()
    maps = maps.data.cpu().numpy()
    print(case_id)
    print(batch_ids)
    print(batch_imgs.shape)
    print(maps.shape)
    print(weights.shape)

    dofs = ['tX', 'tY', 'tZ', 'aX', 'aY', 'aZ']

    for batch_loop in range(len(batch_ids)):
        frame_id = batch_ids[batch_loop]
        frame_map = maps[batch_loop, :, 0, :, :]
        frame_img = batch_imgs[batch_loop, 0, 0, :, :]
        frame_img2 = batch_imgs[batch_loop, 0, -1, :, :]
        diff_img = frame_img2 - frame_img
        # print('frame_id {}, frame_map {}'.format(frame_id, frame_map.shape))

        # dof_atmaps = []
        for dof_id in range(weights.shape[0]):
            dof_weight = weights[dof_id, :]
            dof_weight = np.expand_dims(dof_weight, 1)
            dof_weight = np.expand_dims(dof_weight, 1)

            dof_map = dof_weight * frame_map
            dof_map = np.sum(dof_map, axis=0)
            dof_map = cv2.resize(dof_map, (frame_img.shape[0], frame_img.shape[1]),
                                 interpolation=cv2.INTER_LINEAR)
            # print(dof_weight.shape)
            # print(dof_map.shape)
            plt.imsave('maps/{}_{}_{}.jpg'.format(case_id, frame_id, dofs[dof_id]),
                       dof_map, cmap='jet_r')
            plt.imsave('maps/{}_{}_ad.jpg'.format(case_id, frame_id),
                       diff_img, cmap='jet')
            cv2.imwrite('maps/{}_{}.jpg'.format(case_id, frame_id), frame_img)
            print('Saved {}_{}_{}.jpg'.format(case_id, frame_id, dofs[dof_id]))
            # time.sleep(30)
            # plt.figure()
            # plt.imshow(dof_map, )


            # dof_atmaps.append(dof_map)
    print('batch saved')
    # time.sleep(30)


def compress_frames():
    case_folder = '/zion/guoh9/projects/FreehandUSRecon/data/US_vid_frames/test/Case0005'
    img_list = os.listdir(case_folder)
    img_list.sort()

    crop_img_list = []

    for img_name in img_list:
        print(img_name)
        img_path = path.join(case_folder, img_name)
        orig_img = cv2.imread(img_path, 0)

        crop_img = train_network.data_transform(orig_img, crop_size=224, resize=224, normalize=False, masked_full=False)
        crop_img_list.append(crop_img)
        print(crop_img.shape)

        cv2.imshow('full', orig_img)
        cv2.imshow('crop', crop_img)
        cv2.waitKey(0)

    img_3d = np.stack(crop_img_list, axis=2)
    np.save('data/US_vid_frames/Case0005.npy', img_3d)
    print('img_3d shape {}'.format(img_3d.shape))
    time.sleep(30)


def validate_crop_range(upper_left_pt, crop_size, img_shape):
    x_start, y_start = upper_left_pt
    bbox_h, bbox_w = crop_size
    img_h, img_w = img_shape
    if bbox_h <= 0 or bbox_w <= 0:
        print('ROI cropped size error!')
        return False
    elif x_start < 0 or x_start > img_h or y_start < 0 or y_start > img_w:
        print('ROI upper_left_pt range error!')
        return False
    elif x_start + bbox_h > img_h or y_start + bbox_w > img_w:
        print('ROI cropped size too large!')
        return False
    else:
        return True



def data_transform(input_img, upper_left_pt=(128, 208), crop_size=(224, 224), resize=224, 
                   normalize=False, masked_full=False):
    """
    Crop and resize image as you wish. This function is shared through multiple scripts
    :param input_img: please input a grey-scale numpy array image
    :param crop_size: center crop size, make sure do not contain regions beyond fan-shape
    :param resize: resized size
    :param normalize: whether normalize the image
    :return: transformed image
    """
    if masked_full:
        input_img[fan_mask == 0] = 0
        masked_full_img = input_img[112:412, 59:609]
        return masked_full_img

    range_valid = validate_crop_range(upper_left_pt=upper_left_pt, crop_size=crop_size, 
                                      img_shape=input_img.shape)
    if not range_valid:
        print('Please define a valid ROI range!')
        sys.exit()

    x_start, y_start = upper_left_pt    
    bbox_h, bbox_w = crop_size
    patch_img = input_img[x_start:x_start+bbox_h, y_start:y_start+bbox_w]
    # print('before resize {}'.format(patch_img.shape))
    patch_img = cv2.resize(patch_img, (resize, resize))
    # print('after resize {}'.format(patch_img.shape))
    # sys.exit()
    # cv2.imshow('patch', patch_img)
    # cv2.waitKey(0)
    if normalize:
        patch_img = patch_img.astype(np.float64)
        patch_img = (patch_img - np.min(patch_img)) / (np.max(patch_img) - np.mean(patch_img))

    return patch_img

def sampleGaps(slice_num, neighbour_num, gap=0):
    """ This function samples selected slice ids for a give gap

    Args:
        slice_num (int): Number of slices of this case
        neighbour_num (int): How many consecutive slices to sample
        gap (int, optional): Number of frame gap between two frames. 
            Defaults to 0.

    Returns:
        1D Numpy array: Contain the selected frames ids
    """
    if neighbour_num + (neighbour_num - 1) * gap > slice_num:
        gap = 0

    start_range = slice_num - (neighbour_num + (neighbour_num - 1) * gap)

    if start_range <= 0:
        start_id = 0
    else:
        start_id = np.random.randint(0, start_range, 1)[0]
        
    end_id = start_id + neighbour_num + (neighbour_num - 1) * gap - 1
    select_ids = np.linspace(start_id, end_id, num=neighbour_num, endpoint=True, dtype=np.int32)

    return select_ids

def get6DofLabel(mat1, mat2):
    """
    Give two 4x4 matrix, compute relative 6 DOF
    :param mat1: Transformation matrix of the first frame
    :param mat2: Transformation matrix of the second frame
    :return: Numpy array (6, ), 3 translations and 3 rotations (degrees)
    """
    relative_mat = np.dot(mat2, inv(mat1))
    translations = relative_mat[:3, 3]
    rotations_eulers = np.asarray(tfms.euler_from_matrix(relative_mat))

    rotations_degrees = (rotations_eulers / (2 * math.pi)) * 360

    dof = np.concatenate((translations, rotations_degrees), axis=0)

    return dof

def mats2corner_pts(mats, input_img):
    """
    Transform the Aurora params to corner points coordinates of each frame
    :param params: slice_num x 7(or 9) params matrix
    :param input_img: just use for size
    :return: slice_num x 4 x 3. 4 corner points 3d coordinates (x, y, z)
    """
    h, w = input_img.shape
    corner_pts = np.asarray([[w, 0, 0],
                             [w, h, 0],
                             [0, h, 0],
                             [0, 0, 0]])

    corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
    corner_pts = np.transpose(corner_pts)

    transformed_pts = []
    for trans_mat in mats:
        transformed_corner_pts = np.dot(trans_mat, corner_pts)

        # dist1 = np.linalg.norm(transformed_corner_pts[:3, 0] - transformed_corner_pts[:3, 1])
        # dist2 = np.linalg.norm(transformed_corner_pts[:3, 1] - transformed_corner_pts[:3, 2])
        # scale_ratio = (dist2 / input_img.shape[0] + dist1 / input_img.shape[1]) / 2
        # transformed_corner_pts = transformed_corner_pts / scale_ratio

        # dist3 = np.linalg.norm(transformed_corner_pts[:3, 2] - transformed_corner_pts[:3, 3])
        # dist4 = np.linalg.norm(transformed_corner_pts[:3, 3] - transformed_corner_pts[:3, 0])
        # print(dist1, dist2, dist3, dist4)
        # print('dist1: {}, dist2 {}, ratio {}'.format(dist1, dist2, dist1/dist2))
        # time.sleep(30)

        transformed_corner_pts = np.moveaxis(
            transformed_corner_pts[:3, :], 0, 1)
        transformed_pts.append(transformed_corner_pts)
    transformed_pts = np.asarray(transformed_pts)
    return transformed_pts


def define_model(model_type, pretrained_path='', neighbour_slice=5,
                 input_type='org_img', output_type='average_dof'):
    if input_type == 'diff_img':
        input_channel = neighbour_slice - 1
    else:
        input_channel = neighbour_slice

    if model_type == 'prevost':
        model_ft = generators.PrevostNet()
    elif model_type == 'resnext50':
        model_ft = resnext.resnet50(sample_size=2, sample_duration=16, cardinality=32)
        model_ft.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                                   stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
    elif model_type == 'resnext101':
        model_ft = resnext.resnet101(sample_size=2, sample_duration=16, cardinality=32)
        model_ft.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                                   stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        # model_ft.conv1 = nn.Conv3d(neighbour_slice, 64, kernel_size=7, stride=(1, 2, 2),
        #                            padding=(3, 3, 3), bias=False)
    elif model_type == 'resnet152':
        model_ft = resnet.resnet152(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet101':
        model_ft = resnet.resnet101(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet50':
        model_ft = resnet.resnet50(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet34':
        model_ft = resnet.resnet34(pretrained=False)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'resnet18':
        model_ft = resnet.resnet18(pretrained=True)
        model_ft.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
    elif model_type == 'mynet':
        model_ft = mynet.resnet50(sample_size=2, sample_duration=16, cardinality=32)
        model_ft.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
                                   stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
    elif model_type == 'mynet2':
        model_ft = generators.My3DNet()
    elif model_type == 'p3d':
        model_ft = p3d.P3D63()
        model_ft.conv1_custom = nn.Conv3d(1, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                          padding=(0, 3, 3), bias=False)
    elif model_type == 'densenet121':
        model_ft = densenet.densenet121()
    else:
        print('network type of <{}> is not supported, use original instead'.format(network_type))
        model_ft = generators.PrevostNet()

    num_ftrs = model_ft.fc.in_features

    if model_type == 'mynet':
        num_ftrs = 384
    elif model_type == 'prevost':
        num_ftrs = 576

    if output_type == 'average_dof' or output_type == 'sum_dof':
        # model_ft.fc = nn.Linear(128, 6)
        model_ft.fc = nn.Linear(num_ftrs, 6)
    else:
        # model_ft.fc = nn.Linear(128, (neighbour_slice - 1) * 6)
        model_ft.fc = nn.Linear(num_ftrs, (neighbour_slice - 1) * 6)



    # if args.training_mode == 'finetune':
    #     model_path = path.join(results_dir, args.model_filename)
    #     if path.isfile(model_path):
    #         print('Loading model from <{}>...'.format(model_path))
    #         model_ft.load_state_dict(torch.load(model_path))
    #         print('Done')
    #     else:
    #         print('<{}> not exists! Training from scratch...'.format(model_path))

    if pretrained_path:
        if path.isfile(pretrained_path):
            print('Loading model from <{}>...'.format(pretrained_path))
            model_ft.load_state_dict(torch.load(pretrained_path, map_location='cuda:0'))
            # model_ft.load_state_dict(torch.load(pretrained_path))
            print('Done')
        else:
            print('<{}> not exists! Training from scratch...'.format(pretrained_path))
    else:
        print('Train this model from scratch!')

    device = 0
    model_ft.cuda()
    model_ft = model_ft.to(device)
    model_ft.eval()
    print('define model device {}'.format(device))
    # time.sleep(30)
    return model_ft

def write_list(input_list, txt_path):
    textfile = open(txt_path, "w")

    for element in input_list:

        textfile.write(element + "\n")

    textfile.close()
    
    return 0


if __name__ == '__main__':
    compress_frames()

    # clean_ids()
    #
    # test_ids = np.asarray([8, 12, 15, 43, 54, 55])
    # for id in test_ids:
    #     produce_Aurora(case_id=id)
    #     print('{} finished'.format(id))
    # time.sleep(30)

    # center_crop()
    # test_avg_dof()
    # save_vid_1frame()
    # aurora_path = '/zion/common/data/uronav_data/Case0001/Case0001_Aurora.pos'
    # pos = read_aurora(file_path=aurora_path)
    # print(pos.shape)
    # print(pos)

    # test_us_img = cv2.imread('/zion/guoh9/projects/USFreehandRecon/data/frames/0065.jpg', 0)
    # mask_fan(input_img=test_us_img)
    # save_all_aurora_pos()

    # frame_img = cv2.imread('/home/guoh9/tmp/US_vid_frames/Case0001/0000.jpg', 0)
    # pos = np.loadtxt('/home/guoh9/tmp/US_vid_pos/Case0001.txt')
    # frame_pos = pos[0, :]
    # frame_pos = np.zeros((7,))
    # frame_pos[6] = 1
    # # plot_2d_in_3d(trans_params=frame_pos, frame_color='b', input_img=frame_img)
    #
    # plot_2d_in_3d_test(trans_params1=pos[0, :],
    #                    trans_params2=pos[10, :])

    # visualize_frames(case_id=1)
    # sample_ids(slice_num=78, neighbour_num=10)
    case = VisualizeSequence(case_id=9)
    # case = DofPlot(case_id=370)
    # for i in range(71):
    #     case_plot = DofPlot(case_id=i+1)
    # time.sleep(30)

    case_id = 7
    case_pos_np = np.loadtxt('/home/guoh9/tmp/US_vid_pos/Case{:04}.txt'.format(case_id))
    case_calib_mat = np.loadtxt('/zion/common/data/uronav_data/Case{:04}/'
                                'Case{:04}_USCalib.txt'.format(case_id, case_id))
    pos1 = case_pos_np[1, 2:]
    pos2 = case_pos_np[10, 2:]
    label = get_6dof_label(trans_params1=pos1, trans_params2=pos2,
                           cam_cali_mat=case_calib_mat)
    recon_params = get_next_pos(trans_params1=pos1, dof=label,
                                cam_cali_mat=case_calib_mat)
    print('labels\n{}'.format(label))
    print('params2\n{}'.format(pos2))
    print('recon_params\n{}'.format(recon_params))

def params2mat_new(trans_params, cam_cali_mat=np.identity(4), calibration=False):
    """
    Transform the parameters in Aurora files into 4 x 4 matrix
    :param trans_params: transformation parameters in Aurora.pos. Only the last 7 are useful
    3 are translations, 4 are the quaternion (x, y, z, w) for rotation
    :return: 4 x 4 transformation matrix
    """
    if trans_params.shape[0] == 9:
        trans_params = trans_params[2:]

    translation = trans_params[:3]
    quaternion = trans_params[3:]

    """ Transform quaternion to 3 x 3 rotation matrix, get rid of unstable scipy codes"""
    # r_mat = R.from_quat(quaternion).as_matrix()
    # print('r_mat\n{}'.format(r_mat))

    new_quat = np.zeros((4,))
    new_quat[0] = quaternion[-1]
    new_quat[1:] = quaternion[:3]
    r_mat = tfms.quaternion_matrix(quaternion=new_quat)[:3, :3]
    # print('my_mat\n{}'.format(r_mat))

    trans_mat = np.zeros((4, 4))
    trans_mat[:3, :3] = r_mat
    trans_mat[:3, 3] = translation
    trans_mat[3, 3] = 1

    r_mat44 = tfms.quaternion_matrix(quaternion=new_quat)
    t_mat44 = np.identity(4)
    t_mat44[:3, 3] = translation

    tr_dot = np.dot(t_mat44, r_mat44)
    # rt_dot = np.dot(r_mat44, t_mat44)
    # print("r_mat44\n{}".format(r_mat44))
    # print("t_mat44\n{}".format(t_mat44))
    # print("tr_dot\n{}".format(tr_dot))
    # print("rt_dot\n{}".format(rt_dot))
    # time.sleep(30)
    #
    # print('trans_mat\n{}'.format(trans_mat))
    # trans_mat = tr_dot
    if calibration:
        #         trans_mat = np.dot(cam_cali_mat, trans_mat)
        #         trans_mat = np.linalg.inv(trans_mat)
        #         trans_mat = np.dot(cam_cali_mat, tr_dot)
        #         trans_mat = np.linalg.inv(trans_mat)
        # cam_cali_mat = decompose_mat(input_mat=cam_cali_mat)
        trans_mat = np.dot(tr_dot, cam_cali_mat)  #This is correct for 640 data
        # trans_mat = np.dot(tr_dot, np.linalg.inv(cam_cali_mat))  #This is tested for new sweep
        # trans_mat = tr_dot

    return trans_mat, tr_dot, cam_cali_mat

def params2corner_pts_correction(case_id, params, cam_cali_mat, input_img,
                                 cb_pts=None, ft_pts=None, correction=True):
    """
    Transform the Aurora params to corner points coordinates of each frame
    :param params: slice_num x 7(or 9) params matrix
    :param input_img: just use for size
    :return: slice_num x 4 x 3. 4 corner points 3d coordinates (x, y, z)
    """
    h, w = input_img.shape
    # corner_pts = np.asarray([[w / 2, h / 2, 0],
    #                          [w / 2, -h / 2, 0],
    #                          [-w / 2, -h / 2, 0],
    #                          [-w / 2, h / 2, 0]])

    # corner_pts = np.asarray([[0, 0, 0],
    #                          [w, 0, 0],
    #                          [w, h, 0],
    #                          [0, h, 0]])

    corner_pts = np.asarray([[w, 0, 0],
                             [w, h, 0],
                             [0, h, 0],
                             [0, 0, 0]])

    corner_pts = np.concatenate((corner_pts, np.ones((corner_pts.shape[0], 1))), axis=1)
    corner_pts = np.transpose(corner_pts)

    transformed_pts = []
    transform_mats = []
    original_mats = []
    for frame_id in range(params.shape[0]):
        trans_mat, tr_dot, cam_cali_mat = params2mat_new(trans_params=params[frame_id, :],
                                                         cam_cali_mat=cam_cali_mat,
                                                         calibration=True)

        original_mats.append(trans_mat)
        # """ calibrate all mats to origin """
        # if frame_id == 0:
        #     print('trans_mat\n{}'.format(trans_mat))
        #     orig_mat = trans_mat
        #     translation = tfms.translation_from_matrix(trans_mat)
        #     t_back = tfms.translation_matrix(-translation)
        #     trans_mat = np.dot(t_back, trans_mat)
        #     print('t_back trans_mat\n{}'.format(trans_mat))
        #
        #     inv_r = inv(trans_mat)
        #     rotation = tfms.decompose_matrix(trans_mat)[2]
        #     rotation_inv = tfms.decompose_matrix(inv_r)[2]
        #     print('rotation {}'.format(rotation))
        #     print('rotation_inv {}'.format(rotation_inv))
        #     time.sleep(30)
        #     r_back = tfms.euler_matrix(ai=-rotation[0], aj=-rotation[1],
        #                                ak=-rotation[2])
        #     trans_mat = np.dot(r_back, trans_mat)
        #     print('r_back trans_mat\n{}'.format(trans_mat))
        #
        #     rotation = tfms.decompose_matrix(trans_mat)[2]
        #     print('trans_mat\n{}'.format(trans_mat))
        #     print('t_back\n{}'.format(t_back))
        #     print('translation {}'.format(translation))
        #     print('rotation {}'.format(rotation))
        #     time.sleep(30)
        #     trans_mat = np.dot(inv(trans_mat), trans_mat)
        #
        # else:
        #     # trans_mat = np.dot(inv(original_mats[0]), trans_mat)
        #     trans_mat = np.dot(trans_mat, inv(original_mats[0]))
        #     # trans_mat = relative_mat

        """ Test slicer coordinate system """
        # scales = mat_scales(input_mat=original_mats[0])
        # if frame_id == 0:
        #     trans_mat = np.identity(4)
        #     # trans_mat = scale_mat(trans_mat, scales)
        # else:
        #     trans_mat = np.dot(inv(original_mats[0]), trans_mat)
        #     # trans_mat = scale_mat(trans_mat, scales)

        """ Adjust trajectory to origin plane! """
        # trans_mat = adjust_mat(case_id, mat0=original_mats[0], mat_this=trans_mat)


        """ Fully functional rotation modification"""
        # r_mat = tfms.euler_matrix(math.pi, 0, 0, axes='sxyz')
        # trans_mat = np.dot(r_mat, trans_mat)  # rotation works here

        # t_mat = np.identity(4)
        # t_mat[1, 3] = 690
        # trans_mat = np.dot(t_mat, trans_mat)  # rotation works here
        # print('r_mat\n{}'.format(r_mat))
        # time.sleep(30)

        transformed_corner_pts = np.dot(trans_mat, corner_pts)
        t_mat = np.identity(4)
        if correction:
            cb_pt = cb_pts[frame_id, :]
            ft_pt = ft_pts[frame_id, :]

            """ We need to move cb_pt to ft_pt """
            trans_vec = ft_pt - cb_pt
            t_mat = np.identity(4)
            t_mat[:3, 3] = trans_vec

            """ Test if translation mat works """
            # cb_pt4 = np.insert(cb_pt, 3, 1)
            # cb_pt4 = np.expand_dims(cb_pt4, 1)
            # trans_pt = np.dot(t_mat, cb_pt4)

        """ Apply correction to points """
        transformed_corner_pts = np.dot(t_mat, transformed_corner_pts)
        transformed_corner_pts = np.moveaxis(transformed_corner_pts[:3, :], 0, 1)
        transformed_pts.append(transformed_corner_pts)
        transform_mats.append(np.dot(t_mat, trans_mat))  # This is correct for 640 data
    # print('{}'.format(transform_mats[0]))
    # print('{}'.format(transform_mats[-1]))
    # time.sleep(30)
    transformed_pts = np.asarray(transformed_pts)
    transform_mats = np.asarray(transform_mats)
    # print('transform_mats shape {}'.format(transform_mats.shape))
    return transformed_pts, transform_mats

def mats2cornerpts(mats, input_img, cb_pts=None, ft_pts=None, correction=False):
    h, w = input_img.shape
    corner_pts = np.asarray([[w, 0, 0],
                             [w, h, 0],
                             [0, h, 0],
                             [0, 0, 0]])

    corner_pts = np.concatenate((corner_pts, np.ones((corner_pts.shape[0], 1))), axis=1)
    corner_pts = np.transpose(corner_pts)

    transformed_pts = []
    for frame_id in range(mats.shape[0]):
        trans_mat = mats[frame_id,:,:]

        transformed_corner_pts = np.dot(trans_mat, corner_pts)
        t_mat = np.identity(4)
        if correction:
            cb_pt = cb_pts[frame_id, :]
            ft_pt = ft_pts[frame_id, :]

            """ We need to move cb_pt to ft_pt """
            trans_vec = ft_pt - cb_pt
            t_mat = np.identity(4)
            t_mat[:3, 3] = trans_vec

        """ Apply correction to points """
        transformed_corner_pts = np.dot(t_mat, transformed_corner_pts)
        transformed_corner_pts = np.moveaxis(transformed_corner_pts[:3, :], 0, 1)
        transformed_pts.append(transformed_corner_pts)

    transformed_pts = np.asarray(transformed_pts)
    return transformed_pts

def mat2tfm(input_mat):
    tfm = sitk.AffineTransform(3)
    tfm.SetMatrix(np.reshape(input_mat[:3, :3], (9,)))
    translation = input_mat[:3,3]
    tfm.SetTranslation(translation)
    # tfm.SetCenter([0, 0, 0])
    return tfm

def resample_slice(input_vol, frame_mat, destVol=None, crop_frame=False):
    if not destVol:
        half_size = 500
        xyhw=[105, 54, 320, 565]
        fan_center = (int(xyhw[0]+xyhw[2]//2), int(xyhw[1]+xyhw[3]//2))
        destSpacing = np.asarray([1., 1., 1.])
        destOrigin = np.asarray([fan_center[1]-half_size*destSpacing[0], 
                                    fan_center[0]-half_size*destSpacing[0], 0]).astype(np.float)
        destVol = sitk.Image(int(half_size)*2, int(half_size)*2, 1, sitk.sitkUInt8)
        destVol.SetSpacing(np.asarray([1., 1., 1.]))
        destVol.SetOrigin(destOrigin)
    
    if crop_frame:
        destSpacing = np.asarray([1., 1., 1.])
        destOrigin = np.asarray([0, 0, 0]).astype(np.float)
        destVol = sitk.Image(640, 480, 1, sitk.sitkUInt8)
        destVol.SetSpacing(np.asarray([1., 1., 1.]))
        destVol.SetOrigin(destOrigin)

    """ Pack the 4x4 frame_mat np array as sitk transform """
    tfm = sitk.CompositeTransform(mat2tfm(np.identity(4)))
    tfm_slice = mat2tfm(frame_mat.astype(np.float))
    tfm.AddTransform(tfm_slice)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(destVol)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tfm)
    resampled_slice = sitk.GetArrayFromImage(resampler.Execute(input_vol))[0, :, :]
    return resampled_slice



