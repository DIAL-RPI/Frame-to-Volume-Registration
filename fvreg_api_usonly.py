from unicodedata import decimal
import numpy as np
import time
import cv2
import copy
import os
import os.path as path
import imageio
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import argparse
from numpy.linalg import inv
import torch
from tools import data_transform
import tools
import sys
import SimpleITK as sitk
import network_func
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from utils import data_loading_funcs as load_func
import draw_3d_plot
import math

desc = 'Test reconstruction network'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-d', '--device_no',
                    type=int,
                    choices=[0, 1, 2, 3, 4, 5, 6, 7],
                    help='GPU device number [0-7]',
                    default=0)
parser.add_argument('-avg', '--average_dof',
                    type=bool,
                    help='give the average bof within a sample',
                    default=False)

args = parser.parse_args()
device_no = args.device_no

train_ids = np.loadtxt('infos/train_ids.txt').astype(np.int64)
val_ids = np.loadtxt('infos/val_ids.txt').astype(np.int64)
test_ids = np.loadtxt('infos/test_ids.txt').astype(np.int64)

all_ids = np.concatenate((train_ids, val_ids, test_ids), axis=0)

mask_img = cv2.imread('data/US_mask.png', 0)
# print(mask_img.shape)
# sys.exit()

# frames_folder = '/home/guoh9/tmp/US_vid_frames'
# pos_folder = '/home/guoh9/tmp/US_vid_pos'

# frames_folder = '/zion/guoh9/US_recon/US_vid_frames'
# pos_folder = '/zion/guoh9/US_recon/US_vid_pos'

# frames_folder = 'data/US_vid_frames'
# pos_folder = 'data/US_vid_pos'
# cali_folder = 'data/US_cali_mats'
data_folder = 'data'

uronav_folder = '/zion/common/data/uronav_data'
frames_folder = '/zion/guoh9/US_recon/US_vid_frames'
pos_folder = '/zion/guoh9/US_recon/US_vid_pos'
US_dataset_path = '/zion/guoh9/US_recon/US_dataset'



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

    """ Transform quaternion to rotation matrix"""
    r_mat = R.from_quat(quaternion).as_matrix()

    trans_mat = np.zeros((4, 4))
    trans_mat[:3, :3] = r_mat
    trans_mat[:3, 3] = translation
    trans_mat[3, 3] = 1

    trans_mat = np.dot(cam_cali_mat, trans_mat)
    trans_mat = inv(trans_mat)

    # new_qua = np.zeros((4, ))
    # new_qua[0] = quaternion[3]
    # new_qua[1:] = quaternion[:3]
    # eulers_from_mat = tfms.euler_from_matrix(r_mat)
    # eulers_from_qua = tfms.euler_from_quaternion(new_qua, axes='sxyz')
    # print('eulers mat\n{}'.format(eulers_from_mat))
    # print('eulers qua\n{}'.format(eulers_from_qua))
    #
    # recon_R = tfms.euler_matrix(eulers_from_mat[0],
    #                             eulers_from_mat[1],
    #                             eulers_from_mat[2])
    # print('R\n{}'.format(r_mat))
    # print('recon_R\n{}'.format(recon_R))
    return trans_mat


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
    """
    trans_mat1 = params_to_mat44(trans_params1, cam_cali_mat=cam_cali_mat)
    trans_mat2 = params_to_mat44(trans_params2, cam_cali_mat=cam_cali_mat)

    relative_mat = np.dot(trans_mat2, inv(trans_mat1))

    translations = relative_mat[:3, 3]
    rotations = R.from_matrix(relative_mat[:3, :3])
    rotations_eulers = rotations.as_euler('xyz')

    dof = np.concatenate((translations, rotations_eulers), axis=0)
    return dof


def get_next_pos(trans_params1, dof, cam_cali_mat):
    """
    Given the first frame's Aurora position line and relative 6dof, return second frame's position line
    :param trans_params1: Aurora position line of the first frame
    :param dof: 6 degrees of freedom based on the first frame
    :param cam_cali_mat: Camera calibration matrix of this case
    :return: Aurora position line of the second frame
    """
    trans_mat1 = params_to_mat44(trans_params1, cam_cali_mat=cam_cali_mat)

    relative_mat = np.identity(4)
    r_recon = R.from_euler('xyz', dof[3:])
    relative_mat[:3, :3] = r_recon.as_matrix()
    relative_mat[:3, 3] = dof[:3]

    next_mat = np.dot(inv(cam_cali_mat), inv(np.dot(relative_mat, trans_mat1)))

    next_params = np.zeros(7)
    next_params[:3] = next_mat[:3, 3]
    next_params[3:] = R.from_matrix(next_mat[:3, :3]).as_quat()
    return next_params


def center_crop(input_img, crop_size=480):
    h, w = input_img.shape
    if crop_size > 480:
        crop_size = 480
    x_start = int((h - crop_size) / 2)
    y_start = int((w - crop_size) / 2)

    patch_img = input_img[x_start:x_start+crop_size, y_start:y_start+crop_size]

    return patch_img


class TestNetwork():
    def __init__(self, case_id):
        super(TestNetwork, self).__init__()
        print('*' * 25 + '{}'.format(case_id) + '*' * 25)
        self.case_id = case_id
        self.case_folder = path.join(US_dataset_path, self.case_id)
        self.case_frames_path = path.join(self.case_folder, 'frames')
        self.frames_list = os.listdir(self.case_frames_path)
        self.frames_list.sort()

        self.cam_cali_mat = np.loadtxt(path.join(self.case_folder, 'USCalib.txt'))
        self.case_pos = np.loadtxt(path.join(self.case_folder, 'Position.txt'))[:, -9:]

        self.case_uronav = path.join(uronav_folder, self.case_id)

        self.case_mhd = path.join(self.case_uronav, 'USVol.mhd')
        self.case_mhd = path.join(self.case_uronav, 'USVol.mhd')
        if path.exists(self.case_mhd):
            self.us_img = sitk.ReadImage(self.case_mhd)
            self.raw_spacing = self.us_img.GetSpacing()[0]
        else:
            self.raw_spacing = 0.5
        self.slice_ids = np.linspace(0, self.case_pos.shape[0]-1, self.case_pos.shape[0]).astype(np.uint64)

        # self.alignment_dir = '/zion/guoh9/US_recon/alignment'
        # self.mr_path = path.join(self.alignment_dir, self.case_id, 'MRVol_adjusted.mhd')
        # self.mr_sitk = sitk.ReadImage(self.mr_path)

        self.recon_dir = '/zion/guoh9/US_recon/recon'
        self.us_path = path.join(self.recon_dir, self.case_id, '{}_origin_gt.mhd'.format(case_id))
        self.us_sitk = sitk.ReadImage(self.us_path)


        self.frames_num = self.case_pos.shape[0]
        colors_R = np.linspace(0, 1, self.frames_num).reshape((self.frames_num, 1))
        colors_G = np.zeros((self.frames_num, 1))
        colors_B = np.linspace(1, 0, self.frames_num).reshape((self.frames_num, 1))

        self.colors = np.concatenate((colors_R, colors_G, colors_B), axis=1)

        def divide_batch(slice_num, batch_size=32):
            """
            Divide all the slices into batches for torch parallel computing
            :param slice_num: number of slices in a video
            :param batch_size: default 32
            :return: a list of array, each array is a batch that contains the index of frames
            """
            batches_num = slice_num // batch_size
            last_batch_size = slice_num % batch_size
            print('slice_num {}, batch_size {}'.format(slice_num, batch_size))
            print('batches_num {}, last_batch_size {}'.format(batches_num, last_batch_size))
            batch_ids = []
            for i in range(batches_num):
                # this_batch_id = np.arange(i * batch_size, (i + 1) * batch_size)
                this_batch_id = self.slice_ids[i * batch_size: (i + 1) * batch_size]
                # this_batch_id = np.flip(this_batch_id)
                batch_ids.append(this_batch_id)
            if last_batch_size != 0:
                # last_batch_id = np.arange(batches_num * batch_size, batches_num * batch_size + last_batch_size)
                last_batch_id = self.slice_ids[batches_num * batch_size:slice_num]
                # last_batch_id = np.flip(last_batch_id)
                batch_ids.append(last_batch_id)
            # print(batch_ids)
            # time.sleep(30)
            return batch_ids

        def get_batch_dofs():
            """
            Give the batches as input
            :return: (frames_num - neighbour_slice + 1) x (neighbour_slice - 1) x 6
            contains the relative motion between two slices within a sample group.
            For example, if a neighbouring sample contains 10 slices, then there are 9 relative
            motions within this group
            """
            end_frame_index = self.frames_num - neighbour_slice + 1
            print('end_frame_index/frame_num {}/{}'.format(end_frame_index, self.frames_num))
            batch_groups = divide_batch(slice_num=end_frame_index, batch_size=batch_size)
            # time.sleep(30)
            # all_vectors = np.zeros((1, 2048))
            all_vectors = np.zeros((1, model_ft.fc.in_features))

            if output_type == 'sum_dof':
                result_dof = np.zeros((1, 6))
            else:
                result_dof = np.zeros((1, neighbour_slice - 1, 6))
            # print('batch_groups\n{}'.format(batch_groups))
            # sys.exit()
            for batch_index, this_batch in enumerate(batch_groups):
                this_batch = batch_groups[batch_index]
                batch_imgs = []
                # print(this_batch)
                # sys.exit()
                for group_index, start_frame in enumerate(this_batch):
                    group_id = this_batch[group_index]
                    sample_slices = []
                    # print(group_id)
                    frame_index = batch_index * neighbour_slice + group_index
                    # print('frame_index {}, batch {}'.format(frame_index, batch_index))
                    # print(int(start_frame))
                    for i in range(neighbour_slice):
                        # print('frame {}, i {}, sum {}'.format(frame_index, i, frame_index+i))
                        # frame_id = int(self.slice_ids[start_frame + i])
                        frame_id = int(start_frame + i)
                        # print('frame_id {}'.format(frame_id))
                        frame_path = path.join(self.case_frames_path, '{:04}.jpg'.format(frame_id))
                        # frame_path = path.join(self.case_frames_path, '{:04}.jpg'.format(20))
                        frame_img = cv2.imread(frame_path, 0)
                        frame_img = data_transform(frame_img, masked_full=False)
                        # frame_img = data_transform(frame_img)
                        sample_slices.append(frame_img)

                    if input_type == 'diff_img':
                        diff_imgs = []
                        for sample_id in range(1, len(sample_slices)):
                            diff_imgs.append(sample_slices[sample_id] - sample_slices[sample_id - 1])
                        sample_slices = np.asarray(diff_imgs)
                    else:
                        sample_slices = np.asarray(sample_slices)

                    batch_imgs.append(sample_slices)
                batch_imgs = np.asarray(batch_imgs)
                if network_type in train_dof_img.networks3D:
                    batch_imgs = np.expand_dims(batch_imgs, axis=1)
                batch_imgs = torch.from_numpy(batch_imgs).float().to(device)

                # outputs, maps = model_ft(batch_imgs)
                results = model_ft(batch_imgs)
                # outputs = results['pred']
                # vectors = results['vec']

                outputs, vectors = model_ft(batch_imgs)
                # print('vectors shape {}'.format(vectors.shape))
                # print('mean {}'.format(np.mean(batch_imgs.data.cpu().numpy())))
                # print('std {}'.format(np.std(batch_imgs.data.cpu().numpy())))
                # print('batch_imgs {}'.format(batch_imgs.shape))

                # print('mean {}'.format(np.mean(batch_imgs[0, :, :, :, :].data.cpu().numpy())))
                # print('std {}'.format(np.std(batch_imgs[0, :, :, :, :].data.cpu().numpy())))

                # print('outputs\n{}'.format(np.around(outputs.data.cpu().numpy(), decimals=2)))
                # sys.exit()

                """ Visualize attention heatmaps """
                # tools.visualize_attention(case_id=self.case_id,
                #                           batch_ids=this_batch,
                #                           batch_imgs=batch_imgs,
                #                           maps=maps, weights=fc_weights)

                # print('this_batch {}'.format(this_batch))
                # print('maps shape {}'.format(maps.shape))
                # print('fc_weights shape {}'.format(fc_weights.shape))
                # print('input shape {}'.format(batch_imgs.shape))
                # print('outputs shape {}'.format(outputs.shape))
                outputs = outputs.data.cpu().numpy()
                vectors = vectors.data.cpu().numpy()
                all_vectors = np.concatenate((all_vectors, vectors), axis=0)
                # all_vectors.append(vectors)
                if output_type == 'average_dof':
                    # print('outputs {}'.format(outputs.shape))
                    outputs = np.expand_dims(outputs, axis=1)
                    outputs_reshape = np.repeat(outputs, neighbour_slice - 1, axis=1)
                    # print('outputs_reshape {}'.format(outputs_reshape.shape))
                    # sys.exit()
                elif output_type == 'sum_dof':
                    outputs_reshape = outputs
                else:
                    outputs_reshape = np.reshape(outputs, (outputs.shape[0],
                                                           int(outputs.shape[1] / 6),
                                                           int(outputs.shape[1] / (neighbour_slice - 1))))
                result_dof = np.concatenate((result_dof, outputs_reshape), axis=0)

            if output_type == 'sum_dof':
                result_dof = result_dof[1:, :]
            else:
                result_dof = result_dof[1:, :, :]
            self.all_vectors = all_vectors[1:, :]
            # print('self.all_vectors {}'.format(self.all_vectors.shape))
            # sys.exit()
            return result_dof

        def get_format_dofs(batch_dofs, merge_option='average_dof'):
            """
            Based on the network outputs, here reformat the result into one row for each frame
            (Because there are many overlapping frames due to the input format)
            :return:
            1) gen_dofs is (slice_num - 1) x 6dof. It is the relative 6dof motion comparing to
            the former frame
            2) pos_params is slice_num x 7params. It is the absolute position, exactly the same
            format as Aurora.pos file
            """
            print('Use <{}> formatting dofs'.format(merge_option))
            if merge_option == 'one':
                gen_dofs = np.zeros((self.frames_num - 1, 6))
                gen_dofs[:batch_dofs.shape[0], :] = batch_dofs[:, 0, :]
                gen_dofs[batch_dofs.shape[0], :] = batch_dofs[-1, 1, :]
                print('gen_dof shape {}'.format(gen_dofs.shape))
                print('not average method')

            elif merge_option == 'baton':
                print('baton batch_dofs shape {}'.format(batch_dofs.shape))
                print('slice_num {}'.format(self.frames_num))
                print('neighboring {}'.format(neighbour_slice))

                gen_dofs = []
                slice_params = []
                for slice_idx in range(self.frames_num):
                    if slice_idx == 0:
                        this_params = self.case_pos[slice_idx, :]
                        slice_params.append(this_params)
                    elif slice_idx < neighbour_slice:
                        this_dof = batch_dofs[0, :] / 4
                        this_params = tools.get_next_pos(trans_params1=slice_params[slice_idx-1],
                                                         dof=this_dof,
                                                         cam_cali_mat=self.cam_cali_mat)
                        gen_dofs.append(this_dof)
                        slice_params.append(this_params)
                    else:
                        baton_idx = slice_idx - neighbour_slice + 1
                        baton_params = slice_params[baton_idx]
                        sample_dof = batch_dofs[baton_idx, :]
                        this_params = tools.get_next_pos(trans_params1=baton_params,
                                                         dof=sample_dof,
                                                         cam_cali_mat=self.cam_cali_mat)
                        this_dof = tools.get_6dof_label(trans_params1=slice_params[slice_idx-1],
                                                        trans_params2=this_params,
                                                        cam_cali_mat=self.cam_cali_mat)
                        gen_dofs.append(this_dof)
                        slice_params.append(this_params)
                gen_dofs = np.asarray(gen_dofs)
                slice_params = np.asarray(slice_params)
                print('gen_dof shape {}'.format(gen_dofs.shape))
                print('slice_params shape {}'.format(slice_params.shape))
                # time.sleep(30)
            else:
                frames_pos = []
                for start_sample_id in range(batch_dofs.shape[0]):
                    for relative_id in range(batch_dofs.shape[1]):
                        this_pos_id = start_sample_id + relative_id + 1
                        # print('this_pos_id {}'.format(this_pos_id))
                        this_pos = batch_dofs[start_sample_id, relative_id, :]
                        this_pos = np.expand_dims(this_pos, axis=0)
                        if len(frames_pos) < this_pos_id:
                            frames_pos.append(this_pos)
                        else:
                            frames_pos[this_pos_id - 1] = np.concatenate((frames_pos[this_pos_id - 1],
                                                                          this_pos), axis=0)

                gen_dofs = []
                for i in range(len(frames_pos)):
                    gen_dof = np.mean(frames_pos[i], axis=0)

                    """This is for Linear Motion"""
                    # gen_dof = train_network.dof_stats[:, 0]
                    # gen_dof = np.asarray([-0.07733258, -1.28508398, 0.37141262,
                    #                       -0.57584312, 0.20969176, 0.51404395]) + 0.1

                    gen_dofs.append(gen_dof)
                gen_dofs = np.asarray(gen_dofs)

                print('batch_dofs {}'.format(batch_dofs.shape))
                print('gen_dofs {}'.format(gen_dofs.shape))
                # time.sleep(30)

            # for dof_id in range(6):
            #     gen_dofs[:, dof_id] = tools.smooth_array(gen_dofs[:, dof_id])
            # time.sleep(30)
            return gen_dofs


        def dof2params(format_dofs):
            gen_param_results = []
            for i in range(format_dofs.shape[0]):
                if i == 0:
                    base_param = self.case_pos[i, :]
                else:
                    base_param = gen_param_results[i-1]
                gen_dof = format_dofs[i, :]
                gen_param = tools.get_next_pos(trans_params1=base_param,
                                               dof=gen_dof, cam_cali_mat=self.cam_cali_mat)
                gen_param_results.append(gen_param)
            # time.sleep(30)
            gen_param_results = np.asarray(gen_param_results)
            pos_params = np.zeros((self.frames_num, 7))
            pos_params[0, :] = self.case_pos[0, 2:]
            pos_params[1:, :] = gen_param_results
            print('pos_params shape {}'.format(pos_params.shape))
            # time.sleep(30)
            
            return pos_params

        def plot_frame3d(trans_params, frame_color=(255, 0, 0),
                         input_img=np.ones((480, 640)), plot_img=False):
            """
            Plot a 2D frame into 3D space for sequence visualization
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
            print('transformed_corner_pts:\n{}'.format(transformed_corner_pts))
            print('transformed_corner_pts shape {}'.format(transformed_corner_pts.shape))
            # dst = np.linalg.norm(transformed_corner_pts[:, 0] - transformed_corner_pts[:, 1])
            # dst2 = np.linalg.norm(transformed_corner_pts[:, 1] - transformed_corner_pts[:, 2])
            # print(dst, dst2)
            # time.sleep(30)

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

        def params2corner_pts(params, input_img=np.ones((480, 640))):
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
            corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
            corner_pts = np.transpose(corner_pts)

            transformed_pts = []
            for frame_id in range(params.shape[0]):
                trans_mat = params_to_mat44(trans_params=params[frame_id, :],
                                            cam_cali_mat=self.cam_cali_mat)
                transformed_corner_pts = np.dot(trans_mat, corner_pts)
                transformed_corner_pts = np.moveaxis(transformed_corner_pts[:3, :], 0, 1)
                transformed_pts.append(transformed_corner_pts)
            transformed_pts = np.asarray(transformed_pts)
            return transformed_pts

        def draw_img_sequence(corner_pts):
            draw_ids = np.asarray([0, corner_pts.shape[0] - 1])
            # draw_ids = np.asarray([corner_pts.shape[0] - 1, 0])
            for frame_id in draw_ids:
            # for frame_id in range(corner_pts.shape[0]):
                w_weights, h_weights = np.meshgrid(np.linspace(0, 1, 224),
                                                   np.linspace(0, 1, 224))
                # print('corner_pts shape {}'.format(corner_pts.shape))
                # time.sleep(30)
                X = (1 - w_weights - h_weights) * corner_pts[frame_id, 0, 0] + \
                    h_weights * corner_pts[frame_id, 3, 0] + w_weights * corner_pts[frame_id, 1, 0]
                Y = (1 - w_weights - h_weights) * corner_pts[frame_id, 0, 1] + \
                    h_weights * corner_pts[frame_id, 3, 1] + w_weights * corner_pts[frame_id, 1, 1]
                Z = (1 - w_weights - h_weights) * corner_pts[frame_id, 0, 2] + \
                    h_weights * corner_pts[frame_id, 3, 2] + w_weights * corner_pts[frame_id, 1, 2]

                img_path = path.join(self.case_frames_path, self.frames_list[frame_id])
                input_img = cv2.imread(img_path, 0)
                input_img = train_dof_img.data_transform(input_img)
                # print('frame_path\n{}'.format(self.frames_list[frame_id]))
                # print('input_img shape {}'.format(input_img.shape))
                # print('max {}, min {}'.format(np.max(input_img), np.min(input_img)))
                # print(X.shape, Y.shape, Z.shape)
                # cv2.imshow('img', input_img)
                # cv2.waitKey(0)
                # time.sleep(30)
                input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
                input_img = input_img / 255

                if frame_id == 0 or frame_id == corner_pts.shape[0] - 1:
                    stride = 10
                else:
                    stride = 10
                # self.ax.plot_surface(X, Y, Z, rstride=20, cstride=20, facecolors=input_img)
                self.ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride,
                                     facecolors=input_img, zorder=0.1)

        def draw_one_sequence(corner_pts, name, colorRGB=(255, 0, 0), line_width=3, constant=True):
            colorRGB = tuple(channel/255 for channel in colorRGB)
            seg_num = corner_pts.shape[0] + 1


            if constant:
                constant_color = np.asarray(colorRGB)
                constant_color = np.expand_dims(constant_color, axis=0)
                colors = np.repeat(constant_color, seg_num, axis=0)
            else:
                colors_R = np.linspace(0, colorRGB[0], seg_num).reshape((seg_num, 1))
                colors_G = np.linspace(0, colorRGB[1], seg_num).reshape((seg_num, 1))
                colors_B = np.linspace(1, colorRGB[2], seg_num).reshape((seg_num, 1))

                colors = np.concatenate((colors_R, colors_G, colors_B), axis=1)


            # for frame_id in range(int(corner_pts.shape[0] * 0.5), corner_pts.shape[0]):
            #     if frame_id == int(corner_pts.shape[0] * 0.5):
            for frame_id in range(corner_pts.shape[0]):
                if frame_id == 0:
                    """ First frame draw full bounds"""
                    for pt_id in range(-1, 3):
                        xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id, pt_id + 1, 0]
                        ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id, pt_id + 1, 1]
                        zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id, pt_id + 1, 2]
                        self.ax.plot(xs, ys, zs, color=tuple(colors[frame_id, :]), lw=line_width, zorder=1)
                elif frame_id == corner_pts.shape[0] - 1:
                    """ Connect to the former frame """
                    for pt_id in range(-1, 3):
                        xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id - 1, pt_id, 0]
                        ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id - 1, pt_id, 1]
                        zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id - 1, pt_id, 2]
                        self.ax.plot(xs, ys, zs, color=tuple(colors[frame_id, :]), lw=line_width)
                    """ Last frame draw full bounds"""
                    for pt_id in range(-1, 3):
                        xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id, pt_id + 1, 0]
                        ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id, pt_id + 1, 1]
                        zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id, pt_id + 1, 2]
                        self.ax.plot(xs, ys, zs, color=tuple(colors[-1, :]), lw=line_width)
                        if pt_id == -1:
                            self.ax.plot(xs, ys, zs, color=tuple(colors[-1, :]), lw=line_width, label=name)
                else:
                    """ Connect to the former frame """
                    for pt_id in range(-1, 3):
                        xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id - 1, pt_id, 0]
                        ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id - 1, pt_id, 1]
                        zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id - 1, pt_id, 2]
                        self.ax.plot(xs, ys, zs, color=tuple(colors[frame_id, :]), lw=line_width, zorder=1)

                # if plot_img and frame_id==0:


        def visualize_sequences():
            self.fig = plt.figure()
            self.ax = self.fig.gca(projection='3d')
            draw_img_sequence(corner_pts=self.gt_pts1)
            draw_one_sequence(corner_pts=self.gt_pts1, name='Groundtruth',
                              colorRGB=(0, 153, 76), line_width=3)
            draw_one_sequence(corner_pts=self.trans_pts1, name='DCL-Net ({:.4f}mm)'.format(self.trans_pts1_error),
                              colorRGB=(255, 0, 0))

            # plt.axis('off')
            # self.ax.set_xticklabels([])
            # self.ax.set_yticklabels([])
            # self.ax.set_zticklabels([])
            plt.legend(loc='lower left')
            plt.tight_layout()

            # views_id = np.linspace(0, 360, 36)
            # for ii in views_id:
            #     self.ax.view_init(elev=10., azim=ii)
            #     plt.savefig('views/{}_img.jpg'.format(ii))
            #     # plt.savefig('views/{}.jpg'.format(ii))
            #     print('{} saved'.format(ii))

            self.ax.view_init(elev=10., azim=0)
            plt.title(self.case_id)
            plt.savefig('views/all_cases/{}_{}.jpg'.format(model_string, case_id))
            plt.show()
            plt.close()

        def get_gt_dofs():
            # print('self.case_pos:\n{}'.format(np.around(self.case_pos, decimals=2)))
            # sys.exit()
            gt_dofs = []
            for slice_id in range(1, self.frames_num):
                params1 = self.case_pos[slice_id-1, :]
                params2 = self.case_pos[slice_id, :]
                this_dof = tools.get_6dof_label(trans_params1=params1,
                                                trans_params2=params2,
                                                cam_cali_mat=self.cam_cali_mat)
                # print('params1:\n{}'.format(np.around(params1, decimals=2)))
                # print('params2:\n{}'.format(np.around(params2, decimals=2)))
                # print('cam_cali_mat:\n{}'.format(np.around(self.cam_cali_mat, decimals=2)))
                # print('this_dof:\n{}'.format(np.around(this_dof, decimals=2)))
                # sys.exit()
                gt_dofs.append(this_dof)
            gt_dofs = np.asarray(gt_dofs)
            print('gt_dof shape {}, frames_num {}'.format(gt_dofs.shape, self.frames_num))
            return gt_dofs

        def visualize_dofs():
            frees = ['tX', 'tY', 'tZ', 'aX', 'aY', 'aZ']
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Case{:04}'.format(self.case_id))
            for dof_id in range(len(frees)):
                plot_x = dof_id // 3
                plot_y = dof_id % 3
                axes[plot_x, plot_y].plot(self.gt_dofs[:, dof_id], color='g', label='Groundtruth', alpha=0.5)
                axes[plot_x, plot_y].plot(self.format_dofs[:, dof_id], color='r', label='CNN', alpha=0.5)

                corrcoef = np.corrcoef(self.gt_dofs[:, dof_id], self.format_dofs[:, dof_id])[0, 1]

                axes[plot_x, plot_y].set_title('{}: corrcoef {:.4f}'.format(frees[dof_id], corrcoef))
                axes[plot_x, plot_y].legend(loc='lower left')
                # axes[plot_x, plot_y].show()

                np.savetxt('figures/dof_values/{}_{}_gt.txt'.format(self.case_id, frees[dof_id]),
                           self.gt_dofs[:, dof_id])
                np.savetxt('figures/dof_values/{}_{}_{}_pd.txt'.format(model_string, self.case_id, frees[dof_id]),
                           self.format_dofs[:, dof_id])


            plt.savefig('figures/dof_pred/Case{:04}.jpg'.format(self.case_id))
            # plt.show()

        def mats2params(mats):
            res_params = []
            for trans_mat in mats:
                params = tools.mat2params(trans_mat)
                res_params.append(params)
            res_params = np.asarray(res_params)
            return res_params



        self.all_vectors = []
        self.batch_dofs = get_batch_dofs()
        if output_type == 'sum_dof':
            self.format_dofs = get_format_dofs(self.batch_dofs, merge_option='baton')
        else:
            self.format_dofs = get_format_dofs(self.batch_dofs, merge_option='average')

        if normalize_dof:
            self.format_dofs = self.format_dofs * train_dof_img.dof_std + train_dof_img.dof_avg

        self.gt_dofs = get_gt_dofs()
        np.savetxt('tmp/dof_compare/{}_{}.txt'.format(self.case_id, model_string), self.format_dofs)
        np.savetxt('tmp/dof_compare/{}_{}.txt'.format(self.case_id, 'gt'), self.gt_dofs)
        print('tmp/dof_compare/{}_{}.txt'.format(self.case_id, model_string))
        print('pd_dof {}'.format(self.format_dofs.shape))
        print('gt_dof {}'.format(self.gt_dofs.shape))


        self.gt_means = np.mean(self.gt_dofs, axis=0)
        self.pd_means = np.mean(self.format_dofs, axis=0)

        print('gt_means: {}'.format(np.around(np.mean(self.gt_dofs, axis=0), decimals=2)))
        print('pd_means: {}'.format(np.around(np.mean(self.format_dofs, axis=0), decimals=2)))

        """ All following lines are for old test_network """
        self.result_params = dof2params(self.format_dofs)

        self.gt_pts1, self.gt_mats = tools.params2corner_pts_correction(case_id=self.case_id, params=self.case_pos,
                                                                     cam_cali_mat=self.cam_cali_mat,
                                                                     input_img=np.zeros((480, 640)),
                                                                     correction=False)
        self.trans_pts1, self.pd_mats = tools.params2corner_pts_correction(case_id=self.case_id, params=self.result_params,
                                                                     cam_cali_mat=self.cam_cali_mat,
                                                                     input_img=np.zeros((480, 640)),
                                                                     correction=False)

        self.error_mats = tools.errorMats(self.gt_dofs, self.format_dofs)
        self.frame_error = tools.frameError(self.error_mats, self.raw_spacing, 
                                           img_size=(224, 224))
        self.trans_pts1_error = tools.evaluate_dist(pts1=self.gt_pts1, pts2=self.trans_pts1)
        self.final_drift = tools.final_drift(pts1=self.gt_pts1[-1, :, :], pts2=self.trans_pts1[-1, :, :])
        self.cor_coe = tools.evaluate_correlation(dof1=self.format_dofs, dof2=self.gt_dofs, abs=True)
        self.drift_rate = tools.drfit_rate(gt_pts=self.gt_pts1, pd_pts=self.trans_pts1)
        print('self.gt_pts1 shape {}'.format(self.gt_pts1.shape))
        print('self.trans_pts1 shape {}'.format(self.trans_pts1.shape))
        print('{} error {:.4f}mm'.format(self.case_id, self.trans_pts1_error))
        print('{} final drift {:.4f}mm'.format(self.case_id, self.final_drift))
        print('{} frame error {:.4f}mm'.format(self.case_id, self.frame_error))
        print('{} correlation: {:.4f}'.format(self.case_id, self.cor_coe))
        print('{} drift rate: {:.4f}'.format(self.case_id, self.drift_rate))
        print('result_params {}'.format(self.result_params.shape))
        print('*' * 50)
        # visualize_sequences()

        pred_pos_path = path.join(model_pred_result_dir, '{}_result_pos.txt'.format(case_id))
        np.savetxt(pred_pos_path, self.result_params)
        print('pos saved to <{}>'.format(pred_pos_path))
        

        fig_list = draw_3d_plot.draw_fig(init_mat=self.gt_mats[0,:,:],
                                         gt_mats=self.gt_mats,
                                         pd_mats=self.pd_mats,
                                         title=case_id)

        img_list = []
        for frame_id in range(len(self.frames_list)):
            self.us_frame = tools.format_vid_frame(cv2.imread(path.join(self.case_frames_path, '{:04}.jpg'.format(frame_id)), 0))
            self.us_slice = tools.resample_slice(self.us_sitk, self.gt_mats[frame_id, :, :])
            
            mr_mat_gt = np.dot(inv(self.compose_mat), self.gt_mats[frame_id, :, :])
            mr_mat_pd = np.dot(inv(self.compose_mat), self.pd_mats[frame_id, :, :])
            self.mr_slice_gt = tools.resample_slice(self.mr_sitk, mr_mat_gt)
            self.mr_slice_pd = tools.resample_slice(self.mr_sitk, mr_mat_pd)

            self.fused_gt = load_func.fuse_images(self.mr_slice_gt, self.us_frame, alpha=0.3)
            self.fused_pd = load_func.fuse_images(self.mr_slice_pd, self.us_frame, alpha=0.3)
            
            self.concat = np.concatenate((self.fused_gt, self.fused_pd), axis=1)
            self.concat = cv2.resize(self.concat, (1000, 500))
            fig = cv2.resize(fig_list[frame_id], (500, 500))

            self.concat = np.concatenate((self.concat, fig), axis=1)
            img_list.append(self.concat)
            # plt.imshow(self.concat)
            # plt.show()
            # sys.exit()
        gif_path = 'tmp/gif_dcl2/{}.gif'.format(self.case_id)
        imageio.mimsave(gif_path, img_list, duration=0.1)
        print('gif_path: <{}>'.format(gif_path))
        # sys.exit()

class TestNetwork2():
    def __init__(self, case_id):
        super(TestNetwork2, self).__init__()
        print('*' * 25 + '{}'.format(case_id) + '*' * 25)
        self.case_id = case_id
        self.case_folder = path.join(US_dataset_path, self.case_id)
        self.case_frames_path = path.join(self.case_folder, 'frames')
        self.frames_list = os.listdir(self.case_frames_path)
        self.frames_list.sort()
        
        self.cam_cali_mat = np.loadtxt(path.join(self.case_folder, 'USCalib.txt'))
        self.case_pos = np.loadtxt(path.join(self.case_folder, 'Position.txt'))[:, -9:]

        self.case_uronav = path.join(uronav_folder, self.case_id)

        self.case_mhd = path.join(self.case_uronav, 'USVol.mhd')
        if path.exists(self.case_mhd):
            self.us_img = sitk.ReadImage(self.case_mhd)
            self.raw_spacing = self.us_img.GetSpacing()[0]
        else:
            self.raw_spacing = 0.5
        self.slice_ids = np.linspace(0, self.case_pos.shape[0]-1, self.case_pos.shape[0]).astype(np.uint64)

        # self.alignment_dir = '/zion/guoh9/US_recon/alignment'
        # self.mr_path = path.join(self.alignment_dir, self.case_id, 'MRVol_adjusted.mhd')
        # self.mr_sitk = sitk.ReadImage(self.mr_path)
        # self.mr_seg = None
        # mr_seg_path = path.join(self.alignment_dir, case_id, 'mr_seg.mhd')
        # if os.path.isfile(mr_seg_path):
        #     self.mr_seg = sitk.ReadImage(mr_seg_path)
        # self.compose_mat = np.loadtxt(path.join(self.alignment_dir, self.case_id, 'compose_mat.txt'))

        self.recon_dir = '/zion/guoh9/US_recon/recon'
        self.us_path = path.join(self.recon_dir, self.case_id, '{}_origin_gt.mhd'.format(case_id))
        self.us_sitk = sitk.ReadImage(self.us_path)
        self.raw_spacing = self.us_sitk.GetSpacing()


        self.frames_num = self.case_pos.shape[0]
        colors_R = np.linspace(0, 1, self.frames_num).reshape((self.frames_num, 1))
        colors_G = np.zeros((self.frames_num, 1))
        colors_B = np.linspace(1, 0, self.frames_num).reshape((self.frames_num, 1))

        self.colors = np.concatenate((colors_R, colors_G, colors_B), axis=1)

        self.gt_pts1, self.gt_mats = tools.params2corner_pts_correction(case_id=self.case_id, params=self.case_pos,
                                                                     cam_cali_mat=self.cam_cali_mat,
                                                                     input_img=np.zeros((480, 640)),
                                                                     correction=False)

        # landmark_path = '/zion/guoh9/US_recon/labeled_cases_082422/{}/landmark_mr.xml'.format(case_id)
        # self.landmarks = tools.read_landmarks(landmark_path, np.asarray(self.mr_sitk.GetOrigin()))
        # print('landmarks shape {}'.format(self.landmarks.shape))
        # sys.exit()

        def divide_batch(slice_num, batch_size=32):
            """
            Divide all the slices into batches for torch parallel computing
            :param slice_num: number of slices in a video
            :param batch_size: default 32
            :return: a list of array, each array is a batch that contains the index of frames
            """
            batches_num = slice_num // batch_size
            last_batch_size = slice_num % batch_size
            print('slice_num {}, batch_size {}'.format(slice_num, batch_size))
            print('batches_num {}, last_batch_size {}'.format(batches_num, last_batch_size))
            batch_ids = []
            for i in range(batches_num):
                # this_batch_id = np.arange(i * batch_size, (i + 1) * batch_size)
                this_batch_id = self.slice_ids[i * batch_size: (i + 1) * batch_size]
                # this_batch_id = np.flip(this_batch_id)
                batch_ids.append(this_batch_id)
            if last_batch_size != 0:
                # last_batch_id = np.arange(batches_num * batch_size, batches_num * batch_size + last_batch_size)
                last_batch_id = self.slice_ids[batches_num * batch_size:slice_num]
                # last_batch_id = np.flip(last_batch_id)
                batch_ids.append(last_batch_id)
            # print(batch_ids)
            # time.sleep(30)
            return batch_ids

        def get_batch_dofs():
            """
            Give the batches as input
            :return: (frames_num - neighbour_slice + 1) x (neighbour_slice - 1) x 6
            contains the relative motion between two slices within a sample group.
            For example, if a neighbouring sample contains 10 slices, then there are 9 relative
            motions within this group
            """
            end_frame_index = self.frames_num - neighbour_slice + 1
            print('end_frame_index/frame_num {}/{}'.format(end_frame_index, self.frames_num))
            batch_groups = divide_batch(slice_num=end_frame_index, batch_size=batch_size)
            # time.sleep(30)
            # all_vectors = np.zeros((1, 2048))
            all_vectors = np.zeros((1, model_ft.fc.in_features))

            if output_type == 'sum_dof':
                result_dof = np.zeros((1, 6))
            else:
                result_dof = np.zeros((1, neighbour_slice - 1, 6))
            # print('batch_groups\n{}'.format(batch_groups))
            # sys.exit()
            for batch_index, this_batch in enumerate(batch_groups):
                this_batch = batch_groups[batch_index]
                batch_imgs = []
                # print(this_batch)
                # sys.exit()
                for group_index, start_frame in enumerate(this_batch):
                    group_id = this_batch[group_index]
                    sample_slices = []
                    # print(group_id)
                    frame_index = batch_index * neighbour_slice + group_index
                    # print('frame_index {}, batch {}'.format(frame_index, batch_index))
                    # print(int(start_frame))
                    for i in range(neighbour_slice):
                        # print('frame {}, i {}, sum {}'.format(frame_index, i, frame_index+i))
                        # frame_id = int(self.slice_ids[start_frame + i])
                        frame_id = int(start_frame + i)
                        # print('frame_id {}'.format(frame_id))
                        frame_path = path.join(self.case_frames_path, '{:04}.jpg'.format(frame_id))
                        # frame_path = path.join(self.case_frames_path, '{:04}.jpg'.format(20))
                        frame_img = cv2.imread(frame_path, 0)
                        frame_img = data_transform(frame_img, masked_full=False)
                        # frame_img = data_transform(frame_img)
                        sample_slices.append(frame_img)

                    if input_type == 'diff_img':
                        diff_imgs = []
                        for sample_id in range(1, len(sample_slices)):
                            diff_imgs.append(sample_slices[sample_id] - sample_slices[sample_id - 1])
                        sample_slices = np.asarray(diff_imgs)
                    else:
                        sample_slices = np.asarray(sample_slices)

                    batch_imgs.append(sample_slices)
                batch_imgs = np.asarray(batch_imgs)
                if network_type in train_dof_img.networks3D:
                    batch_imgs = np.expand_dims(batch_imgs, axis=1)
                batch_imgs = torch.from_numpy(batch_imgs).float().to(device)

                outputs, vectors = model_ft(batch_imgs)
                
                outputs = outputs.data.cpu().numpy()
                vectors = vectors.data.cpu().numpy()
                all_vectors = np.concatenate((all_vectors, vectors), axis=0)
                # all_vectors.append(vectors)
                if output_type == 'average_dof':
                    outputs = np.expand_dims(outputs, axis=1)
                    outputs_reshape = np.repeat(outputs, neighbour_slice - 1, axis=1)
                elif output_type == 'sum_dof':
                    outputs_reshape = outputs
                else:
                    outputs_reshape = np.reshape(outputs, (outputs.shape[0],
                                                           int(outputs.shape[1] / 6),
                                                           int(outputs.shape[1] / (neighbour_slice - 1))))
                result_dof = np.concatenate((result_dof, outputs_reshape), axis=0)

            if output_type == 'sum_dof':
                result_dof = result_dof[1:, :]
            else:
                result_dof = result_dof[1:, :, :]
            self.all_vectors = all_vectors[1:, :]
            # print('self.all_vectors {}'.format(self.all_vectors.shape))
            # sys.exit()
            return result_dof

        def get_format_dofs(batch_dofs, merge_option='average_dof'):
            """
            Based on the network outputs, here reformat the result into one row for each frame
            (Because there are many overlapping frames due to the input format)
            :return:
            1) gen_dofs is (slice_num - 1) x 6dof. It is the relative 6dof motion comparing to
            the former frame
            2) pos_params is slice_num x 7params. It is the absolute position, exactly the same
            format as Aurora.pos file
            """
            print('Use <{}> formatting dofs'.format(merge_option))
            if merge_option == 'one':
                gen_dofs = np.zeros((self.frames_num - 1, 6))
                gen_dofs[:batch_dofs.shape[0], :] = batch_dofs[:, 0, :]
                gen_dofs[batch_dofs.shape[0], :] = batch_dofs[-1, 1, :]
                print('gen_dof shape {}'.format(gen_dofs.shape))
                print('not average method')

            elif merge_option == 'baton':
                print('baton batch_dofs shape {}'.format(batch_dofs.shape))
                print('slice_num {}'.format(self.frames_num))
                print('neighboring {}'.format(neighbour_slice))

                gen_dofs = []
                slice_params = []
                for slice_idx in range(self.frames_num):
                    if slice_idx == 0:
                        this_params = self.case_pos[slice_idx, :]
                        slice_params.append(this_params)
                    elif slice_idx < neighbour_slice:
                        this_dof = batch_dofs[0, :] / 4
                        this_params = tools.get_next_pos(trans_params1=slice_params[slice_idx-1],
                                                         dof=this_dof,
                                                         cam_cali_mat=self.cam_cali_mat)
                        gen_dofs.append(this_dof)
                        slice_params.append(this_params)
                    else:
                        baton_idx = slice_idx - neighbour_slice + 1
                        baton_params = slice_params[baton_idx]
                        sample_dof = batch_dofs[baton_idx, :]
                        this_params = tools.get_next_pos(trans_params1=baton_params,
                                                         dof=sample_dof,
                                                         cam_cali_mat=self.cam_cali_mat)
                        this_dof = tools.get_6dof_label(trans_params1=slice_params[slice_idx-1],
                                                        trans_params2=this_params,
                                                        cam_cali_mat=self.cam_cali_mat)
                        gen_dofs.append(this_dof)
                        slice_params.append(this_params)
                gen_dofs = np.asarray(gen_dofs)
                slice_params = np.asarray(slice_params)
                print('gen_dof shape {}'.format(gen_dofs.shape))
                print('slice_params shape {}'.format(slice_params.shape))
                # time.sleep(30)
            else:
                frames_pos = []
                for start_sample_id in range(batch_dofs.shape[0]):
                    for relative_id in range(batch_dofs.shape[1]):
                        this_pos_id = start_sample_id + relative_id + 1
                        # print('this_pos_id {}'.format(this_pos_id))
                        this_pos = batch_dofs[start_sample_id, relative_id, :]
                        this_pos = np.expand_dims(this_pos, axis=0)
                        if len(frames_pos) < this_pos_id:
                            frames_pos.append(this_pos)
                        else:
                            frames_pos[this_pos_id - 1] = np.concatenate((frames_pos[this_pos_id - 1],
                                                                          this_pos), axis=0)

                gen_dofs = []
                for i in range(len(frames_pos)):
                    gen_dof = np.mean(frames_pos[i], axis=0)

                    """This is for Linear Motion"""
                    # gen_dof = train_network.dof_stats[:, 0]
                    # gen_dof = np.asarray([-0.07733258, -1.28508398, 0.37141262,
                    #                       -0.57584312, 0.20969176, 0.51404395]) + 0.1

                    gen_dofs.append(gen_dof)
                gen_dofs = np.asarray(gen_dofs)

                print('batch_dofs {}'.format(batch_dofs.shape))
                print('gen_dofs {}'.format(gen_dofs.shape))
                # time.sleep(30)

            # for dof_id in range(6):
            #     gen_dofs[:, dof_id] = tools.smooth_array(gen_dofs[:, dof_id])
            # time.sleep(30)
            return gen_dofs

        

        def draw_img_sequence(corner_pts):
            draw_ids = np.asarray([0, corner_pts.shape[0] - 1])
            for frame_id in draw_ids:
                w_weights, h_weights = np.meshgrid(np.linspace(0, 1, 224),
                                                   np.linspace(0, 1, 224))
                X = (1 - w_weights - h_weights) * corner_pts[frame_id, 0, 0] + \
                    h_weights * corner_pts[frame_id, 3, 0] + w_weights * corner_pts[frame_id, 1, 0]
                Y = (1 - w_weights - h_weights) * corner_pts[frame_id, 0, 1] + \
                    h_weights * corner_pts[frame_id, 3, 1] + w_weights * corner_pts[frame_id, 1, 1]
                Z = (1 - w_weights - h_weights) * corner_pts[frame_id, 0, 2] + \
                    h_weights * corner_pts[frame_id, 3, 2] + w_weights * corner_pts[frame_id, 1, 2]

                img_path = path.join(self.case_frames_path, self.frames_list[frame_id])
                input_img = cv2.imread(img_path, 0)
                input_img = train_dof_img.data_transform(input_img)

                input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
                input_img = input_img / 255

                if frame_id == 0 or frame_id == corner_pts.shape[0] - 1:
                    stride = 10
                else:
                    stride = 10
                self.ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride,
                                     facecolors=input_img, zorder=0.1)

        def draw_one_sequence(corner_pts, name, colorRGB=(255, 0, 0), line_width=3, constant=True):
            colorRGB = tuple(channel/255 for channel in colorRGB)
            seg_num = corner_pts.shape[0] + 1


            if constant:
                constant_color = np.asarray(colorRGB)
                constant_color = np.expand_dims(constant_color, axis=0)
                colors = np.repeat(constant_color, seg_num, axis=0)
            else:
                colors_R = np.linspace(0, colorRGB[0], seg_num).reshape((seg_num, 1))
                colors_G = np.linspace(0, colorRGB[1], seg_num).reshape((seg_num, 1))
                colors_B = np.linspace(1, colorRGB[2], seg_num).reshape((seg_num, 1))

                colors = np.concatenate((colors_R, colors_G, colors_B), axis=1)


            # for frame_id in range(int(corner_pts.shape[0] * 0.5), corner_pts.shape[0]):
            #     if frame_id == int(corner_pts.shape[0] * 0.5):
            for frame_id in range(corner_pts.shape[0]):
                if frame_id == 0:
                    """ First frame draw full bounds"""
                    for pt_id in range(-1, 3):
                        xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id, pt_id + 1, 0]
                        ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id, pt_id + 1, 1]
                        zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id, pt_id + 1, 2]
                        self.ax.plot(xs, ys, zs, color=tuple(colors[frame_id, :]), lw=line_width, zorder=1)
                elif frame_id == corner_pts.shape[0] - 1:
                    """ Connect to the former frame """
                    for pt_id in range(-1, 3):
                        xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id - 1, pt_id, 0]
                        ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id - 1, pt_id, 1]
                        zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id - 1, pt_id, 2]
                        self.ax.plot(xs, ys, zs, color=tuple(colors[frame_id, :]), lw=line_width)
                    """ Last frame draw full bounds"""
                    for pt_id in range(-1, 3):
                        xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id, pt_id + 1, 0]
                        ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id, pt_id + 1, 1]
                        zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id, pt_id + 1, 2]
                        self.ax.plot(xs, ys, zs, color=tuple(colors[-1, :]), lw=line_width)
                        if pt_id == -1:
                            self.ax.plot(xs, ys, zs, color=tuple(colors[-1, :]), lw=line_width, label=name)
                else:
                    """ Connect to the former frame """
                    for pt_id in range(-1, 3):
                        xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id - 1, pt_id, 0]
                        ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id - 1, pt_id, 1]
                        zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id - 1, pt_id, 2]
                        self.ax.plot(xs, ys, zs, color=tuple(colors[frame_id, :]), lw=line_width, zorder=1)

                # if plot_img and frame_id==0:


        def visualize_sequences():
            self.fig = plt.figure()
            self.ax = self.fig.gca(projection='3d')
            draw_img_sequence(corner_pts=self.gt_pts1)
            draw_one_sequence(corner_pts=self.gt_pts1, name='Groundtruth',
                              colorRGB=(0, 153, 76), line_width=3)
            draw_one_sequence(corner_pts=self.trans_pts1, name='DCL-Net ({:.4f}mm)'.format(self.trans_pts1_error),
                              colorRGB=(255, 0, 0))

            # plt.axis('off')
            # self.ax.set_xticklabels([])
            # self.ax.set_yticklabels([])
            # self.ax.set_zticklabels([])
            plt.legend(loc='lower left')
            plt.tight_layout()

            # views_id = np.linspace(0, 360, 36)
            # for ii in views_id:
            #     self.ax.view_init(elev=10., azim=ii)
            #     plt.savefig('views/{}_img.jpg'.format(ii))
            #     # plt.savefig('views/{}.jpg'.format(ii))
            #     print('{} saved'.format(ii))

            self.ax.view_init(elev=10., azim=0)
            plt.title(self.case_id)
            plt.savefig('views/all_cases/{}_{}.jpg'.format(model_string, case_id))
            plt.show()
            plt.close()

        def get_gt_dofs():
            # print('self.case_pos:\n{}'.format(np.around(self.case_pos, decimals=2)))
            # sys.exit()
            gt_dofs = []
            for slice_id in range(1, self.frames_num):
                params1 = self.case_pos[slice_id-1, :]
                params2 = self.case_pos[slice_id, :]
                this_dof = tools.get_6dof_label(trans_params1=params1,
                                                trans_params2=params2,
                                                cam_cali_mat=self.cam_cali_mat)
                gt_dofs.append(this_dof)
            gt_dofs = np.asarray(gt_dofs)
            print('gt_dof shape {}, frames_num {}'.format(gt_dofs.shape, self.frames_num))
            return gt_dofs

        def visualize_dofs():
            frees = ['tX', 'tY', 'tZ', 'aX', 'aY', 'aZ']
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Case{:04}'.format(self.case_id))
            for dof_id in range(len(frees)):
                plot_x = dof_id // 3
                plot_y = dof_id % 3
                axes[plot_x, plot_y].plot(self.gt_dofs[:, dof_id], color='g', label='Groundtruth', alpha=0.5)
                axes[plot_x, plot_y].plot(self.format_dofs[:, dof_id], color='r', label='CNN', alpha=0.5)

                corrcoef = np.corrcoef(self.gt_dofs[:, dof_id], self.format_dofs[:, dof_id])[0, 1]

                axes[plot_x, plot_y].set_title('{}: corrcoef {:.4f}'.format(frees[dof_id], corrcoef))
                axes[plot_x, plot_y].legend(loc='lower left')
                # axes[plot_x, plot_y].show()

                np.savetxt('figures/dof_values/{}_{}_gt.txt'.format(self.case_id, frees[dof_id]),
                           self.gt_dofs[:, dof_id])
                np.savetxt('figures/dof_values/{}_{}_{}_pd.txt'.format(model_string, self.case_id, frees[dof_id]),
                           self.format_dofs[:, dof_id])


            plt.savefig('figures/dof_pred/Case{:04}.jpg'.format(self.case_id))
            # plt.show()

        def mats2params(mats):
            res_params = []
            for trans_mat in mats:
                params = tools.mat2params(trans_mat)
                res_params.append(params)
            res_params = np.asarray(res_params)
            return res_params
    
    def dof2params(self, format_dofs):
        gen_param_results = []
        for i in range(format_dofs.shape[0]):
            if i == 0:
                base_param = self.case_pos[i, :]
            else:
                base_param = gen_param_results[i-1]
            gen_dof = format_dofs[i, :]
            gen_param = tools.get_next_pos(trans_params1=base_param,
                                            dof=gen_dof, cam_cali_mat=self.cam_cali_mat)
            gen_param_results.append(gen_param)
        # time.sleep(30)
        gen_param_results = np.asarray(gen_param_results)
        pos_params = np.zeros((format_dofs.shape[0]+1, 7))
        pos_params[0, :] = self.case_pos[0, 2:]
        pos_params[1:, :] = gen_param_results
        print('pos_params shape {}'.format(pos_params.shape))
        # time.sleep(30)
        return pos_params

    def reg2frames(self, frame1, frame2, mat1, dof1=None, use_slice=False):
        frame1 = data_transform(frame1, masked_full=False)
        frame2 = data_transform(frame2, masked_full=False)
        # print('frame1 {:.2f}, {:.2f}'.format(np.mean(frame1), np.std(frame1)))
        # print('frame2 {:.2f}, {:.2f}'.format(np.mean(frame2), np.std(frame2)))
        sample_slices = np.asarray([frame1, frame2])
        # print('sample_slices {:.2f}, {:.2f}'.format(np.mean(sample_slices), np.std(sample_slices)))
        sample_slices = torch.from_numpy(sample_slices).float().unsqueeze(0).unsqueeze(0).to(model_device)
        # print('sample_slices {}'.format(sample_slices.shape))
        if use_slice:
            outputs, vectors = model_ft_slice(sample_slices)
        else:
            outputs, vectors = model_ft(sample_slices)
        # print('outputs {}'.format(outputs.shape))

        dof_pd = outputs.data.cpu().numpy().squeeze()

        tr_dot = np.dot(mat1, np.linalg.inv(self.cam_cali_mat))
        prev_params = tools.mat2params(tr_dot)

        this_params = tools.get_next_pos(trans_params1=prev_params,
                                         dof=dof_pd, 
                                         cam_cali_mat=self.cam_cali_mat)

        this_mat = tools.params2mat_new(trans_params=this_params,
                                        cam_cali_mat=self.cam_cali_mat,
                                        calibration=True)[0]

        # gt_mat1 = self.gt_mats[0, :, :]
        # gt_mat2 = self.gt_mats[1, :, :]
        # dof12 = tools.get_6dof_from_mats(gt_mat1, gt_mat2)
        # gt_mat2_back = tools.get_next_mat(gt_mat1, dof12)
        # print('gt_mat2\n{}'.format(gt_mat2))
        # print('gt_mat2_back\n{}'.format(gt_mat2_back))
        # sys.exit()

        return this_mat, dof_pd
    
    def adjust_mat1(self, frame, input_mat, gt_mat=None, frame_id=None):
        pd_mat = copy.copy(input_mat)
        us_frame = copy.copy(frame)
        # slice_gt = tools.resample_slice(self.us_sitk, gt_mat, crop_frame=True)
        slice_pd = tools.resample_slice(self.us_sitk, pd_mat, crop_frame=True)
        # concat = np.concatenate((us_canvas, slice_gt, slice_pd), axis=0)
        
        mat_correction, _ = self.reg2frames(slice_pd, us_frame, pd_mat, use_slice=True)
        # print(mat_correction.shape)
        
        return mat_correction
    
    def adjust_mat2(self, frame, input_mat, gt_mat=None, frame_id=None):
        pd_mat = copy.copy(input_mat)
        us_frame = copy.copy(frame)
        slice_pd = tools.resample_slice(self.us_sitk, pd_mat, crop_frame=True)
        
        mat_correction, _ = self.reg2frames(slice_pd, us_frame, pd_mat, use_slice=True)
        
        slice_pd_adjust = tools.resample_slice(self.us_sitk, mat_correction, crop_frame=True)

        sim_1 = tools.img_cor(frame, slice_pd)
        sim_2 = tools.img_cor(frame, slice_pd_adjust)
        # print('sim1 {:.4f}, sim2 {:.4f}'.format(sim_1, sim_2))

        if sim_1 > sim_2:
            return pd_mat
        else:
            return mat_correction
    
    def adjust_mat3(self, frame, input_mat, gt_mat=None, frame_id=None):
        pd_mat = copy.copy(input_mat)
        us_frame = copy.copy(frame)


        slice_pd = tools.resample_slice(self.us_sitk, pd_mat, crop_frame=True)
        sim_1 = tools.img_cor(frame, slice_pd)

        prev_mat = pd_mat
        prev_sim = sim_1
        # print('sim {:.2f}'.format(prev_sim))
        # sys.exit()

        while True:
            mat_correction, _ = self.reg2frames(slice_pd, us_frame, prev_mat, use_slice=True)
            slice_pd_adjust = tools.resample_slice(self.us_sitk, mat_correction, crop_frame=True)
            current_sim = tools.img_cor(frame, slice_pd_adjust)

            if current_sim == 0:
                break

            if current_sim >= prev_sim:
                prev_mat = mat_correction
                prev_sim = current_sim
            else:
                break

        return prev_mat
    
    def adjust_mat4(self, frame, input_mat, gt_mat=None, frame_id=None, iterations=5):
        pd_mat = copy.copy(input_mat)
        us_frame = copy.copy(frame)


        slice_pd = tools.resample_slice(self.us_sitk, pd_mat, crop_frame=True)
        sim_1 = tools.img_cor(frame, slice_pd)

        prev_mat = [pd_mat]
        prev_sim = [sim_1]
        # print('sim {:.2f}'.format(prev_sim))
        # sys.exit()

        for it in range(iterations):
            mat_correction, _ = self.reg2frames(slice_pd, us_frame, prev_mat[-1], use_slice=True)
            slice_pd_adjust = tools.resample_slice(self.us_sitk, mat_correction, crop_frame=True)
            current_sim = tools.img_cor(frame, slice_pd_adjust)

            prev_mat.append(mat_correction)
            prev_sim.append(current_sim)
        
        best_idx = np.argmax(np.asarray(prev_sim))

        return prev_mat[best_idx], best_idx

        
    def reg_sequence(self, seq_ids=None, adjust='0'):
        if seq_ids is None:
            seq_ids = np.linspace(0, self.frames_num-1, self.frames_num).astype(np.int)[::-1]

        # print(seq_ids)
        vis_dir = 'tmp/vis_fig/{}'.format(case_id)
        if not os.path.isdir(vis_dir):
            os.mkdir(vis_dir)

        all_mats = [case.gt_mats[seq_ids[0], :, :]]
        all_params = [case.case_pos[seq_ids[0], :]]
        all_dofs = []
        
        all_err = []
        all_sim = []
        all_dice = []
        all_framerate = []

        img_list = []

        all_params = [self.case_pos[seq_ids[0], :]]

        mr_mats_gt = []
        mr_mats_pd = []
        # start = time.time()

        landmark_dists_gt = []
        landmark_dists_pd = []

        best_idx_list = []

        for i in range(1, seq_ids.shape[0]):
            frame_id1, frame_id2 = seq_ids[i-1], seq_ids[i]

            frame1 = cv2.imread(path.join(case.case_frames_path, '{:04}.jpg'.format(frame_id1)), 0)
            frame2 = cv2.imread(path.join(case.case_frames_path, '{:04}.jpg'.format(frame_id2)), 0)
            # print(path.join(case.case_frames_path, '{:04}.jpg'.format(frame_id1)))

            # sys.exit()
            # frame1 = tools.resample_slice(self.us_sitk, self.gt_mats[frame_id1], crop_frame=True)
            # frame2 = tools.resample_slice(self.us_sitk, self.gt_mats[frame_id2], crop_frame=True)

            mat1 = all_mats[-1]
            start = time.time()
            this_mat, this_dof = self.reg2frames(frame1, frame2, mat1, use_slice=False)
            if adjust == '1':
                this_mat = self.adjust_mat1(frame2, this_mat, 
                                        gt_mat=self.gt_mats[frame_id2, :, :],
                                        frame_id=frame_id2)
            elif adjust == '2':
                this_mat = self.adjust_mat2(frame2, this_mat, 
                                        gt_mat=self.gt_mats[frame_id2, :, :],
                                        frame_id=frame_id2)
            elif adjust == '3':
                this_mat = self.adjust_mat3(frame2, this_mat, 
                                        gt_mat=self.gt_mats[frame_id2, :, :],
                                        frame_id=frame_id2)
            elif adjust == '4':
                this_mat, best_idx = self.adjust_mat4(frame2, this_mat, 
                                        gt_mat=self.gt_mats[frame_id2, :, :],
                                        frame_id=frame_id2,
                                        iterations=10)
                best_idx_list.append(best_idx)
            end = time.time()
            # print('this_mat {}'.format(this_mat.shape))
            # sys.exit()
            # print('mat1\n{}'.format(mat1))
            # print('mat2\n{}'.format(this_mat))

            us_slice_gt = tools.resample_slice(self.us_sitk, self.gt_mats[frame_id2, :, :], crop_frame=True)
            us_slice_pd = tools.resample_slice(self.us_sitk, this_mat, crop_frame=True)

            # us_gt_slice = tools.crop_fan(us_slice_gt)
            # us_pd_slice = tools.crop_fan(us_slice_pd)
            # us_frame = tools.crop_fan(frame2)
            # img_dir = 'tmp/slices_fig/{}'.format(case_id)
            # if not os.path.isdir(img_dir):
            #     os.mkdir(img_dir)
            # cv2.imwrite(path.join(img_dir, '{:04}_gt.jpg'.format(frame_id2)), us_gt_slice)
            # cv2.imwrite(path.join(img_dir, '{:04}_pd_{}.jpg'.format(frame_id2, adjust)), us_pd_slice)
            # cv2.imwrite(path.join(img_dir, '{:04}_frame.jpg'.format(frame_id2)), us_frame)
            
            # us_frame = tools.format_vid_frame(frame2)
            # mr_mat_gt = np.dot(inv(self.compose_mat), self.gt_mats[frame_id2, :, :])
            # mr_mat_pd = np.dot(inv(self.compose_mat), this_mat)
            # mr_slice_gt = tools.resample_slice(self.mr_sitk, mr_mat_gt)
            # mr_slice_pd = tools.resample_slice(self.mr_sitk, mr_mat_pd)
            # fused_gt = load_func.fuse_images(mr_slice_gt, us_frame, alpha=0.3)
            # fused_pd = load_func.fuse_images(mr_slice_pd, us_frame, alpha=0.3)
            # concat = np.concatenate((fused_gt, fused_pd), axis=1)

            # cv2.imwrite(path.join(vis_dir, '{:04}_gt.jpg'.format(frame_id2)), fused_gt)
            # cv2.imwrite(path.join(vis_dir, '{:04}_pd_{}.jpg'.format(frame_id2, adjust)), fused_pd)

            # if self.mr_seg:
            #     mr_seg_gt = tools.resample_slice(self.mr_seg, mr_mat_gt).astype(np.uint8) * 255
            #     mr_seg_pd = tools.resample_slice(self.mr_seg, mr_mat_pd).astype(np.uint8) * 255
            #     # plt.imshow(mr_seg_gt)
            #     # plt.contour(mr_seg_gt)
            #     # plt.show()
            #     # sys.exit()
            #     seg_concat = np.concatenate((mr_seg_gt, mr_seg_pd), axis=1)
            #     # print('concat {}, seg {}'.format(concat.shape, seg_concat.shape))
            #     # sys.exit()
            #     concat = concat.astype(np.uint8)
            #     # seg_concat = seg_concat.astype(np.uint8) * 255
            #     concat = load_func.fuse_images(concat, seg_concat, alpha=0.7)

            #     mr_seg_gt = load_func.fuse_images(mr_slice_gt, mr_seg_gt, alpha=0.7)
            #     mr_seg_pd = load_func.fuse_images(mr_slice_pd, mr_seg_pd, alpha=0.7)

            #     mr_seg_gt = cv2.cvtColor(mr_seg_gt, cv2.COLOR_BGR2RGB)
            #     mr_seg_pd = cv2.cvtColor(mr_seg_pd, cv2.COLOR_BGR2RGB)

            #     # cv2.imwrite('tmp/seg_slice/{:04}_gt.jpg'.format(frame_id2), mr_seg_gt)
            #     # cv2.imwrite('tmp/seg_slice/{:04}_pd.jpg'.format(frame_id2), mr_seg_pd)
            #     # plt.figure()
            #     # plt.imshow(seg_concat)
            #     # plt.show()
            #     # sys.exit()

            # landmark_dist_gt = tools.landmark_distance(self.landmarks, mr_mat_gt)
            # landmark_dist_pd = tools.landmark_distance(self.landmarks, mr_mat_pd)
            # landmark_dists_gt.append(landmark_dist_gt)
            # landmark_dists_pd.append(landmark_dist_pd)

            score_err  = tools.frame_err(gt_mat=self.gt_mats[frame_id2, :, :], pd_mat=this_mat,
                                         spacing=[1])
                                        #  spacing=self.raw_spacing)
            score_sim  = tools.img_cor(gt_img=tools.data_transform(us_slice_gt), 
                                       pd_img=tools.data_transform(us_slice_pd))
            # score_dice = tools.dice(pred=mr_seg_pd, true=mr_seg_gt)
            score_dice = 1.0
            # print('Err : {:.4}mm'.format(score_err))
            # print('Sim : {:.4}'.format(score_sim))
            # print('Dice: {:.4}'.format(score_dice))
            # print('spacing {}'.format(self.raw_spacing))
            # sys.exit()

            all_err.append(score_err)
            all_sim.append(score_sim)

            if score_dice >= 0:
                all_dice.append(score_dice)
            all_framerate.append(1/(end-start))
            all_mats.append(this_mat)
            # all_params.append(this_params)
            all_dofs.append(this_dof)
            # img_list.append(concat)

            # print('{}/{}: {:04}-{:04} finished'.format(i+1, seq_ids.shape[0], frame_id1, frame_id2))
        # sys.exit()
        end = time.time()
        time_cost = end - start
        frame_rate = len(seq_ids) / time_cost
        all_mats = np.asarray(all_mats)
        all_dofs = np.asarray(all_dofs)

        # mr_mats_gt = np.asarray(mr_mats_gt)
        # mr_mats_pd = np.asarray(mr_mats_pd)

        # landmark_dists_gt = np.asarray(landmark_dists_gt)
        # landmark_dists_pd = np.asarray(landmark_dists_pd)
        # np.save('tmp/evaluations/{}_{}_landmark_dist_gt.npy'.format(case_id, adjust), landmark_dists_gt)
        # np.save('tmp/evaluations/{}_{}_landmark_dist_pd.npy'.format(case_id, adjust), landmark_dists_pd)
        # np.save('tmp/evaluations/{}_best_idx.npy'.format(case_id, adjust), np.asarray(best_idx_list))
        # # print(best_idx_list)
        # # sys.exit()

        all_err = np.asarray(all_err)
        mean_err = np.mean(all_err)

        all_sim = np.asarray(all_sim)
        mean_sim = np.mean(all_sim)

        all_dice = np.asarray(all_dice)
        mean_dice = np.mean(all_dice)

        all_framerate = np.asarray(all_framerate)
        mean_framerate = np.mean(all_framerate)
        std_framerate = np.std(all_framerate)


        print('Mean Err : {:.4f} mm'.format(mean_err))
        print('Mean Sim : {:.4f}'.format(mean_sim))
        print('Mean Dice: {:.4f}'.format(mean_dice))
        print('FrameRate: {:.4f} ({:.4f})'.format(mean_framerate, std_framerate))
        # sys.exit()
        evaluations = {'err': all_err,
                       'sim': all_sim,
                       'dice': all_dice}
        
        # self.result_params = self.dof2params(all_dofs)

        # gt_corner_pts = tools.mats2pts_correction(mr_mats_gt)
        # pd_corner_pts = tools.mats2pts_correction(mr_mats_gt)

        # print('landmark {}'.format(self.landmarks.shape))
        # print('mr_mats_gt {}'.format(mr_mats_gt.shape))

        # sys.exit()
        # print('all_mats {}'.format(all_mats.shape))
        # print('all_dofs {}'.format(all_dofs.shape))
        # sys.exit()
        # mr_vol_info = {'img_size': self.mr_sitk.GetSize(),
        #                'img_spacing': self.mr_sitk.GetSpacing(),
        #                'img_origin': self.mr_sitk.GetOrigin()}
        # fig_list = draw_3d_plot.draw_fig(init_mat=mr_mats_gt[0,:,:],
        #                                  gt_mats=mr_mats_gt,
        #                                  pd_mats=mr_mats_pd,
        #                                  frame_ids=seq_ids,
        #                                  frames_num=self.frames_num,
        #                                  mr_vol_info=mr_vol_info,
        #                                  landmarks=self.landmarks,
        #                                  title=case_id,
        #                                  caption='{:04} - {:04}'.format(seq_ids[0], seq_ids[-1]))

        # print('mr_mats_gt {}'.format(mr_mats_gt.shape))
        # print('mr_mats_pd {}'.format(mr_mats_pd.shape))
        # print('self.landmarks {}'.format(self.landmarks.shape))

        

        # np.save(path.join(vis_dir, 'mats_gt.npy'), mr_mats_gt)
        # np.save(path.join(vis_dir, 'mats_{}.npy'.format(adjust)), mr_mats_pd)
        # np.save(path.join(vis_dir, 'landmarks.npy'), self.landmarks)
        # np.save(path.join(vis_dir, 'mr_vol_info.npy'), mr_vol_info)

        # np.save(path.join(vis_dir, 'evaluations_{}.npy'.format(adjust)), evaluations)
        # np.save('tmp/c95_5cv_eval/{}_m{}.npy'.format(case_id, adjust), evaluations)
        # np.save('tmp/range_exp/10-60/{}_m{}.npy'.format(case_id, adjust), evaluations)
        # np.save('tmp/range_exp/20-70/{}_m{}.npy'.format(case_id, adjust), evaluations)
        # np.save('tmp/range_exp/30-80/{}_m{}.npy'.format(case_id, adjust), evaluations)
        np.save('tmp/range_exp/10-60/{}_m{}.npy'.format(case_id, adjust), evaluations)
        # print(evaluations)

        # concate = []
        # img_h = 500
        # for i, image in enumerate(img_list):
        #     gt_image = image[:, :1000, ::-1]
        #     pd_image = image[:, 1000:, ::-1]
        #     cv2.imwrite(path.join(vis_dir, '{:04}_gt.jpg'.format(i)), gt_image)
        #     cv2.imwrite(path.join(vis_dir, '{:04}_{}.jpg'.format(i, adjust)), pd_image)


        #     image = cv2.resize(image, (img_h*2, img_h))
        #     fig = fig_list[i]
        #     fig = cv2.resize(fig, (img_h, img_h))
        #     concate.append(np.concatenate((image, fig), axis=1))
        # gif_path = 'tmp/gif_dcl2/{}_{:04}_{:04}_{}.gif'.format(self.case_id, seq_ids[0], seq_ids[-1], adjust)
        # imageio.mimsave(gif_path, concate, duration=0.1)
        # print('Adjust method: {}'.format(adjust))
        # print('gif_path: <{}>'.format(gif_path))
        
        # evaluations = {'err': all_err,
        #                'sim': all_sim,
        #                'dice': all_dice}
        # print(all_err)
        print('='*50)
        return evaluations




if __name__ == '__main__':
    batch_size = 5
    neighbour_slice = 2
    network_type = 'resnext50'
    # network_type = 'convlstm'
    input_type = 'org_img'
    output_type = 'average_dof'
    normalize_dof = False
    # normalize_dof = True
    # device = torch.device("cuda:{}".format(device_no))

    # device_no = -1

    if device_no >= 0:
        model_device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
    else:
        model_device = torch.device("cpu")
        print('use cpu')
    
    for fold_idx in range(9, 10):

        fold = 'f{}'.format(fold_idx)
        test_ids = tools.read_list('infos/sets/c95_5cv/{}/test.txt'.format(fold))
        model_string = 'dcl2_big_m25_ns2_bi_{}'.format(fold)
        model_string2 = 'dcl2_big_m25_ns2_bi_slice_{}'.format(fold)

        model_folder = '/zion/guoh9/projects/FreehandUSReconClean/pretrained_networks'
        model_path = path.join(model_folder, '3d_best_Generator_{}.pth'.format(model_string))
        model_ft = tools.define_model(model_type=network_type,
                                            pretrained_path=model_path,
                                            input_type=input_type,
                                            output_type=output_type,
                                            neighbour_slice=neighbour_slice,
                                            model_device=model_device)

        model_path_slice = path.join(model_folder, '3d_best_Generator_{}.pth'.format(model_string2))
        model_ft_slice = tools.define_model(model_type=network_type,
                                            pretrained_path=model_path_slice,
                                            input_type=input_type,
                                            output_type=output_type,
                                            neighbour_slice=neighbour_slice,
                                            model_device=model_device)


        model_pred_result_dir = '/zion/guoh9/US_recon/pred_pos/{}'.format(model_string)
        if not path.isdir(model_pred_result_dir):
            os.mkdir(model_pred_result_dir)

        params = model_ft.state_dict()
        fc_weights = params['fc.weight'].data.cpu().numpy()

        since = time.time()

        # print(test_ids)

        """ Following parts are for our full dataset testing """
        errors = []
        final_drifts = []
        frame_errors = []
        frame_nums = []
        corr_coefs = []
        drift_rates = []
        
        # test_ids = tools.read_list('infos/sets/fvr_ready/test.txt')
        # test_ids = tools.read_list('infos/c95/seg_nii.txt')
        # print(test_ids)
        # sys.exit()

        # print(test_ids)
        # sys.exit()


        for i, case_id in enumerate(test_ids):
            # wrong_ids = ['Case0500', 'Case0621', "Case0650"]
            # if case_id in wrong_ids:
            #     continue
            # case_id = 'Case0500'
            # case_id = 'Case0004'
            # case_id = 'Case0005'
            print('=== <{}> ==='.format(case_id))
            case = TestNetwork2(case_id=case_id)

            frames_num = case.frames_num
            mid_id = frames_num // 2
            # sys.exit()

            # """ Long sequence Case0004, 60-110, 10 rounds"""
            # start, end = 50, 100
            # start, end = 60, 110
            # seq_ids = np.linspace(start, end, end-start+1).astype(np.int)
            # seq_ids_back = seq_ids[::-1]
            # new_seq = np.asarray([])
            # for i in range(10):
            #     new_seq = np.concatenate((new_seq, seq_ids, seq_ids_back))
            # seq_ids = new_seq.astype(np.uint8)
            # print(seq_ids.shape)
            # # print(seq_ids)

            # """ Long sequence Case0004, 85-110-60, 10 rounds """
            # mid, start, end = 70, 60, 110
            # seq_ids1 = np.linspace(mid, end, end-mid+1).astype(np.int)
            # seq_ids2 = np.linspace(end, start, end-start+1).astype(np.int)
            # seq_ids3 = np.linspace(start, mid, mid-start+1).astype(np.int)[:-1]
            # seq_ids = np.concatenate((seq_ids1, seq_ids2, seq_ids3))
            # new_seq = np.asarray([])
            # for i in range(10):
            #     new_seq = np.concatenate((new_seq, seq_ids))
            # seq_ids = new_seq.astype(np.uint8)
            # print(seq_ids.shape)
            # print(seq_ids)
            # # sys.exit()


            # start = int(frames_num*0.1)
            # end = int(frames_num*0.9)

            start = int(frames_num*0.1)
            end = int(frames_num*0.6)
            seq_ids = np.linspace(start, end, end-start+1).astype(np.int)
            # print(seq_ids)
            evals_0 = case.reg_sequence(seq_ids, adjust='0')
            # np.save('tmp/evaluations/{}_{}.npy'.format(case_id, '0'), evals_0)
            evals_1 = case.reg_sequence(seq_ids, adjust='1')
            # np.save('tmp/evaluations/{}_{}.npy'.format(case_id, '1'), evals_1)
            # evals_1 = case.reg_sequence(seq_ids, adjust='2')
            # np.save('tmp/evaluations/{}_{}.npy'.format(case_id, '1'), evals_1)
            evals_3 = case.reg_sequence(seq_ids, adjust='3')
            # np.save('tmp/evaluations/{}_{}.npy'.format(case_id, '3'), evals_3)
            # sys.exit()

        


