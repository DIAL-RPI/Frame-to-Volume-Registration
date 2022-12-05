# Demo image registration using SimpleITK

from copy import copy
from cv2 import sqrt
from matplotlib import pyplot as plt
import numpy as np
import SimpleITK as sitk
import time
import pandas as pd
from os import path
import os
import sys
import cv2
# import imageio
import torch
import torchgeometry as tgm
import math
from utils import transformations as tfms
import random
# import nvidia_smi
from numpy.linalg import inv
import copy
from networks import resnext
import torch
import torch.nn as nn

uronav_dataset = '/zion/common/data/uronav_data'
usrecon_dataset = '/zion/guoh9/US_recon/US_dataset'
myvol_dataset = '/zion/guoh9/US_recon/recon'
seq_dataset = '/zion/guoh9/US_recon/new_data'
fan_mask = cv2.imread('data/avg_img.png', 0)

def print_gpu_use(gpu_id=0):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    print('GPU Occupied {:02}%: {}/{}, '.format(info.used*100/info.total, info.used, info.total))
    return 0

def pic2gif(folder):
    gifs = []
    for i in range(fixedArray.shape[0]):
        gifs.append(fixedArray[i, :, :])
    imageio.mimsave('plots/compare.gif', gifs, duration=0.2)


def folder2imglist(folder):
    file_list = os.listdir(folder)
    file_list.sort()
    img_list = []
    for filename in file_list:
        img_path = path.join(folder, filename)
        img_list.append(cv2.imread(img_path, 1))
    return img_list

def case2gif(case_id):
    multimodal_folder = 'results/{}/multimodal'.format(case_id)
    img_list = folder2imglist(folder=multimodal_folder)
    gif_path = 'results/{}/{}_fused.gif'.format(case_id, case_id)
    imageio.mimsave(gif_path, img_list, duration=0.2)
    print('{} gif saved!'.format(case_id))

def folder2gif(folder, gif_path, duration=0.2):
    img_list = folder2imglist(folder=folder)
    imageio.mimsave(gif_path, img_list, duration=duration)
    print('{} gif saved!'.format(gif_path))

def data_transform(input_img, crop_size=224, resize=224, normalize=False, masked_full=False):
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
        print(masked_full)
        return masked_full_img

    h, w = input_img.shape
    if crop_size > 480:
        crop_size = 480
    x_start = int((h - crop_size) / 2)
    y_start = int((w - crop_size) / 2)

    patch_img = input_img[x_start:x_start+crop_size, y_start:y_start+crop_size]
    # print('x_start {}, y_start {}, crop_size {}'.format(x_start, y_start, crop_size))


    patch_img = cv2.resize(patch_img, (resize, resize))
    # cv2.imshow('patch', patch_img)
    # cv2.waitKey(0)
    if normalize:
        patch_img = patch_img.astype(np.float64)
        patch_img = (patch_img - np.min(patch_img)) / (np.max(patch_img) - np.mean(patch_img))

    return patch_img


def volCompare(case_id):
    uronav_case_folder = path.join(uronav_dataset, case_id)
    myvol_case_folder = path.join(myvol_dataset, case_id)
    print(os.listdir(uronav_case_folder))
    print(os.listdir(myvol_case_folder))

    vol_uronav = sitk.ReadImage(path.join(uronav_case_folder, 'USVol.mhd'),
                                sitk.sitkFloat64)
    vol_my = sitk.ReadImage(path.join(myvol_case_folder, '{}_myrecon.mhd'.format(case_id)),
                            sitk.sitkFloat64)
    print('vol_uronav\n{}'.format(vol_uronav.GetSize()))
    print('vol_my\n{}'.format(vol_my.GetSize()))

    vol_uronav_np = sitk.GetArrayFromImage(vol_uronav)
    vol_my_np = sitk.GetArrayFromImage(vol_my)

    print('uronav_np {}, my_np {}'.format(
        vol_uronav_np.shape, vol_my_np.shape))
    cv2.imwrite('tmp.jpg', vol_uronav_np[20, :, :])
    cv2.imwrite('tmp2.jpg', vol_my_np[20, :, :])


def readMatsFromSequence(case_id, type='adjusted', model_str='gt', on_arc=False):
    """ Read a sequence .mhd file and return frame_num*4*4 transformation mats

    Args:
        case_id (str): case ID like "Case0005"
        type (str, optional): Whether bottom centerline is adjuested 
        or origin. Defaults to 'adjusted'.
        model_str (str, optional): Could be model's time string. Defaults to 'gt'.

    Returns:
        Numpy array: frame_num x 4 x 4 transformation mats for each frame
    """
    if on_arc:
        case_seq_folder = '/raid/shared/guoh9/US_recon/new_data/{}'.format(case_id)
        # case_seq_folder = '/raid/shared/guoh9/US_recon'
    else:
        case_seq_folder = path.join(seq_dataset, case_id)
    # print(os.listdir(case_seq_folder))
    # sys.exit()
    case_seq_path = path.join(
        case_seq_folder, '{}_{}_{}.mhd'.format(case_id, type, model_str))
    # print(case_seq_path)
    file = open(case_seq_path, 'r')
    lines = file.readlines()
    mats = []
    for line in lines:
        words = line.split(' ')
        if words[0].endswith('ImageToProbeTransform'):
            # print(words)
            words[-1] = words[-1][:-2]
            nums = np.asarray(words[2:]).astype(np.float)
            nums.shape = (4, 4)
            mats.append(nums)
    mats = np.asarray(mats)
    return mats

def readMatsFromSequence2(sequence_path, on_arc=False):
    """ Read a sequence .mhd file and return frame_num*4*4 transformation mats

    Args:
        case_id (str): case ID like "Case0005"
        type (str, optional): Whether bottom centerline is adjuested 
        or origin. Defaults to 'adjusted'.
        model_str (str, optional): Could be model's time string. Defaults to 'gt'.

    Returns:
        Numpy array: frame_num x 4 x 4 transformation mats for each frame
    """
    case_seq_path = sequence_path
    # print(case_seq_path)
    file = open(case_seq_path, 'r')
    lines = file.readlines()
    mats = []
    for line in lines:
        words = line.split(' ')
        if words[0].endswith('ImageToProbeTransform'):
            # print(words)
            words[-1] = words[-1][:-2]
            nums = np.asarray(words[2:]).astype(np.float)
            nums.shape = (4, 4)
            mats.append(nums)
    mats = np.asarray(mats)
    return mats

def read_list(path):
    mylist = []
    with open(path, 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            mylist.append(currentPlace)
    return mylist

def computeScale(input_mat):
    scale1 = np.linalg.norm(input_mat[:3, 0])
    scale2 = np.linalg.norm(input_mat[:3, 1])
    scale3 = np.linalg.norm(input_mat[:3, 2])
    # print('scale1 {}'.format(scale1))
    # print('scale2 {}'.format(scale2))
    # print('scale3 {}'.format(scale3))
    # print(0.478425 * 0.35)
    # sys.exit()
    return np.asarray([scale1, scale2, scale3])



def samplePlane(case_id, trans_mats, frame_id):
    us_path = path.join(myvol_dataset, '{}/{}_myrecon.mhd'.format(case_id, case_id))
    us_img = sitk.ReadImage(us_path)
    us_np = sitk.GetArrayFromImage(us_img)
    print(us_img.GetOrigin())
    print('us_np shape {}'.format(us_np.shape))
    print('us_img size {}'.format(us_img.GetSize()))
    fixed_path = path.join(usrecon_dataset, '{}/frames/{:04}.jpg'.format(case_id, frame_id))
    fixed_origin = cv2.imread(fixed_path, 0)

    clip_x, clip_y, clip_h, clip_w = 105, 54, 320, 565
    fixed_np = fixed_origin[clip_x:clip_x+clip_h, clip_y:clip_y+clip_w]
    # fixed_np = fixed_origin[105:105+320, 54:54+565]

    # spacing = 0.4   # For my Slicer reconstructed volume
    # spacing = 0.35  # For uronac reconstructed volume
    mat_scales = computeScale(input_mat=trans_mats[frame_id, :, :])
    spacing = np.mean(mat_scales[:2]) / us_img.GetSpacing()[0]
    print('frame_scale = {}'.format(spacing))
    frame_w = int(spacing * fixed_np.shape[1])
    frame_h = int(spacing * fixed_np.shape[0])
    fixed_np = cv2.resize(fixed_np, (frame_w, frame_h))
    fixed_np = fixed_np.astype(np.float64)
    fixed_np = np.expand_dims(fixed_np, axis=0)
    print('fixed_np shape {}'.format(fixed_np.shape))

    fixed_image = sitk.GetImageFromArray(fixed_np)
    # fixed_image.SetSpacing(us_img.GetSpacing())

    frame_mat = trans_mats[frame_id, :, :]
    # print('us_img {}'.format(us_img))
    # print('frame_mat\n{}'.format(frame_mat))


    # tfm2us = sitk.Transform(mat2tfm(np.identity(4)))
    # affine_tfm = sitk.AffineTransform(3)
    # affine_tfm.SetMatrix(frame_mat[:3, :3].flatten())
    # affine_tfm.SetTranslation(frame_mat[:3, 3])
    # print(affine_tfm)

    # spacing1 = us_img.GetSpacing()[0]
    # print('spacing1 {}, spacing {}'.format(spacing1, spacing))
    # width, length = fixed_origin.shape[1], fixed_origin.shape[0]
    destVol = sitk.Image(int(clip_w*spacing), int(clip_h*spacing), 1, sitk.sitkUInt8)
    destSpacing = np.asarray([spacing, spacing, spacing])
    destVol.SetSpacing((1/destSpacing[0], 1/destSpacing[1], 1/destSpacing[2]))
    corner = np.asarray([clip_y, clip_x, 0])
    trans_corner = sitk.TranslationTransform(3, corner.astype(np.float64))

    # computeScale(input_mat=frame_mat)

    # tfm2us = sitk.Transform(mat2tfm(np.identity(4)))
    tfm2us = sitk.Transform(mat2tfm(input_mat=frame_mat))
    tfm2us.AddTransform(trans_corner)
    print(tfm2us)

    """ US volume resampler, with final_transform"""
    resampler_us = sitk.ResampleImageFilter()
    resampler_us.SetReferenceImage(destVol)
    resampler_us.SetInterpolator(sitk.sitkLinear)
    resampler_us.SetDefaultPixelValue(0)
    resampler_us.SetTransform(tfm2us)
    outUSImg = resampler_us.Execute(us_img)
    outUSNp = sitk.GetArrayFromImage(outUSImg[:, :, 0])
    print('outUSNp shape {}'.format(outUSNp.shape))

    resampler_slice = sitk.ResampleImageFilter()
    resampler_slice.SetReferenceImage(destVol)
    resampler_slice.SetInterpolator(sitk.sitkLinear)
    resampler_slice.SetDefaultPixelValue(0)
    resampler_slice.SetTransform(trans_corner)
    outFrameImg = resampler_slice.Execute(sitk.GetImageFromArray(np.expand_dims(fixed_origin, axis=0)))
    # outFrameImg = resampler_slice.Execute(fixed_image)
    outFrameNp = sitk.GetArrayFromImage(outFrameImg[:, :, 0])
    print('fixed_origin shape {}'.format(outFrameNp.shape))

    frame_resample_concate = np.concatenate((outFrameNp, outUSNp), axis=0)
    cv2.imwrite('tmp.jpg', frame_resample_concate)

def samplePlane2(vol_path, trans_mats, frame_id):
    us_path = vol_path
    us_img = sitk.ReadImage(us_path)
    us_np = sitk.GetArrayFromImage(us_img)
    print(us_img.GetOrigin())
    print('us_np shape {}'.format(us_np.shape))
    print('us_img size {}'.format(us_img.GetSize()))
    # fixed_path = path.join(usrecon_dataset, '{}/frames/{:04}.jpg'.format(case_id, frame_id))
    # fixed_origin = cv2.imread(fixed_path, 0)

    clip_x, clip_y, clip_h, clip_w = 105, 54, 320, 565
    # fixed_np = fixed_origin[clip_x:clip_x+clip_h, clip_y:clip_y+clip_w]
    # fixed_np = fixed_origin[105:105+320, 54:54+565]

    # spacing = 0.4   # For my Slicer reconstructed volume
    # spacing = 0.35  # For uronac reconstructed volume
    mat_scales = computeScale(input_mat=trans_mats[frame_id, :, :])
    spacing = np.mean(mat_scales[:2]) / us_img.GetSpacing()[0]
    # print('frame_scale = {}'.format(spacing))
    # frame_w = int(spacing * fixed_np.shape[1])
    # frame_h = int(spacing * fixed_np.shape[0])
    # fixed_np = cv2.resize(fixed_np, (frame_w, frame_h))
    # fixed_np = fixed_np.astype(np.float64)
    # fixed_np = np.expand_dims(fixed_np, axis=0)
    # print('fixed_np shape {}'.format(fixed_np.shape))

    # fixed_image = sitk.GetImageFromArray(fixed_np)
    # fixed_image.SetSpacing(us_img.GetSpacing())

    frame_mat = trans_mats[frame_id, :, :]
    # print('us_img {}'.format(us_img))
    # print('frame_mat\n{}'.format(frame_mat))


    # tfm2us = sitk.Transform(mat2tfm(np.identity(4)))
    # affine_tfm = sitk.AffineTransform(3)
    # affine_tfm.SetMatrix(frame_mat[:3, :3].flatten())
    # affine_tfm.SetTranslation(frame_mat[:3, 3])
    # print(affine_tfm)

    # spacing1 = us_img.GetSpacing()[0]
    # print('spacing1 {}, spacing {}'.format(spacing1, spacing))
    # width, length = fixed_origin.shape[1], fixed_origin.shape[0]
    destVol = sitk.Image(int(clip_w*spacing), int(clip_h*spacing), 1, sitk.sitkUInt8)
    destSpacing = np.asarray([spacing, spacing, spacing])
    destVol.SetSpacing((1/destSpacing[0], 1/destSpacing[1], 1/destSpacing[2]))
    corner = np.asarray([clip_y, clip_x, 0])
    trans_corner = sitk.TranslationTransform(3, corner.astype(np.float64))

    # computeScale(input_mat=frame_mat)

    # tfm2us = sitk.Transform(mat2tfm(np.identity(4)))
    tfm2us = sitk.Transform(mat2tfm(input_mat=frame_mat))
    tfm2us.AddTransform(trans_corner)
    print(tfm2us)

    """ US volume resampler, with final_transform"""
    resampler_us = sitk.ResampleImageFilter()
    resampler_us.SetReferenceImage(destVol)
    resampler_us.SetInterpolator(sitk.sitkLinear)
    resampler_us.SetDefaultPixelValue(0)
    resampler_us.SetTransform(tfm2us)
    outUSImg = resampler_us.Execute(us_img)
    outUSNp = sitk.GetArrayFromImage(outUSImg[:, :, 0])
    print('outUSNp shape {}'.format(outUSNp.shape))

    resampler_slice = sitk.ResampleImageFilter()
    resampler_slice.SetReferenceImage(destVol)
    resampler_slice.SetInterpolator(sitk.sitkLinear)
    resampler_slice.SetDefaultPixelValue(0)
    resampler_slice.SetTransform(trans_corner)
    outFrameImg = resampler_slice.Execute(sitk.GetImageFromArray(np.expand_dims(fixed_origin, axis=0)))
    # outFrameImg = resampler_slice.Execute(fixed_image)
    outFrameNp = sitk.GetArrayFromImage(outFrameImg[:, :, 0])
    print('fixed_origin shape {}'.format(outFrameNp.shape))

    frame_resample_concate = np.concatenate((outFrameNp, outUSNp), axis=0)
    cv2.imwrite('tmp.jpg', frame_resample_concate)

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

def samplePlaneFromVol(case_id, trans_mats, frame_id):
    us_path = path.join(myvol_dataset, '{}/{}_myrecon.mhd'.format(case_id, case_id))
    us_img = sitk.ReadImage(us_path)
    us_np = sitk.GetArrayFromImage(us_img)
    print(us_img.GetOrigin())
    print('us_np shape {}'.format(us_np.shape))
    print('us_img size {}'.format(us_img.GetSize()))
    fixed_path = path.join(usrecon_dataset, '{}/frames/{:04}.jpg'.format(case_id, frame_id))
    fixed_origin = cv2.imread(fixed_path, 0)

    clip_x, clip_y, clip_h, clip_w = 105, 54, 320, 565
    fixed_np = fixed_origin[clip_x:clip_x+clip_h, clip_y:clip_y+clip_w]
    # fixed_np = fixed_origin[105:105+320, 54:54+565]

    # spacing = 0.4   # For my Slicer reconstructed volume
    # spacing = 0.35  # For uronac reconstructed volume
    mat_scales = computeScale(input_mat=trans_mats[frame_id, :, :])
    spacing = np.mean(mat_scales[:2]) / us_img.GetSpacing()[0]
    print('frame_scale = {}'.format(spacing))
    frame_w = int(spacing * fixed_np.shape[1])
    frame_h = int(spacing * fixed_np.shape[0])
    fixed_np = cv2.resize(fixed_np, (frame_w, frame_h))
    fixed_np = fixed_np.astype(np.float64)
    fixed_np = np.expand_dims(fixed_np, axis=0)
    print('fixed_np shape {}'.format(fixed_np.shape))

    fixed_image = sitk.GetImageFromArray(fixed_np)
    # fixed_image.SetSpacing(us_img.GetSpacing())

    frame_mat = trans_mats[frame_id, :, :]
    # print('us_img {}'.format(us_img))
    # print('frame_mat\n{}'.format(frame_mat))


    # tfm2us = sitk.Transform(mat2tfm(np.identity(4)))
    # affine_tfm = sitk.AffineTransform(3)
    # affine_tfm.SetMatrix(frame_mat[:3, :3].flatten())
    # affine_tfm.SetTranslation(frame_mat[:3, 3])
    # print(affine_tfm)

    # spacing1 = us_img.GetSpacing()[0]
    # print('spacing1 {}, spacing {}'.format(spacing1, spacing))
    # width, length = fixed_origin.shape[1], fixed_origin.shape[0]
    destVol = sitk.Image(int(clip_w*spacing), int(clip_h*spacing), 1, sitk.sitkUInt8)
    destSpacing = np.asarray([spacing, spacing, spacing])
    destVol.SetSpacing((1/destSpacing[0], 1/destSpacing[1], 1/destSpacing[2]))
    corner = np.asarray([clip_y, clip_x, 0])
    trans_corner = sitk.TranslationTransform(3, corner.astype(np.float64))

    # computeScale(input_mat=frame_mat)

    # tfm2us = sitk.Transform(mat2tfm(np.identity(4)))
    tfm2us = sitk.Transform(mat2tfm(input_mat=frame_mat))
    tfm2us.AddTransform(trans_corner)
    print(tfm2us)

    """ US volume resampler, with final_transform"""
    resampler_us = sitk.ResampleImageFilter()
    resampler_us.SetReferenceImage(destVol)
    resampler_us.SetInterpolator(sitk.sitkLinear)
    resampler_us.SetDefaultPixelValue(0)
    resampler_us.SetTransform(tfm2us)
    outUSImg = resampler_us.Execute(us_img)
    outUSNp = sitk.GetArrayFromImage(outUSImg[:, :, 0])
    print('outUSNp shape {}'.format(outUSNp.shape))

    resampler_slice = sitk.ResampleImageFilter()
    resampler_slice.SetReferenceImage(destVol)
    resampler_slice.SetInterpolator(sitk.sitkLinear)
    resampler_slice.SetDefaultPixelValue(0)
    resampler_slice.SetTransform(trans_corner)
    outFrameImg = resampler_slice.Execute(sitk.GetImageFromArray(np.expand_dims(fixed_origin, axis=0)))
    # outFrameImg = resampler_slice.Execute(fixed_image)
    outFrameNp = sitk.GetArrayFromImage(outFrameImg[:, :, 0])
    print('fixed_origin shape {}'.format(outFrameNp.shape))

    frame_resample_concate = np.concatenate((outFrameNp, outUSNp), axis=0)
    cv2.imwrite('tmp.jpg', frame_resample_concate)

def cell_images():
    set_path = '/home/guoh9/tmp/cells/full_frames'
    case_id_list = os.listdir(set_path)
    print(os.listdir(set_path))

    for i in range(1, 33):
        case_id = 'XY{:02}_video'.format(i)
        frame0_path = path.join(set_path, case_id, 'frame0.jpg')
        print(frame0_path)
        frame0 = cv2.imread(frame0_path, 0)
        target_path = path.join(set_path, 'collections/{}.jpg'.format(case_id))
        cv2.imwrite(target_path, frame0)
        print('{} frame0 saved'.format(case_id))

def myAffineGrid(input_tensor, input_mat, input_spacing=[1, 1, 1]):
    input_spacing = np.asarray(input_spacing)
    image_size = np.asarray([input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]])
    image_phy_size = (image_size - 1) * input_spacing
    # image_phy_size = [input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]]
    grid_size = input_tensor.shape
    t_mat = input_mat
    image_tensor = input_tensor

    # generate grid of input image (i.e., the coordinate of the each pixel in the input image. The center point of the input image volume is assigned as (0, 0, 0).)
    grid_x_1d = torch.linspace(-0.5 * image_phy_size[0], 0.5 * image_phy_size[0], steps=grid_size[4])
    grid_y_1d = torch.linspace(-0.5 * image_phy_size[1], 0.5 * image_phy_size[1], steps=grid_size[3])
    grid_z_1d = torch.linspace(-0.5 * image_phy_size[2], 0.5 * image_phy_size[2], steps=grid_size[2])
    grid_z, grid_y, grid_x = torch.meshgrid(grid_z_1d, grid_y_1d, grid_x_1d)
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)
    grid_z = grid_z.unsqueeze(0)
    origin_grid = torch.cat([grid_x, grid_y, grid_z, torch.ones_like(grid_x)], dim=0)
    origin_grid = origin_grid.view(4, -1)

    # compute the rasample grid through matrix multiplication
    print('t_mat {}, origin_grid {}'.format(t_mat.shape, origin_grid.shape))
    print('img_tensor type {}'.format(image_tensor.type()))
    t_mat = torch.tensor(t_mat)
    t_mat = t_mat.float()
    # origin_grid = origin_grid.unsqueeze(0)
    print('t_mat shape {}'.format(t_mat.shape))
    print('origin_grid shape {}'.format(origin_grid.shape))
    resample_grid = torch.matmul(t_mat, origin_grid)[0:3, :]

    # convert the resample grid coordinate from physical coordinate system to a range of [-1, 1] (which is required by the PyTorch interface 'grid_sample'). 
    resample_grid[0, :] = (resample_grid[0, :] + 0.5 * image_phy_size[0]) / image_phy_size[0] * 2 - 1
    resample_grid[1, :] = (resample_grid[1, :] + 0.5 * image_phy_size[1]) / image_phy_size[1] * 2 - 1
    resample_grid[2, :] = (resample_grid[2, :] + 0.5 * image_phy_size[2]) / image_phy_size[2] * 2 - 1
    print('before {}'.format(resample_grid.shape))
    resample_grid = resample_grid.permute(1,0)
    print('after {}'.format(resample_grid.shape))
    resample_grid = resample_grid.contiguous()
    print('after2 {}'.format(resample_grid.shape))
    resample_grid = resample_grid.reshape(grid_size[2], grid_size[3], grid_size[4], 3)
    resample_grid = resample_grid.unsqueeze(0)
    print('resample_grid {}'.format(resample_grid.shape))
    # sys.exit()
    return resample_grid.double()

def myAffineGrid2(input_tensor, input_mat, input_spacing=[1, 1, 1], device=None):
    # print('input_tensor shape {}'.format(input_tensor.shape))
    # print('input_mat shape {}'.format(input_mat.shape))
    # sys.exit()

    input_spacing = np.asarray(input_spacing)
    image_size = np.asarray([input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]])
    image_phy_size = (image_size - 1) * input_spacing
    # image_phy_size = [input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]]
    grid_size = input_tensor.shape

    # generate grid of input image (i.e., the coordinate of the each pixel in the input image. The center point of the input image volume is assigned as (0, 0, 0).)
    grid_x_1d = torch.linspace(-0.5 * image_phy_size[0], 0.5 * image_phy_size[0], steps=grid_size[4])
    grid_y_1d = torch.linspace(-0.5 * image_phy_size[1], 0.5 * image_phy_size[1], steps=grid_size[3])
    grid_z_1d = torch.linspace(-0.5 * image_phy_size[2], 0.5 * image_phy_size[2], steps=grid_size[2])
    grid_z, grid_y, grid_x = torch.meshgrid(grid_z_1d, grid_y_1d, grid_x_1d)
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)
    grid_z = grid_z.unsqueeze(0)
    origin_grid = torch.cat([grid_x, grid_y, grid_z, torch.ones_like(grid_x)], dim=0)
    origin_grid = origin_grid.view(4, -1)
    if device:
        origin_grid = origin_grid.to(device)
        origin_grid.requires_grad = True

    # compute the rasample grid through matrix multiplication
    # print('t_mat {}, origin_grid {}'.format(t_mat.shape, origin_grid.shape))
    # t_mat = input_mat
    # t_mat = torch.tensor(t_mat)
    # t_mat = t_mat.float()
    # t_mat.requires_grad = True

    # t_mat = t_mat.squeeze()
    # origin_grid = origin_grid.unsqueeze(0)
    # print('t_mat shape {}'.format(t_mat.shape))
    # print('origin_grid shape {}'.format(origin_grid.shape))
    # resample_grid = torch.matmul(t_mat, origin_grid)[0:3, :]
    resample_grid = torch.matmul(input_mat, origin_grid)[:, 0:3, :]
    # print('resample_grid {}'.format(resample_grid.shape))

    # convert the resample grid coordinate from physical coordinate system to a range of [-1, 1] (which is required by the PyTorch interface 'grid_sample'). 
    resample_grid[:, 0, :] = (resample_grid[:, 0, :] + 0.5 * image_phy_size[0]) / image_phy_size[0] * 2 - 1
    resample_grid[:, 1, :] = (resample_grid[:, 1, :] + 0.5 * image_phy_size[1]) / image_phy_size[1] * 2 - 1
    resample_grid[:, 2, :] = (resample_grid[:, 2, :] + 0.5 * image_phy_size[2]) / image_phy_size[2] * 2 - 1
    # print('resample_grid2 {}'.format(resample_grid.shape))
    resample_grid = resample_grid.permute(0,2,1).contiguous()
    resample_grid = resample_grid.reshape(grid_size[0], grid_size[2], grid_size[3], grid_size[4], 3)
    # resample_grid = resample_grid.unsqueeze(1)
    # print('resample_grid {}'.format(resample_grid.shape))
    # sys.exit()
    return resample_grid

def processFrame(us_spacing, frame_np, frame_mat, clip_info):
    """Crop the frame with reconstruction ROI, respacing to the same as US volume

    Args:
        us_spacing (tuple): sitk_img.GetSpacing()
        frame_np (np array): Raw 1-channel grey image from frame
        frame_mat ([np array]): 4x4 matrix of this frame, read from sequence mhd file

    Returns:
        fixed_np: cropped and resize frame ROI
    """
    # print('us_spacing {}'.format(us_spacing))
    # print('frame_np {}'.format(frame_np))
    # print('frame_mat {}'.format(frame_mat))
    # print('clip_info {}'.format(clip_info))
    # sys.exit()
    clip_x, clip_y, clip_h, clip_w = clip_info
    # print('clip_info {}'.format(clip_info))
    fixed_np = frame_np[clip_x:clip_x+clip_h, clip_y:clip_y+clip_w]
    mat_scales = computeScale(input_mat=frame_mat)
    # print('matscales {}'.format(mat_scales))
    spacing = np.mean(mat_scales[:2]) / us_spacing[0]
    # print("******spacing {}".format(spacing))
    frame_w = int(spacing * fixed_np.shape[1])
    frame_h = int(spacing * fixed_np.shape[0])
    fixed_np = cv2.resize(fixed_np, (frame_w, frame_h))
    fixed_np = fixed_np.astype(np.float64)
    return fixed_np

def reformatFrame(us_spacing, frame_crop, frame_mat, clip_info):
    mat_scales = computeScale(input_mat=frame_mat)
    # print('matscales {}'.format(mat_scales))
    spacing = np.mean(mat_scales[:2]) / 0.36
    print('{} {}'.format(frame_crop.shape, spacing))
    frame_w = int(frame_crop.shape[1]/spacing)
    frame_h = int(frame_crop.shape[0]/spacing)
    frame_crop = cv2.resize(frame_crop, (frame_w, frame_h))
    frame_crop = frame_crop.astype(np.float64)

    print('{} {}'.format(frame_crop.shape, spacing))
    return frame_crop



def mat2dof_np(input_mat):
    # print('input_mat\n{}'.format(input_mat))
    mat_copy = copy.copy(input_mat)
    translations = mat_copy[:3, 3]
    rotations_eulers = np.asarray(tfms.euler_from_matrix(mat_copy))
    rotations_degrees = (rotations_eulers / (2 * math.pi)) * 360
    scales = computeScale(input_mat=mat_copy)

    dof = np.concatenate((translations, rotations_degrees, scales), axis=0)

    # print('dof\n{}\n'.format(dof))
    # sys.exit()
    return dof

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

def matSitk2Stn(input_mat, clip_size, raw_spacing, frame_shape,
                img_size, img_spacing, img_origin):
    frame_gt_mat = input_mat
    clip_x, clip_y = clip_size
    corner = np.asarray([clip_y, clip_x, 0])
    
    pos_spacing = np.mean(computeScale(input_mat=frame_gt_mat))
    spacing_mat = np.diag([1/pos_spacing, 1/pos_spacing, 1/pos_spacing, 1])
    trans_mat = np.identity(4)
    trans_mat[:3, 3] = corner
    frame_gt_mat[:3, 3] -= img_origin
    frame_gt_mat = np.dot(frame_gt_mat, trans_mat)
    frame_gt_mat = np.dot(frame_gt_mat, spacing_mat)
    frame_gt_mat[:3, 3] *= [img_spacing[0]/raw_spacing[0],
                            img_spacing[1]/raw_spacing[1], 
                            img_spacing[2]/raw_spacing[2]]

    """ origin_translate makes the volume center at coordinate center """
    origin_translate = np.identity(4)
    origin_translate[:3, 3] = -0.5 * np.asarray(img_size) * np.asarray(img_spacing)

    """ dest_translate makes the resultant sampling plane at the coordinate center"""
    dest_translate = np.identity(4)
    dest_translate[:3, 3] = np.asarray([frame_shape[1]/2, frame_shape[0]/2,0])

    frame_gt_mat = np.dot(origin_translate, frame_gt_mat)
    frame_gt_mat = np.dot(frame_gt_mat, dest_translate)
    return frame_gt_mat

def volContainer(input_tensor, container_size=(292, 158, 229)):
    # print('input_tensor shape {}'.format(input_tensor.shape))
    input_shape = list(input_tensor.shape)
    input_tensor_compact = torch.squeeze(input_tensor)
    vol_d, vol_h, vol_w = input_tensor_compact.shape
    con_d, con_h, con_w = container_size
    d_start = int((con_d-vol_d)/2)
    h_start = int((con_h-vol_h)/2)
    w_start = int((con_w-vol_w)/2)
    # print('vol_d {}, vol_h {}, vol_w {}'.format(vol_d, vol_h, vol_w))
    # print('d_start {}, h_start {}, w_start {}'.format(d_start, h_start, w_start))
    output_shape = [con_d, con_h, con_w]
    output_tensor = torch.zeros(output_shape)
    output_tensor[d_start:d_start+vol_d, h_start:h_start+vol_h, w_start:w_start+vol_w] = input_tensor_compact
    for i in range(len(input_shape)-3):
        output_tensor = output_tensor.unsqueeze(0)
    # print('output tensor shape {}'.format(output_tensor.shape))
    return output_tensor
    # sys.exit()

def frameContainer(input_tensor, container_size=(292, 158, 229), start=(0, 0)):
    # print('input_tensor shape {}'.format(input_tensor.shape))
    input_shape = list(input_tensor.shape)
    input_tensor_compact = torch.squeeze(input_tensor)
    frame_h, frame_w = input_tensor_compact.shape
    con_d, con_h, con_w = container_size
    # print('frame_h {}, frame_w {}'.format(frame_h, frame_w))
    # print('con_h {}, con_w {}'.format(con_h, con_w))
    h_start, w_start = start
    # print('vol_d {}, vol_h {}, vol_w {}'.format(vol_d, vol_h, vol_w))
    # print('h_start {}, w_start {}'.format(h_start, w_start))
    output_shape = [con_h, con_w]
    output_tensor = torch.zeros(output_shape)
    output_tensor[h_start:h_start+frame_h, w_start:w_start+frame_w] = input_tensor_compact
    for i in range(len(input_shape)-3):
        output_tensor = output_tensor.unsqueeze(0)
    # print('output tensor shape {}'.format(output_tensor.shape))
    return output_tensor

def frameCrop(input_np, crop_size=(128, 128)):
    input_h, input_w = input_np.shape
    crop_h, crop_w = crop_size
    max_h = max(input_h, crop_h)
    max_w = max(input_w, crop_w)

    if crop_h > input_h or crop_w > input_w:
        container = np.zeros((max_h, max_w))
        con_start_h = int((max_h - input_h)/2)
        con_start_w = int((max_w - input_w)/2)
        container[con_start_h:con_start_h+input_h, con_start_w:con_start_w+input_w] = input_np
        input_np = container

    start_h = int((input_np.shape[0] - crop_h)/2)
    start_w = int((input_np.shape[1] - crop_w)/2)
    output_np = input_np[start_h:start_h+crop_h, start_w:start_w+crop_w]
    return output_np

def chooseRandInit(frame_num, frame_id, rand_range=20):
    """Choose a random slice in a range [-20, 20], for subvolume initialization

    Args:
        frame_num ([int]): total number of frame
        frame_id ([int]): current frame id
        rand_range (int, optional): Range of initialization. Defaults to 20.

    Returns:
        [int]: initialization frame id
    """
    # print('num {}, id {}'.format(frame_num, frame_id))
    upper = frame_id + rand_range
    lower = frame_id - rand_range
    upper = min(upper, frame_num-1)
    lower = max(lower, 0)
    rand_id = random.randint(lower, upper)
    # print('upper {}, lower {}'.format(upper, lower))
    # print('rand_id {}'.format(rand_id))
    return rand_id

def sampleSubvol(sitk_img, init_mat, crop_size):
    # print('sitk_img origin {}'.format(sitk_img.GetOrigin()))

    source_img = sitk_img
    init_tfm = mat2tfm(input_mat=init_mat)
    # destVol = sitk.Image(sitk_img.GetSize()[0], sitk_img.GetSize()[1], 1, sitk.sitkUInt8)
    destVol = sitk.Image(crop_size[0], crop_size[1], crop_size[2], sitk.sitkUInt8)
    destSpacing = np.asarray(sitk_img.GetSpacing())
    destVol.SetSpacing((destSpacing[0], destSpacing[1], destSpacing[2]))

    destVol.SetOrigin(-0.5*np.asarray(destVol.GetSize())
                      *np.asarray(destVol.GetSpacing()))
    source_img.SetOrigin(-0.5*np.asarray(source_img.GetSize())
                                *np.asarray(source_img.GetSpacing()))
    # print('source_img origin {}'.format(source_img.GetOrigin()))
    # print('destVol origin {}'.format(destVol.GetOrigin()))
    """ US volume resampler, with frame position groundtruth """
    resampler_us = sitk.ResampleImageFilter()
    resampler_us.SetReferenceImage(destVol)
    resampler_us.SetInterpolator(sitk.sitkLinear)
    resampler_us.SetDefaultPixelValue(0)
    resampler_us.SetTransform(init_tfm)
    outUSImg = resampler_us.Execute(source_img)
    outUSNp = sitk.GetArrayFromImage(outUSImg)
    # print('outUSNp {}'.format(outUSNp.shape))
    # cv2.imwrite('tmp_sitk.jpg', outUSNp[32, :, :])
    # sys.exit()
    return outUSNp

def sampleSubvol2(sitk_img, init_mat, crop_size):
    half_size = 500
    xyhw=[105, 54, 320, 565]
    fan_center = (int(xyhw[0]+xyhw[2]//2), int(xyhw[1]+xyhw[3]//2))

    destSpacing = np.asarray([1., 1., 1.])
    destOrigin = np.asarray([fan_center[1]-half_size*destSpacing[0], 
                                fan_center[0]-half_size*destSpacing[0], 0]).astype(np.float)
    destVol = sitk.Image(int(half_size)*2, int(half_size)*2, 1, sitk.sitkUInt8)
    destVol.SetSpacing(np.asarray([1., 1., 1.]))
    destVol.SetOrigin(destOrigin)

    tfm = sitk.CompositeTransform(mat2tfm(np.identity(4)))
    tfm_slice = mat2tfm(init_mat.astype(np.float))
    tfm.AddTransform(tfm_slice)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(destVol)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tfm)

    us_resample = sitk.GetArrayFromImage(resampler.Execute(sitk_img))[0, :, :]
    return us_resample


def array_normalize(input_array):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    # print('max {}, min {}'.format(max_value, min_value))
    k = 255 / (max_value - min_value)
    min_array = np.ones_like(input_array) * min_value
    normalized = k * (input_array - min_array)
    return normalized

def dof2mat_tensor(input_dof, device):
    rad = tgm.deg2rad(input_dof[:, 3:])

    ai = rad[:, 0]
    aj = rad[:, 1]
    ak = rad[:, 2]

    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = torch.zeros((input_dof.shape[0], 4, 4))

    if device:
        M = M.to(device)
        M.requires_grad = True

    M[:, 0, 0] = cj*ck
    M[:, 0, 1] = sj*sc-cs
    M[:, 0, 2] = sj*cc+ss
    M[:, 1, 0] = cj*sk
    M[:, 1, 1] = sj*ss+cc
    M[:, 1, 2] = sj*cs-sc
    M[:, 2, 0] = -sj
    M[:, 2, 1] = cj*si
    M[:, 2, 2] = cj*ci
    M[:, :3, 3] = input_dof[:, :3]

    M[:, 3, 3] = 1

    # print('out_mat {}\n{}'.format(M.shape, M))
    # sys.exit()
    return M

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

def correlation_coefficient(T1, T2):
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result

def generateRandomGuess(means, stds):
    random_dof = []

    for i in range(means.shape[0]):
        this_mean, this_std = means[i], stds[i]
        rand_dof = np.random.normal(this_mean, this_std, 1)[0]
        # print('mean {:.4f}, std {:.4f}, rand {:.4f}'.format(this_mean, this_std, rand_dof))
        random_dof.append(rand_dof)
    # print(random_dof)
    # sys.exit()
    return np.asarray(random_dof)

def fuseImgSeg(input_img, input_seg, alpha=0.5):
    input_img = np.squeeze(input_img).astype(np.float32)
    input_seg = np.squeeze(input_seg).astype(np.float32)

    input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    
    seg_foreground = input_seg > 0.5
    seg_background = input_seg <= 0.5

    seg_slice = np.zeros_like(input_seg)
    seg_slice[seg_foreground] = 255.0
    
    seg_bgr = np.zeros_like(input_img)
    seg_bgr[:, :, 2] = seg_slice

    output_fused = np.zeros_like(input_img)
    output_fused[seg_foreground] = alpha * seg_bgr[seg_foreground] + (1 - alpha) * input_img[seg_foreground]
    output_fused[seg_background] = input_img[seg_background]

    return output_fused

def reshapeSeg(input_img, corner, crop_size, resize):
    input_img = np.squeeze(input_img)
    # print('input_img shape {}'.format(input_img.shape))
    # print('corner {}'.format(corner))
    # print('crop_size {}'.format(crop_size))
    # print('resize {}'.format(resize))

    corner = corner.astype(np.int)
    crop_size = np.asarray(crop_size).astype(np.int)
    print(corner)

    cropped_img = input_img[corner[1]:corner[1]+crop_size[1],
                            corner[0]:corner[0]+crop_size[0]]
    resized_img = cv2.resize(cropped_img, (resize[0], resize[1]))
    # print('resize {}'.format(resized_img.shape))
    # cv2.imwrite('tmp_crop.jpg', resized_img)
    # sys.exit()



    return cleanSeg(resized_img)

def cleanSeg(input_seg):
    input_seg = np.squeeze(input_seg).astype(np.float32)
    seg_foreground = input_seg > 0.5
    seg_background = input_seg <= 0.5

    seg_slice = np.zeros_like(input_seg)
    seg_slice[seg_foreground] = 255.0

    return seg_slice


def mat2tfm_euler(input_mat):
    tfm = sitk.Euler3DTransform()
    # tfm = sitk.Similarity3DTransform()
    tfm.SetMatrix(np.reshape(input_mat[:3, :3], (9,)))
    translation = input_mat[:3,3]
    tfm.SetTranslation(translation)
    # tfm.SetCenter([0, 0, 0])
    return tfm


def read_list(path):
    mylist = []
    with open(path, 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            mylist.append(currentPlace)
    return mylist

def write_list(input_list, path):
    with open(path, 'w') as filehandle:
        for item in input_list:
            filehandle.write(str(item)+'\n')
    return 0

# def drawEdge(img, edge, color='red'):
#     b = np.repeat(a[:, :, np.newaxis], 3, axis=2)

#     return 0
# mats = readMatsFromSequence(case_id='Case0005')
# samplePlane(case_id='Case0005', trans_mats=mats, frame_id=43)
# print('mats shape {}'.format(mats.shape))

# volCompare(case_id='Case0009')

def mat2tfm(input_mat):
    tfm = sitk.AffineTransform(3)
    tfm.SetMatrix(np.reshape(input_mat[:3, :3], (9,)))
    translation = input_mat[:3,3]
    tfm.SetTranslation(translation)
    # tfm.SetCenter([0, 0, 0])
    return tfm


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

def mats2pts_correction(trans_mats, input_img=np.zeros((480, 640)),
                        cb_pts=None, ft_pts=None, correction=False):
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

    corner_pts = np.concatenate((corner_pts, np.ones((corner_pts.shape[0], 1))), axis=1)
    corner_pts = np.transpose(corner_pts)

    transformed_pts = []
    transform_mats = []
    original_mats = []
    for frame_id in range(trans_mats.shape[0]):
        trans_mat = trans_mats[frame_id, :, :]

        original_mats.append(trans_mat)
        # print('corner_pts {}'.format(corner_pts.shape))
        # print('corner_pts\n{}'.format(corner_pts))
        # print('trans_mat {}'.format(trans_mat.shape))
        # sys.exit()
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
    return transformed_pts

def dice(pred, true, k = 1):
    # pred[pred>0] = 255
    # true[true>0] = 255
    if np.sum(pred) == 0 and np.sum(true) == 0:
        return -1.0
    elif np.sum(pred) == 0 and np.sum(true) == 0:
        return 0.0

    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))

    if np.sum(pred) == np.sum(true) == 0:
        dice = 1.0
    return dice

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

def frame_err(gt_mat, pd_mat, spacing):
    # print('gt_mat {}, pd_mat {}'.format(gt_mat.shape, pd_mat.shape))
    # print('spacing {}'.format(spacing))

    gt_mat_in = copy.copy(gt_mat)
    pd_mat_in = copy.copy(pd_mat)

    gt_mat_in = np.expand_dims(gt_mat_in, axis=0)
    pd_mat_in = np.expand_dims(pd_mat_in, axis=0)

    gt_pts = mats2pts_correction(gt_mat_in)
    pd_pts = mats2pts_correction(pd_mat_in)

    dist = evaluate_dist(gt_pts, pd_pts, resolution=spacing[0])

    return dist

def img_cor(gt_img, pd_img):
    img1 = copy.copy(gt_img)
    img2 = copy.copy(pd_img)

    # img1 = data_transform(img1, masked_full=False)
    # img2 = data_transform(img2, masked_full=False)

    if np.sum(img1)==0 and np.sum(img2)==0:
        return 1.
    elif np.sum(img1)==0 or np.sum(img2)==0:
        return 0.
    
    return np.corrcoef(gt_img.flat, pd_img.flat)[0, 1]

    # sys.exit()
    # return 0

def params_to_mat44(trans_params_in, cam_cali_mat):
    """
    Transform the parameters in Aurora files into 4 x 4 matrix
    :param trans_params: transformation parameters in Aurora.pos. Only the last 7 are useful
    3 are translations, 4 are the quaternion (x, y, z, w) for rotation
    :return: 4 x 4 transformation matrix
    """
    trans_params = copy.copy(trans_params_in)
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

def get_next_pos(trans_params1, dof, cam_cali_mat):
    """
    Given the first frame's Aurora position line and relative 6dof, return second frame's position line
    :param trans_params1: Aurora position line of the first frame
    :param dof: 6 degrees of freedom based on the first frame, rotations should be degrees
    :param cam_cali_mat: Camera calibration matrix of this case
    :return: Aurora position line of the second frame
    """
    trans_mat1 = params_to_mat44(trans_params1, cam_cali_mat=cam_cali_mat)
    # print('trans_mat1\n{}'.format(trans_mat1))
    dof_in = copy.copy(dof)
    """ Transfer degrees to euler """
    dof_in[3:] = dof_in[3:] * (2 * math.pi) / 360

    rot_mat = tfms.euler_matrix(dof_in[5], dof_in[4], dof_in[3], 'rzyx')[:3, :3]

    relative_mat = np.identity(4)
    relative_mat[:3, :3] = rot_mat
    relative_mat[:3, 3] = dof_in[:3]

    next_mat = np.dot(inv(cam_cali_mat), inv(np.dot(relative_mat, trans_mat1)))
    quaternions = tfms.quaternion_from_matrix(next_mat)  # wxyz

    next_params = np.zeros(7)
    next_params[:3] = next_mat[:3, 3]
    next_params[3:6] = quaternions[1:]
    next_params[6] = quaternions[0]
    return next_params

def get_next_pos2(mat1, dof, cam_cali_mat):
    """ Transfer degrees to euler """
    dof_in = copy.copy(dof)
    dof_in[3:] = dof_in[3:] * (2 * math.pi) / 360

    rot_mat = tfms.euler_matrix(dof_in[5], dof_in[4], dof_in[3], 'rzyx')[:3, :3]

    relative_mat = np.identity(4)
    relative_mat[:3, :3] = rot_mat
    relative_mat[:3, 3] = dof_in[:3]

    next_mat = np.dot(inv(cam_cali_mat), inv(np.dot(relative_mat, mat1)))
    quaternions = tfms.quaternion_from_matrix(next_mat)  # wxyz

    next_params = np.zeros(7)
    next_params[:3] = next_mat[:3, 3]
    next_params[3:6] = quaternions[1:]
    next_params[6] = quaternions[0]
    return next_params

def mat2params(mat):
    quaternions = tfms.quaternion_from_matrix(mat)  # wxyz

    params = np.zeros(7)
    params[:3] = mat[:3, 3]
    params[3:6] = quaternions[1:]
    params[6] = quaternions[0]
    return params

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
    dof = copy.copy(input_dof)
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
    # print("r_mat44\n{}".format(r_mat44))
    # print("t_mat44\n{}".format(t_mat44))
    # print("tr_dot\n{}".format(tr_dot))
    # sys.exit()

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



def format_vid_frame(us_frame, canvas_size=500):
    xyhw=[105, 54, 320, 565]

    canvas = np.zeros((canvas_size*2, canvas_size*2))

    fan_center = (xyhw[0]+xyhw[2]//2, xyhw[1]+xyhw[3]//2)
    move = (canvas_size-fan_center[0], canvas_size-fan_center[1])

    h, w = us_frame.shape

    canvas[move[0]:move[0]+h, move[1]:move[1]+w] = us_frame

    return canvas.astype(np.uint8)

def canvas_to_frame(us_canvas, canvas_size=500, xyhw=[105, 54, 320, 565]):
    """Crop the 480*640 us frame from the canvas, useful to slice frame
    from the US volumes

    Args:
        us_canvas (np): sliced canvas from tools.resample_slice
        canvas_size (int, optional): the original size of that canvas. Defaults to 500.
        xyhw (list, optional): information about the fan area. Defaults to [105, 54, 320, 565].

    Returns:
        np: cropped frame, usually 480*640 size
    """
    h, w = 480, 640
    fan_center = (xyhw[0]+xyhw[2]//2, xyhw[1]+xyhw[3]//2)
    move = (canvas_size-fan_center[0], canvas_size-fan_center[1])

    frame_crop = us_canvas[move[0]:move[0]+h, move[1]:move[1]+w]

    return frame_crop

def get_next_mat(base_mat, this_dof):
    """ Transfer degrees to euler """
    dof = copy.copy(this_dof)
    dof[3:] = dof[3:] * (2 * math.pi) / 360

    rot_mat = tfms.euler_matrix(dof[5], dof[4], dof[3], 'rzyx')[:3, :3]

    relative_mat = np.identity(4)
    relative_mat[:3, :3] = rot_mat
    relative_mat[:3, 3] = dof[:3]

    next_mat = np.dot(relative_mat, base_mat)
    # next_mat = np.dot(base_mat, relative_mat)
    return next_mat

def get_6dof_from_mats(trans_mat1, trans_mat2):
    relative_mat = np.dot(trans_mat2, inv(trans_mat1))

    translations = relative_mat[:3, 3]
    rotations_eulers = np.asarray(tfms.euler_from_matrix(relative_mat))

    rotations_degrees = (rotations_eulers / (2 * math.pi)) * 360

    dof = np.concatenate((translations, rotations_degrees), axis=0)

    return dof


def transform_vol():
    print('hereh')
    case_id = 'Case0005'
    alignment_dir = '/zion/guoh9/US_recon/alignment'
    us_vol = sitk.ReadImage('/zion/guoh9/US_recon/recon/{}/{}_origin_gt.mhd'.format(case_id, case_id))
    mr_vol = sitk.ReadImage('/zion/guoh9/US_recon/alignment/{}/MRVol_adjusted.mhd'.format(case_id))
    alignmat = np.loadtxt('tmp/prediction_0005.txt')
    print('us {}, mr {}'.format(us_vol.GetSize(), mr_vol.GetSize()))
    print('alignmat {}'.format(alignmat.shape))
    print('alignmat\n{}'.format(alignmat))

    # max_size = 2000
    # destVol = sitk.Image(max_size, max_size, max_size, sitk.sitkFloat32)
    # destVol.SetOrigin(-0.5*np.asarray([max_size, max_size, max_size])
    #                     *np.asarray(us_vol.GetSpacing()))
    # destSpacing = np.asarray(us_vol.GetSpacing())
    # destVol.SetSpacing((destSpacing[0], destSpacing[1], destSpacing[2]))

    # alignmat = np.linalg.inv(alignmat)
    alignmat = np.identity(4)
    alignmat[0][3] = 5
    print('alignmat\n{}'.format(alignmat))
    usmr_tfm = mat2tfm(alignmat)
    tfm_chain_my = sitk.CompositeTransform(mat2tfm(np.identity(4)))
    tfm_chain_my.AddTransform(usmr_tfm)

    resampler_us = sitk.ResampleImageFilter()
    # resampler_us.SetReferenceImage(mr_vol)
    # resampler_us.SetReferenceImage(destVol)
    resampler_us.SetReferenceImage(us_vol)
    resampler_us.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler_us.SetDefaultPixelValue(0)
    resampler_us.SetTransform(tfm_chain_my)
    outMovingImg = resampler_us.Execute(us_vol)
    us_my_align_chain_path = path.join(alignment_dir, case_id, 'USVol_xs.mhd')
    sitk.WriteImage(outMovingImg, us_my_align_chain_path)
    print('saved to <{}>'.format(us_my_align_chain_path))
    return 0

def get_mr_pts(img_size, img_spacing, img_origin):
    # print('img_size {}'.format(img_size))
    # print('img_spacing {}'.format(img_spacing))
    # print('img_origin {}'.format(img_origin))

    size_3d = np.asarray(img_size) * np.asarray(img_spacing)
    # print('size_3d {}'.format(size_3d))
    sx, sy, sz = size_3d
    ox, oy, oz = img_origin
    # print('{} {} {}'.format(sx, sy, sz))
    # print('{} {} {}'.format(ox, oy, oz))

    corner_pts = [[[ox, oy, oz],
                  [ox+sx, oy, oz],
                  [ox+sx, oy+sy, oz],
                  [ox, oy+sy, oz]],
                  [[ox, oy, oz+sz],
                  [ox+sx, oy, oz+sz],
                  [ox+sx, oy+sy, oz+sz],
                  [ox, oy+sy, oz+sz]]]
    corner_pts = np.asarray(corner_pts)
    # sys.exit()
    return corner_pts

def read_landmarks(xml_path, origin):
    if not os.path.isfile(xml_path):
        print('Not exsits! <{}>'.format(xml_path))
        # sys.exit()
        return np.zeros((2,3))

    landmarks = []
    with open(xml_path, 'r') as filehandle:
        for line in filehandle:
            if '=' in line:
                components = line[:-1].split('=')
                # print(components)
                landmarks.append(float(components[-1]))
    landmarks = np.asarray(landmarks)
    landmarks = np.reshape(landmarks, (landmarks.shape[0]//3, 3))
    landmarks = np.flip(landmarks, 1)
    # print(landmarks)
    landmarks = landmarks + origin
    # print(origin)
    # print(landmarks)
    return landmarks

def plane_abcd_from_pts(frame_pts):
    x1, y1, z1 = frame_pts[0, 0, :]
    x2, y2, z2 = frame_pts[0, 1, :]
    x3, y3, z3 = frame_pts[0, 2, :]

    a = (y3-y1)*(z3-z1) - (z2-z1)*(y3-y1)
    b = (x3-x1)*(z2-z1) - (x2-x1)*(z3-z1)
    c = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    d = -(a*x1 + b*y1 + c*z1)

    return a, b, c, d

def landmark_distance(landmark, plane_mat):
    # print('landmark {}'.format(landmark.shape))
    frame_pts = mats2pts_correction(np.expand_dims(plane_mat, 0))
    # print('frame_pts {}'.format(frame_pts.shape))
    a, b, c, d = plane_abcd_from_pts(frame_pts)

    dists = []
    for i in range(landmark.shape[0]):
        x, y, z = tuple(landmark[i,:])
        dist = abs(a*x+b*y+c*z+d) / math.sqrt(a*a+b*b+c*c)
        dists.append(dist)
    return dists

def umeyama_mat(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    # Cross-covariance matrix
    C = np.dot(np.transpose(centeredP), centeredQ) / n

    # Singular vector decomposition
    V, S, W = np.linalg.svd(C)
    # d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    # print('d {}'.format(d))
    # if d:
    #     S[-1] = -S[-1]
    #     V[:, -1] = -V[:, -1]

    # Compute Rotation matrix 3x3
    R = np.dot(V, W)

    # Computer Scaling factor
    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    c = 1

    # Compute translation, first rotate source points P 
    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    r_mat = np.identity(4)
    r_mat[:3, :3] = c*R
    t_mat = np.identity(4)
    t_mat[3, :3] = t
    new_mat = np.dot(r_mat, t_mat)
    new_mat = np.transpose(new_mat)
    return new_mat

def tfm2mat(input_tfm):
    # print(input_tfm)
    # print(input_tfm.GetName())
    parameters = input_tfm.GetParameters()
    parameters = np.asarray(parameters)

    mat44 = np.identity(4)
    if parameters.shape[0] == 12:
        print('This is <AffineTransform>')
        mat44[:3, :3] = np.reshape(parameters[:9], (3, 3))
        mat44[:3, 3] = parameters[9:]

        mat_rot = np.identity(4)
        mat_rot[:3, :3] = np.reshape(parameters[:9], (3, 3))
        mat_trans = np.identity(4)
        mat_trans[:3, 3] = parameters[9:]
        # mat44 = np.dot(mat_rot, mat_trans)
        mat44 = np.dot(mat_trans, mat_rot)

        # print(input_tfm)
    elif parameters.shape[0] == 6:
        print('This is <Euler3DTransform>')
        fixed_parameters = input_tfm.GetFixedParameters()
        # print(input_tfm)
        # print('fixed_parameters: {}'.format(fixed_parameters))
        # print('parameters: {}'.format(parameters))

        # mat44 = tfms.compose_matrix(angles=parameters[:3])
        # mat44[:3, 3] = parameters[3:]

        mat_rot = tfms.compose_matrix(angles=parameters[:3])
        mat_trans = np.identity(4)
        mat_trans[:3, 3] = parameters[3:]

        mat44 = np.dot(mat_rot, mat_trans)
        # mat44 = np.dot(mat_trans, mat_rot)
    elif parameters.shape[0] == 3:
        print('This is <TranslationTransform>')
        mat44[:3, 3] = parameters[:]
        # sys.exit()
    # fixed_parameters = input_tfm.GetFixedParameters()
    # parameters = np.asarray(parameters)
    # # print(mat33)

    # mat44 = np.identity(4)
    # mat44[:3, :3] = np.reshape(parameters[:9], (3, 3))
    # mat44[:3, 3] = parameters[9:]
    # print(mat44)

    # sys.exit()
    # return mat44
    return mat44

def tfm2mat_composite(input_tfm):
    # print(input_tfm.FlattenTransform())
    input_tfm.FlattenTransform()
    num_tfm = input_tfm.GetNumberOfTransforms()
    print('num_tfm {}'.format(num_tfm))

    mats = []

    for i in range(num_tfm):
        tfm = input_tfm.GetNthTransform(i)
        # print(tfm)
        mat = tfm2mat(tfm)
        mats.append(mat)
    # print('here')
    compose_mat = np.identity(4)
    # print('here')
    # print(mats)
    for mat in mats:
        compose_mat = np.dot(compose_mat, mat)
    #     print('3')
    # print(compose_mat)
    # sys.exit()
    return compose_mat, mats

def crop_fan(input_img):
    # xyhw=[105, 54, 320, 565]
    out_img = input_img[105:105+320, 54:54+565]
    return out_img

def define_model(model_type, model_device, pretrained_path='', neighbour_slice=5,
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
    elif model_type == 'convlstm':
        model_ft = convlstm_regress.ConvLSTM(input_dim=1,
                                hidden_dim=[64, 64, 16],
                                kernel_size=(3, 3),
                                num_layers=3,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)
    else:
        print('network type of <{}> is not supported, use original instead'.format(model_type))
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

    if pretrained_path:
        if path.isfile(pretrained_path):
            print('Loading model from <{}>...'.format(pretrained_path))
            model_ft.load_state_dict(torch.load(pretrained_path, map_location=model_device))
            # model_ft.load_state_dict(torch.load(pretrained_path))
            print('Model loaded successfully!')
        else:
            print('<{}> not exists! Training from scratch...'.format(pretrained_path))
    else:
        print('Train this model from scratch!')

#     model_ft.cuda()
    model_ft.eval()
#     model_device = torch.device("cuda:{}".format(device_no))
#     model_device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(model_device)
    # print('define model device {}'.format(device))
    return model_ft

if __name__ == "__main__":
    transform_vol()
    # draw_mr_box()