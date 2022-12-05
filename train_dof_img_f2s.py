from cv2 import phase
from numpy import indices
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import numpy as np
# import torchvision.models.resnet as resnet
from networks import resnet
from networks import resnext
import time
import os
from os import path
import random
# from stl import mesh
import SimpleITK as sitk
import cv2
from datetime import datetime
import argparse
import tools
import loss_functions
# from functions import mahalanobis
from networks import generators
from networks import mynet
from networks import p3d
from networks import densenet
import sys
import network_func
import matplotlib.pyplot as plt
################

desc = 'Training registration generator'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-i', '--init_mode',
                    type=str,
                    help="mode of training with different transformation matrics",
                    default='random_SRE2')

parser.add_argument('-t', '--training_mode',
                    type=str,
                    help="mode of training with different starting points",
                    default='scratch')

parser.add_argument('-m', '--model_filename',
                    type=str,
                    help="name of the pre-trained mode file",
                    default='None')

parser.add_argument('-l', '--learning_rate',
                    type=float,
                    help='Learning rate',
                    default=1e-4)

parser.add_argument('-d', '--device_no',
                    type=int,
                    choices=[0, 1, 2, 3, 4, 5, 6, 7],
                    help='GPU device number [0-7]',
                    default=0)

parser.add_argument('-e', '--epochs',
                    type=int,
                    help='number of training epochs',
                    default=500)

parser.add_argument('-n', '--network_type',
                    type=str,
                    help='choose different network architectures'
                         'the size of inputs/outputs are the same'
                         'could be original, resnext101',
                    # default='convlstm')
                    default='resnext50')

parser.add_argument('-info', '--information',
                    type=str,
                    help='infomation of this round of experiment',
                    default='Here is the information')

parser.add_argument('-ns', '--neighbour_slice',
                    type=int,
                    help='number of slice that acts as one sample',
                    default='2')

parser.add_argument('-it', '--input_type',
                    type=str,
                    help='input type of the network,'
                         'org_img, diff_img, optical flow',
                    default='org_img')

parser.add_argument('-ot', '--output_type',
                    type=str,
                    help='output type of the network,'
                         'average_dof, separate_dof, sum_dof',
                    default='average_dof')

pretrain_model_str = '0213-092230'
networks3D = ['resnext50', 'resnext101', 'densenet121', 'mynet', 'mynet2', 'p3d', 'convlstm']

net = 'Generator'
batch_size = 4
use_last_pretrained = False
current_epoch = 0

args = parser.parse_args()
device_no = args.device_no
epochs = args.epochs

training_progress = np.zeros((epochs, 4))

hostname = os.uname().nodename
zion_common = '/zion/guoh9'
on_arc = False
if 'arc' == hostname:
    on_arc = True
    print('on_arc {}'.format(on_arc))
    # device = torch.device("cuda:{}".format(device_no))
    zion_common = '/raid/shared/guoh9'
    batch_size = 48
    # batch_size = 32
    # batch_size = 16
# device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:{}".format(device_no))
# print('start device {}'.format(device))
US_dataset_path = path.join(zion_common, 'US_recon/US_dataset')

fan_mask = cv2.imread('data/avg_img.png', 0)

# normalize_dof = True
normalize_dof = False
# dof_stats = np.loadtxt('infos/dof_stats.txt')
dof_avg = np.loadtxt('infos/mean_dof.txt')
dof_std = np.loadtxt('infos/std_dof.txt')


high_low_diary = []

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


def define_model(model_type, pretrained_path='', neighbour_slice=args.neighbour_slice,
                 input_type=args.input_type, output_type=args.output_type, device_no=0):
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

    model_ft.cuda()
    model_ft.eval()
    model_device = torch.device("cuda:{}".format(device_no))
    model_ft = model_ft.to(model_device)
    # print('define model device {}'.format(device))
    return model_ft


# input an image array
# normalize values to 0-255
def array_normalize(input_array):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    # print('max {}, min {}'.format(max_value, min_value))
    k = 255 / (max_value - min_value)
    min_array = np.ones_like(input_array) * min_value
    normalized = k * (input_array - min_array)
    return normalized


def filename_list(dir):
    images = []
    dir = os.path.expanduser(dir)
    # print('dir {}'.format(dir))
    for filename in os.listdir(dir):
        # print(filename)
        file_path = path.join(dir, filename)
        images.append(file_path)
        # print(file_path)
    # print(images)
    return images


def normalize_volume(input_volume):
    # print('input_volume shape {}'.format(input_volume.shape))
    mean = np.mean(input_volume)
    std = np.std(input_volume)

    normalized_volume = (input_volume - mean) / std
    # print('normalized shape {}'.format(normalized_volume.shape))
    # time.sleep(30)
    return normalized_volume


def scale_volume(input_volume, upper_bound=255, lower_bound=0):
    max_value = np.max(input_volume)
    min_value = np.min(input_volume)

    k = (upper_bound - lower_bound) / (max_value - min_value)
    scaled_volume = k * (input_volume - min_value) + lower_bound
    # print('min of scaled {}'.format(np.min(scaled_volume)))
    # print('max of scaled {}'.format(np.max(scaled_volume)))
    return scaled_volume


class FreehandUS4D(Dataset):

    def __init__(self, case_list, initialization, phase=None):
        """
        """
        self.samples = case_list
        self.phase = phase
        self.initialization = initialization

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        # print(self.samples)
        # sys.exit()
        # case_folder = '/zion/guoh9/US_recon/US_vid_frames/train/Case0141'
        case_id = self.samples[idx]
        case_folder = path.join(US_dataset_path, case_id)
        # case_folder = path.join(data_dir, self.phase, self.samples[idx])
        # print(case_folder)
        # print(os.listdir(case_folder))
        # sys.exit()
        # case_id = int(case_folder[-4:])

        # norm_path = path.normpath(case_folder)
        # res = norm_path.split(os.sep)
        # status = res[-2]

        # """ Make sure we do not use weird BK scans """
        # if case_id not in clean_ids[status]:
        #     case_id = int(np.random.choice(clean_ids[status], 1)[0])
        #     case_folder = path.join(data_dir, status, 'Case{:04}'.format(case_id))

        # aurora_pos = np.loadtxt(path.join(pos_dir, 'Case{:04}.txt'.format(case_id)))
        # calib_mat = np.loadtxt(path.join(uronav_dir, '{}/Case{:04}/Case{:04}_USCalib.txt'.format(status, case_id, case_id)))
        # print('aurora_pos {}'.format(np.mean(aurora_pos)))
        # print('calib_mat {}'.format(np.mean(calib_mat)))
        aurora_pos = np.loadtxt(path.join(US_dataset_path, case_id, 'Position.txt'))
        calib_mat = np.loadtxt(path.join(US_dataset_path, case_id, 'USCalib.txt'))
        # print('aurora_pos {}'.format(np.mean(aurora_pos)))
        # print('calib_mat {}'.format(np.mean(calib_mat)))

        # frame_num = len(os.listdir(case_folder))
        frame_num = aurora_pos.shape[0]
        sample_size = args.neighbour_slice
        # print('Case{:04} have {} frames'.format(case_id, frame_num))
        # print('sample_size {}'.format(sample_size))

        valid_range = frame_num - sample_size
        start_id = np.random.randint(low=0, high=valid_range, size=1)[0]
        start_params = aurora_pos[start_id, :]
        # start_mat = tools.params_to_mat44(trans_params=start_params, cam_cali_mat=calib_mat)
        # print('selected start index {}'.format(rand_num))

        select_ids = tools.sample_ids(slice_num=frame_num, neighbour_num=sample_size,
                                      sample_option='normal',
                                      random_reverse_prob=0.5, self_prob=0)
        # print('{} slices, select_ids\n{}'.format(frame_num, select_ids))
        # time.sleep(30)

        # recon_dir = path.join(zion_common, 'US_recon/recon_gt')
        # us_path = path.join(recon_dir, '{}_origin_gt.mhd'.format(case_id))
        # us_sitk = sitk.ReadImage(us_path)
        # gt_mats = np.load(path.join(zion_common, 'US_recon/recon_mats/{}_gt_mats.npy'.format(case_id)))
        # gt_pts1, gt_mats = tools.params2corner_pts_correction(case_id=case_id, 
        #                             params=aurora_pos,
        #                             cam_cali_mat=calib_mat,
        #                             input_img=np.zeros((480, 640)),
        #                             correction=False)
        # print('<{}>'.format(us_path))
        recon_slices_dir = path.join(zion_common, 'US_recon/recon_slices')
        # print(gt_mats)
        # print(case_id)
        # sys.exit()

        sample_slices = []
        labels = []
        # for slice_index in range(start_id, start_id + sample_size):
        # for slice_index in select_ids:
        for i in range(select_ids.shape[0]):
            slice_index = select_ids[i]

            if i == 0:
                """ Previous frame is sliced from volume, for FVR correction! """
                slice_path = path.join(recon_slices_dir, case_id, '{:04}.jpg'.format(slice_index))
                slice_img = cv2.imread(slice_path, 0)
                # slice_img = tools.resample_slice(us_sitk, gt_mats[slice_index, :, :], crop_frame=True)
                # slice_path = path.join(US_dataset_path, case_id, 'frames', '{:04}.jpg'.format(slice_index))
                # slice_img_raw = cv2.imread(slice_path, 0)                
                # concat = np.concatenate((slice_img, slice_img_raw), axis=0)
                # print('slice_img {}'.format(slice_img.shape))
                # cv2.imwrite('tmp/tmp.jpg', concat)
                # plt.figure()
                # plt.imshow(concat)
                # plt.show()
                # sys.exit()
            else:
                slice_path = path.join(US_dataset_path, case_id, 'frames', '{:04}.jpg'.format(slice_index))
                slice_img = cv2.imread(slice_path, 0)
            
            slice_img = data_transform(slice_img, masked_full=False)
            sample_slices.append(slice_img)
            # print('slice_img shape {}'.format(slice_img.shape))

            # if slice_index != select_ids[0]:
            if i != select_ids.shape[0] - 1:
                first_id = select_ids[i]
                second_id = select_ids[i + 1]
                dof = tools.get_6dof_label(trans_params1=aurora_pos[first_id, :],
                                           trans_params2=aurora_pos[second_id, :],
                                           cam_cali_mat=calib_mat)
                # print(dof)
                # dof = tools.get_6dof_from_mats(gt_mats[first_id,:,:], gt_mats[second_id,:,:])
                # print(dof)
                # print('loadmat1\n{}'.format(gt_mats[first_id,:,:]))
                # print('loadmat2\n{}'.format(gt_mats[second_id,:,:]))

                # sys.exit()
                labels.append(dof)
        # format_labels = np.asarray(labels)
        # format_labels = np.around(format_labels, decimals=3)
        # print('labels {}'.format(np.asarray(labels).shape))
        # sys.exit()

        if args.input_type == 'diff_img':
            diff_imgs = []
            for sample_id in range(1, len(sample_slices)):
                diff_imgs.append(sample_slices[sample_id] - sample_slices[sample_id - 1])
            sample_slices = np.asarray(diff_imgs)
            # print('sample_slices shape {}'.format(sample_slices.shape))
            # time.sleep(30)
        else:
            sample_slices = np.asarray(sample_slices)

        if args.output_type == 'average_dof':
            labels = np.mean(np.asarray(labels), axis=0)
        elif args.output_type == 'sum_dof':
            end2end_dof = tools.get_6dof_label(trans_params1=aurora_pos[select_ids[0], :],
                                               trans_params2=aurora_pos[select_ids[-1], :],
                                               cam_cali_mat=calib_mat)
            labels = end2end_dof
        else:
            labels = np.asarray(labels).flatten()
        # print('sample_slices shape {}'.format(sample_slices.shape))
        # print('labels shape {}'.format(labels.shape))
        # print('int_label\n{}'.format(format_labels))
        # time.sleep(30)

        if network_type in networks3D:
            sample_slices = np.expand_dims(sample_slices, axis=0)
            # print('sample_slices shape {}'.format(sample_slices.shape))
            # time.sleep(30)

        if normalize_dof:
            # labels = (labels - dof_stats[:, 0]) / dof_stats[:, 1]
            labels = (labels - dof_avg) / dof_std
            # print(labels.shape)
            # print(labels)
            # time.sleep(30)


        sample_slices = torch.from_numpy(sample_slices).float().to(device)
        labels = torch.from_numpy(labels).float().to(device)
        # print('dataloader device {}'.format(device))
        # print('sample_slices shape {}'.format(sample_slices.shape))
        # print('labels shape {}'.format(labels.shape))

        # print('selected_ids\n{}'.format(select_ids))
        # print('labels\n{}'.format(labels))
        # time.sleep(30)
        return sample_slices, labels, case_id, start_params, calib_mat


def get_dist_loss(labels, outputs, start_params, calib_mat):
    # print('labels shape {}'.format(labels.shape))
    # print('outputs shape {}'.format(outputs.shape))
    # print('start_params shape {}'.format(start_params.shape))
    # print('calib_mat shape {}'.format(calib_mat.shape))

    # print('labels_before\n{}'.format(labels.shape))
    labels = labels.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy()
    if normalize_dof:
        # labels = labels / dof_stats[:, 1] + dof_stats[:, 0]
        # outputs = outputs / dof_stats[:, 1] + dof_stats[:, 0]

        labels = labels / dof_std + dof_avg
        outputs = outputs / dof_std + dof_avg


    start_params = start_params.data.cpu().numpy()
    calib_mat = calib_mat.data.cpu().numpy()

    if args.output_type == 'sum_dof':
        batch_errors = []
        for sample_id in range(labels.shape[0]):
            gen_param = tools.get_next_pos(trans_params1=start_params[sample_id, :],
                                           dof=outputs[sample_id, :],
                                           cam_cali_mat=calib_mat[sample_id, :, :])
            gt_param = tools.get_next_pos(trans_params1=start_params[sample_id, :],
                                          dof=labels[sample_id, :],
                                          cam_cali_mat=calib_mat[sample_id, :, :])
            gen_param = np.expand_dims(gen_param, axis=0)
            gt_param = np.expand_dims(gt_param, axis=0)

            result_pts = tools.params2corner_pts(params=gen_param, cam_cali_mat=calib_mat[sample_id, :, :])
            gt_pts = tools.params2corner_pts(params=gt_param, cam_cali_mat=calib_mat[sample_id, :, :])

            sample_error = tools.evaluate_dist(pts1=gt_pts, pts2=result_pts)
            batch_errors.append(sample_error)

        batch_errors = np.asarray(batch_errors)

        avg_batch_error = np.asarray(np.mean(batch_errors))
        error_tensor = torch.tensor(avg_batch_error, requires_grad=True)
        error_tensor = error_tensor.type(torch.FloatTensor)
        error_tensor = error_tensor.to(device)
        error_tensor = error_tensor * 0.99
        # print('disloss device {}'.format(device))
        # print(error_tensor)
        # time.sleep(30)
        return error_tensor




    if args.output_type == 'average_dof':
        labels = np.expand_dims(labels, axis=1)
        labels = np.repeat(labels, args.neighbour_slice - 1, axis=1)
        outputs = np.expand_dims(outputs, axis=1)
        outputs = np.repeat(outputs, args.neighbour_slice - 1, axis=1)
    else:
        labels = np.reshape(labels, (labels.shape[0], labels.shape[1] // 6, 6))
        outputs = np.reshape(outputs, (outputs.shape[0], outputs.shape[1] // 6, 6))
    # print('labels_after\n{}'.format(labels.shape))
    # print('outputs\n{}'.format(outputs.shape))
    # time.sleep(30)

    batch_errors = []
    final_drifts = []
    for sample_id in range(labels.shape[0]):
        gen_param_results = []
        gt_param_results = []
        for neighbour in range(labels.shape[1]):
            if neighbour == 0:
                base_param_gen = start_params[sample_id, :]
                base_param_gt = start_params[sample_id, :]
            else:
                base_param_gen = gen_param_results[neighbour - 1]
                base_param_gt = gt_param_results[neighbour - 1]
            gen_dof = outputs[sample_id, neighbour, :]
            gt_dof = labels[sample_id, neighbour, :]
            gen_param = tools.get_next_pos(trans_params1=base_param_gen, dof=gen_dof,
                                           cam_cali_mat=calib_mat[sample_id, :, :])
            gt_param = tools.get_next_pos(trans_params1=base_param_gt, dof=gt_dof,
                                          cam_cali_mat=calib_mat[sample_id, :, :])
            gen_param_results.append(gen_param)
            gt_param_results.append(gt_param)
        gen_param_results = np.asarray(gen_param_results)
        gt_param_results = np.asarray(gt_param_results)
        # print('gen_param_results shape {}'.format(gen_param_results.shape))

        result_pts = tools.params2corner_pts(params=gen_param_results, cam_cali_mat=calib_mat[sample_id, :, :])
        gt_pts = tools.params2corner_pts(params=gt_param_results, cam_cali_mat=calib_mat[sample_id, :, :])
        # print(result_pts.shape, gt_pts.shape)
        # time.sleep(30)

        results_final_vec = np.mean(result_pts[-1, :, :], axis=0)
        gt_final_vec = np.mean(gt_pts[-1, :, :], axis=0)
        final_drift = np.linalg.norm(results_final_vec - gt_final_vec) * 0.2
        final_drifts.append(final_drift)
        # print(results_final_vec, gt_final_vec)
        # print(final_drift)
        # time.sleep(30)

        sample_error = tools.evaluate_dist(pts1=gt_pts, pts2=result_pts)
        batch_errors.append(sample_error)

    batch_errors = np.asarray(batch_errors)
    avg_batch_error = np.asarray(np.mean(batch_errors))

    error_tensor = torch.tensor(avg_batch_error, requires_grad=True)
    error_tensor = error_tensor.type(torch.FloatTensor)
    error_tensor = error_tensor.to(device)
    error_tensor = error_tensor * 0.99
    # print('disloss device {}'.format(device))
    # print(error_tensor)
    # time.sleep(30)

    avg_final_drift = np.asarray(np.mean(np.asarray(final_drifts)))
    final_drift_tensor = torch.tensor(avg_final_drift, requires_grad=True)
    final_drift_tensor = final_drift_tensor.type(torch.FloatTensor)
    final_drift_tensor = final_drift_tensor.to(device)
    final_drift_tensor = final_drift_tensor * 0.99
    return error_tensor, final_drift_tensor


def get_correlation_loss(labels, outputs):
    # print('labels shape {}, outputs shape {}'.format(labels.shape, outputs.shape))
    x = outputs.flatten()
    y = labels.flatten()
    # print('x shape {}, y shape {}'.format(x.shape, y.shape))
    # print('x shape\n{}\ny shape\n{}'.format(x, y))
    xy = x * y
    mean_xy = torch.mean(xy)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = mean_xy - mean_x * mean_y
    # print('xy shape {}'.format(xy.shape))
    # print('xy {}'.format(xy))
    # print('mean_xy {}'.format(mean_xy))
    # print('cov_xy {}'.format(cov_xy))

    var_x = torch.sum((x - mean_x) ** 2 / x.shape[0])
    var_y = torch.sum((y - mean_y) ** 2 / y.shape[0])
    # print('var_x {}'.format(var_x))

    corr_xy = cov_xy / (torch.sqrt(var_x * var_y))
    # print('correlation_xy {}'.format(corr_xy))

    loss = 1 - corr_xy
    # time.sleep(30)
    # x = output
    # y = target
    #
    # vx = x - torch.mean(x)
    # vy = y - torch.mean(y)
    #
    # loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    # print('correlation loss {}'.format(loss))
    # time.sleep(30)
    return loss



#

# ----- #
def _get_random_value(r, center, hasSign):
    randNumber = random.random() * r + center


    if hasSign:
        sign = random.random() > 0.5
        if sign == False:
            randNumber *= -1

    return randNumber


# ----- #
def get_array_from_itk_matrix(itk_mat):
    mat = np.reshape(np.asarray(itk_mat), (3, 3))
    return mat


# ----- #
def create_transform(aX, aY, aZ, tX, tY, tZ, mat_base=None):
    if mat_base is None:
        mat_base = np.identity(3)

    t_all = np.asarray((tX, tY, tZ))

    # Get the transform
    rotX = sitk.VersorTransform((1, 0, 0), aX / 180.0 * np.pi)
    matX = get_array_from_itk_matrix(rotX.GetMatrix())
    #
    rotY = sitk.VersorTransform((0, 1, 0), aY / 180.0 * np.pi)
    matY = get_array_from_itk_matrix(rotY.GetMatrix())
    #
    rotZ = sitk.VersorTransform((0, 0, 1), aZ / 180.0 * np.pi)
    matZ = get_array_from_itk_matrix(rotZ.GetMatrix())

    # Apply all the rotations
    mat_all = matX.dot(matY.dot(matZ.dot(mat_base[:3, :3])))

    return mat_all, t_all

def pairedDist(labels):
    batch_size = labels.shape[0]
    source = labels.unsqueeze(0).repeat(batch_size, 1, 1)
    target = labels.unsqueeze(1).repeat(1, batch_size, 1)

    # print(source.shape)
    # print(target.shape)
    epsilon = 1e-10

    # dist = torch.sqrt(torch.sum(torch.square(source-target+epsilon), 2))

    dist = torch.sqrt(torch.sum((source-target+epsilon)*(source-target+epsilon), 2))

    # print(dist)
    # sys.exit()
    return dist

def pairedDist2(dist):
    dist_row = dist.unsqueeze(0)
    dist_col = dist.unsqueeze(1)
    # print('dist_row {}'.format(dist_row.shape))
    # print('dist_col {}'.format(dist_col.shape))

    dist = dist_row - dist_col
    # print('dist:\n{}'.format(dist.shape))
    # print('dist:\n{}'.format(dist))
    # sys.exit()
    return dist



def contrastive_loss(label, vec, margin=0):
    bs = label.shape[0]

    lab_dist_mat = pairedDist(label)
    vec_dist_mat = pairedDist(vec)

    # print('lab_dist_mat\n{}'.format(lab_dist_mat))
    # print('vec_dist_mat\n{}'.format(vec_dist_mat))
    # print('\n')

    lab_dist_vec = lab_dist_mat[torch.triu_indices(bs,bs,1).unbind()]
    vec_dist_vec = vec_dist_mat[torch.triu_indices(bs,bs,1).unbind()]

    # print('lab_dist_vec\n{}'.format(lab_dist_vec))
    # print('vec_dist_vec\n{}'.format(vec_dist_vec))
    # print('\n')

    lab_dist_mat2 = pairedDist2(lab_dist_vec)
    vec_dist_mat2 = pairedDist2(vec_dist_vec)

    bs2 = lab_dist_mat2.shape[0]

    lab_dist_vec2 = lab_dist_mat2[torch.triu_indices(bs2,bs2,1).unbind()]
    vec_dist_vec2 = vec_dist_mat2[torch.triu_indices(bs2,bs2,1).unbind()]

    # print('lab_dist_mat2\n{}'.format(lab_dist_mat2))
    # print('vec_dist_mat2\n{}'.format(vec_dist_mat2))
    # print('\n')


    # print('lab_dist_vec2\n{}'.format(lab_dist_vec2))
    # print('vec_dist_vec2\n{}'.format(vec_dist_vec2))
    # print('\n')

    target = lab_dist_vec2.sign()

    # print('target\n{}'.format(target))
    # print('\n')

    mrl_loss = criterion_mrl(vec_dist_vec2, torch.zeros_like(vec_dist_vec2), target)
    # print(mrl_loss)

    # relu = torch.nn.ReLU()
    # mrl_loss = torch.sum(relu(-target * vec_dist_vec2)) / target.shape[0]
    # sys.exit()

    return mrl_loss

def contrastive_loss2(dist, vec, threshold=1.5):
    print(dist)

    bs = dist.shape[0]

    high = dist > threshold
    low  = torch.logical_and((dist > 0), (dist <= threshold))
    
    # print('high\n{}'.format(high))
    # print('low\n{}'.format(low))

    # high_loss = torch.zeros(1, requires_grad=True)
    # low_loss  = torch.zeros(1, requires_grad=True)

    high_loss = []
    low_loss = []


    # print('dist shape0 {}'.format(dist.shape[0]))

    for i in range(dist.shape[0]):
        # high_idx = high[:, i].nonzero().squeeze()
        # low_idx  = low[:, i].nonzero().squeeze()
        high_idx = high[:, i].nonzero()
        low_idx  = low[:, i].nonzero()

        # print('high_idx {}'.format(high_idx.shape))
        # print('low_idx  {}'.format(low_idx.shape))
        # sys.exit()

        if high_idx.shape[0] > 0:
            index = np.random.choice(high_idx[:, 0].data.cpu().numpy(), 1)[0]
            # print('high index {}'.format(index))
            # high_loss += torch.sqrt(torch.sum(torch.square(vec[i, :]-vec[index, :])))
            high_loss.append(torch.sqrt(torch.sum(torch.square(vec[i, :]-vec[index, :]))))
        else:
            high_loss.append(torch.sum(torch.square(vec[i, :]-vec[i, :])))

        
        if low_idx.shape[0] > 0:
            index = np.random.choice(low_idx[:, 0].data.cpu().numpy(), 1)[0]
            # print('low  index {}'.format(index))
            # low_loss += torch.sqrt(torch.sum(torch.square(vec[i, :]-vec[index, :])))
            low_loss.append(torch.sqrt(torch.sum(torch.square(vec[i, :]-vec[index, :]))))
        else:
            low_loss.append(torch.sum(torch.square(vec[i, :]-vec[i, :])))
            
        # sys.exit()
        # print('high_loss {}'.format(high_loss))
        # print('low_loss  {}'.format(low_loss))
        # print('\n')

    # sys.exit()
    low_loss = sum(low_loss) / bs
    high_loss = sum(high_loss) / bs

    print('low  {}'.format(low_loss))
    print('high {}'.format(high_loss))

    high_low_diary.append([high_loss.data.cpu().numpy(),
                           low_loss.data.cpu().numpy()])

    # print(high_low_array)
    print('featvec shape {}'.format(vec.shape))
    sys.exit()
    return low_loss - high_loss

def train_model(model, criterion, optimizer, scheduler, fn_save, num_epochs=25):
    since = time.time()

    lowest_loss = 2000
    lowest_dist = 2000
    best_ep = 0
    tv_hist = {'train': [], 'val': []}
    # print('trainmodel device {}'.format(device))

    for epoch in range(num_epochs):
        global current_epoch
        current_epoch = epoch + 1

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # print('Network is in {}...'.format(phase))

            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_mse = 0.0
            running_cst = 0.0
            running_dist = 0.0
            running_cor = 0.0
            # running_corrects = 0

            # Iterate over data.
            for inputs, labels, case_id, start_params, calib_mat in dataloaders[phase]:
                # Get images from inputs
                # print('*'*10 + ' printing inputs and labels ' + '*'*10)
                labels = labels.type(torch.FloatTensor)
                inputs = inputs.type(torch.FloatTensor)
                # base_mat = base_mat.type(torch.FloatTensor)
                # img_id = img_id.type(torch.FloatTensor)
                labels = labels.to(device)
                inputs = inputs.to(device)
                # base_mat = base_mat.to(device)
                # img_id = img_id.to(device)

                labels.require_grad = True

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print('inputs shape {}'.format(inputs.shape))
                    # print('labels shape {}'.format(labels.shape))
                    # print('labels {}'.format(labels))
                    # sys.exit()
                    outputs, vec = model(inputs)
                    # print('outputs shape {}'.format(outputs.shape))
                    # print('vec     shape {}'.format(vec.shape))
                    # sys.exit()

                    # paired_dist = pairedDist(labels)

                    # print('contrast {}'.format(contrast))

                    # high_low_array = np.asarray(high_low_diary)
                    # np.savetxt('results/high_low_array.txt', high_low_array)



                    '''Weighted MSE loss function'''
                    # my_weight = torch.Tensor([0.5/4,0.5/2,0.5/2,0.5/4,0.5/4,0.5/4]).cuda()
                    # loss = weighted_mse_loss(input=outputs, target=labels, weights=my_weight)
                    # loss_weight = torch.Tensor([1, 1, 1, 1, 0, 0]).cuda().to(device)
                    # loss = weighted_mse_loss(input=outputs, target=labels, weights=loss_weight)

                    # print('outputs type {}, labels type {}'.format(outputs.dtype, labels.type))
                    # dist_loss, drift_loss = get_dist_loss(labels=labels, outputs=outputs,
                    #                                       start_params=start_params, calib_mat=calib_mat)
                    loss_cst = contrastive_loss(labels, vec)
                    loss_cor = get_correlation_loss(labels=labels, outputs=outputs)
                    # corr_loss = loss_functions.get_correlation_loss(labels=labels,
                    #                                                 outputs=outputs,
                    #                                                 dof_based=True)
                    # print('corr_loss {:.5f}'.format(corr_loss))
                    # print('dist_loss {:.5f}'.format(dist_loss))
                    # time.sleep(30)

                    loss_mse = criterion(outputs, labels)

                    # loss = loss_functions.dof_MSE(labels=labels, outputs=outputs,
                    #                               criterion=criterion, dof_based=True)

                    hybrid_loss = loss_mse*5 + loss_cor + loss_cst
                    # hybrid_loss = loss_mse*5 + loss_cor
                    # hybrid_loss = loss_mse*5
                    

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # loss.backward()
                        hybrid_loss.backward()
                        # dist.backward()
                        optimizer.step()
                    # print('update loss')
                    # sys.exit()
                # statistics
                running_loss += hybrid_loss.item() * inputs.size(0)
                running_mse += loss_mse.item() * inputs.size(0)
                running_cor += loss_cor.item() * inputs.size(0)
                running_cst += loss_cst.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_mse = running_mse / dataset_sizes[phase]
            epoch_cor = running_cor / dataset_sizes[phase]
            epoch_cst = running_cst / dataset_sizes[phase]
            
            # print('epoch_dist {}'.format(epoch_dist))

            tv_hist[phase].append([epoch_loss, epoch_mse, epoch_cor, epoch_cst])

            # deep copy the model
            # if (phase == 'val' and epoch_loss <= lowest_loss) or current_epoch % 10 == 0:
            if phase == 'val' and epoch_loss <= lowest_loss:
            # if phase == 'val':
            # if phase == 'val' and epoch_dist <= lowest_dist:
                lowest_loss = epoch_loss
                # lowest_dist = epoch_dist
                best_ep = epoch
                torch.save(model.state_dict(), fn_save)
                # print('**** best model updated with dist={:.4f} ****'.format(lowest_dist))
                print('**** best model updated with loss={:.4f} ****'.format(lowest_loss))

        update_info(best_epoch=best_ep+1, current_epoch=epoch+1, lowest_val_TRE=lowest_loss)
        print('{}/{}: ttl ({:.2f}, {:.2f}), mse ({:.2f}, {:.2f}), cor ({:.2f}, {:.2f}), cst ({:.2f}, {:.2f})'.format(
            epoch + 1, num_epochs,
            tv_hist['train'][-1][0],
            tv_hist['val'][-1][0],
            tv_hist['train'][-1][1],
            tv_hist['val'][-1][1],
            tv_hist['train'][-1][2],
            tv_hist['val'][-1][2],
            tv_hist['train'][-1][3],
            tv_hist['val'][-1][3])
        )
        # print(tv_hist)
        np.save(diary_path, tv_hist)

    time_elapsed = time.time() - since
    print('*' * 10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('*' * 10 + 'Lowest val TRE: {:4f} at epoch {}'.format(lowest_dist, best_ep))
    print()

    return tv_hist

def save_info():
    file = open('data/experiment_diary/{}.txt'.format(now_str), 'a+')
    file.write('Time_str: {}\n'.format(now_str))
    # file.write('Initial_mode: {}\n'.format(args.init_mode))
    file.write('Training_mode: {}\n'.format(args.training_mode))
    file.write('Model_filename: {}\n'.format(args.model_filename))
    file.write('Device_no: {}\n'.format(args.device_no))
    file.write('Epochs: {}\n'.format(args.epochs))
    file.write('Network_type: {}\n'.format(args.network_type))
    file.write('Learning_rate: {}\n'.format(args.learning_rate))
    file.write('Neighbour_slices: {}\n'.format(args.neighbour_slice))
    file.write('Infomation: {}\n'.format(args.information))
    file.write('Best_epoch: 0\n')
    file.write('Val_loss: {:.4f}\n'.format(1000))
    file.close()
    print('Information has been saved!')

def update_info(best_epoch, current_epoch, lowest_val_TRE):
    readFile = open('data/experiment_diary/{}.txt'.format(now_str))
    lines = readFile.readlines()
    readFile.close()

    file = open('data/experiment_diary/{}.txt'.format(now_str), 'w')
    file.writelines([item for item in lines[:-2]])
    file.write('Best_epoch: {}/{}\n'.format(best_epoch, current_epoch))
    file.write('Val_loss: {:.4f}'.format(lowest_val_TRE))
    file.close()
    # print('Info updated in {}!'.format(now_str))


if __name__ == '__main__':
    # data_dir = path.join('/home/guoh9/tmp/US_vid_frames')
    # results_dir = path.join('/home/guoh9/tmp/US_vid_frames')

    data_dir = path.join(zion_common, 'US_recon/US_vid_frames')
    pos_dir = path.join(zion_common, 'US_recon/US_vid_pos')
    uronav_dir = path.join(zion_common, 'uronav_data')

    train_ids = np.loadtxt('infos/train_ids.txt')
    val_ids = np.loadtxt('infos/val_ids.txt')
    clean_ids = {'train': train_ids, 'val': val_ids}

    if 'arc' == hostname:
        results_dir = '/home/guoh9/US_recon/results'
    else:
        results_dir = path.join(zion_common, 'US_recon/results')
    
    results_dir = 'pretrained_networks'

    init_mode = args.init_mode
    network_type = args.network_type
    print('Training mode: {}'.format(args.training_mode))

    """ Decide training datasets split """
    # dataset_dir = 'infos/training_sets/mc72_new'
    dataset_dir = 'infos/training_sets/c95'
    # dataset_dir = 'infos/training_sets/mc72'
    # dataset_dir = 'infos/training_sets/media_mc72_up'
    # dataset_dir = 'infos/training_sets/mc72_valid/fold_1'

    dataset_dir = 'infos/training_sets/c95_5cv/f1'
    dataset_dir = 'infos/training_sets/c95_5cv/f2'
    # dataset_dir = 'infos/training_sets/c95_5cv/f3'
    # dataset_dir = 'infos/training_sets/c95_5cv/f4'
    # dataset_dir = 'infos/training_sets/c95_5cv/f5'
    # dataset_dir = 'infos/training_sets/c95_5cv/f6'
    # dataset_dir = 'infos/training_sets/c95_5cv/f7'
    # dataset_dir = 'infos/training_sets/c95_5cv/f8'
    # dataset_dir = 'infos/training_sets/c95_5cv/f9'


    train_ids = tools.read_list(path.join(dataset_dir, 'train.txt'))
    test_ids = tools.read_list(path.join(dataset_dir, 'test.txt'))
    val_ids = tools.read_list(path.join(dataset_dir, 'val.txt'))

    all_ids = {'train': train_ids*10,
               'val'  : val_ids*10,
               'test' : test_ids}
    print('train {}, val {}, test {}'.format(len(all_ids['train']), len(all_ids['val']), len(all_ids['test'])))
    # sys.exit()

    # image_datasets = {x: FreehandUS4D(os.path.join(dataset_dir, '{}.txt'.format(x)), init_mode)
    #                   for x in ['train', 'val']}
    image_datasets = {x: FreehandUS4D(all_ids[x], init_mode)
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    print('Number of training samples: {}'.format(dataset_sizes['train']))
    print('Number of validation samples: {}'.format(dataset_sizes['val']))

    model_folder = '/zion/guoh9/US_recon/results'
    model_folder = 'pretrained_networks'

    pretrain_model_str = 'dcl2_big_m25_ns7_nocst'
    model_path = path.join(model_folder, '3d_best_Generator_{}.pth'.format(pretrain_model_str))  # 10
    # model_ft = define_model(model_type=network_type, pretrained_path=model_path)
    # model_ft = define_model(model_type=network_type, device_no=args.device_no)
    model_ft = network_func.define_model(model_type=network_type, 
                            pretrained_path=None,
                            input_type=args.input_type,
                            output_type=args.output_type,
                            neighbour_slice=args.neighbour_slice,
                            device_no=args.device_no)



    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    criterion_mrl = nn.MarginRankingLoss(margin=0.25)
    # criterion = nn.()
    # criterion = nn.L1Loss()

    # mahalanobis_dist = mahalanobis.MahalanobisMetricLoss()

    if args.training_mode == 'finetune':
        # overwrite the learning rate for finetune
        lr = 5e-6
        print('Learning rate is overwritten to be {}'.format(lr))
    else:
        lr = args.learning_rate
        print('Learning rate = {}'.format(lr))

    optimizer = optim.Adam(model_ft.parameters(), lr=lr)
    # optimizer = optim.Adagrad(model_ft.parameters(), lr=1)
    # optimizer = optim.SGD(model_ft.parameters(), lr=lr)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    now = datetime.now()
    now_str = now.strftime('%m%d-%H%M%S')
    now_str = 'dcl2_big_m10_lr4'
    # now_str = 'dcl2_big_m25_again'
    # now_str = 'dcl2_big_m25_lr5'
    # now_str = 'dcl2_big_m15'
    now_str = 'dcl_big_nocst_again'
    now_str = 'dcl_small_nocst'
    now_str = 'dcl_small_nocst_pretrain'
    now_str = 'dcl_small_nocst_pretrain2'
    # now_str = 'dcl_small_nocst_pretrain3'
    # now_str = 'dcl_big_nocst_nocor'

    """ Neighboring slices number ablation"""
    now_str = 'dcl2_big_m25_ns2'
    now_str = 'dcl2_big_m25_ns3'
    now_str = 'dcl2_big_m25_ns4'
    # now_str = 'dcl2_big_m25_ns5'
    now_str = 'dcl2_big_m25_ns6'
    now_str = 'dcl2_big_m25_ns7'
    now_str = 'dcl2_big_m25_ns8'
    now_str = 'dcl2_big_m25_ns9'

    """ Component ablation study """
    now_str = 'dcl2_big_m25_ns7_nocst'
    now_str = 'dcl2_big_m25_ns7_nocst_nocor'

    now_str = 'convlstm'
    now_str = 'dcl_mc72_ns5_nocst_lr4'
    now_str = 'dcl_mc72_ns5_nocst_lr5'
    now_str = 'dcl_mc72_valid_ns5_nocst_lr4'
    now_str = 'dcl_mc72_valid_big_ns5_nocst_lr4'
    now_str = 'dcl_mc72_valid_big_ns5_nocst_lr4_f4'
    now_str = 'dcl_mc72_valid_big_ns5_nocst_lr5_f1'
    now_str = 'convlstm_mc72_f1'
    now_str = 'dcl_mc72_valid_big_ns2_nocst_lr4_f1'

    now_str = 'dcl_c95_ns5_normalized'
    now_str = 'dcl2_big_m25_ns2_bi'
    now_str = 'dcl2_big_m25_ns2_new'
    now_str = 'dcl2_big_m25_ns2_bi_slice' # directly slice from the volumes, slow
    now_str = 'dcl2_big_m25_ns2_bi_slice2' # load the pre-sliced frames
    now_str = 'dcl2_big_m25_ns2_bi_slice2_aug' # load the pre-sliced frames, 50% augment cut slices

    now_str = 'dcl2_big_m25_ns2_bi_slice_f1' # directly slice from the volumes, slow
    now_str = 'dcl2_big_m25_ns2_bi_slice_f2' # directly slice from the volumes, slow
    # now_str = 'dcl2_big_m25_ns2_bi_slice_f3' # directly slice from the volumes, slow
    # now_str = 'dcl2_big_m25_ns2_bi_slice_f4' # directly slice from the volumes, slow
    now_str = 'dcl2_big_m25_ns2_bi_slice_f5' # directly slice from the volumes, slow
    # now_str = 'dcl2_big_m25_ns2_bi_slice_f6' # directly slice from the volumes, slow
    # now_str = 'dcl2_big_m25_ns2_bi_slice_f7' # directly slice from the volumes, slow
    # now_str = 'dcl2_big_m25_ns2_bi_slice_f8' # directly slice from the volumes, slow
    # now_str = 'dcl2_big_m25_ns2_bi_slice_f9' # directly slice from the volumes, slow

    print('This model string is {}'.format(now_str))
    save_info()

    # Train and evaluate
    fn_best_model = path.join(results_dir, '3d_best_{}_{}.pth'.format(net, now_str))
    print('Start training...')
    # print('This model is <3d_best_{}_{}_{}.pth>'.format(net, now_str, init_mode))
    print('This model is <{}>'.format(fn_best_model))
    # sys.exit()
    # txt_path = path.join(results_dir, 'training_progress_{}_{}.txt'.format(net, now_str))
    diary_path = path.join(results_dir, 'training_progress_{}_{}.npy'.format(net, now_str))
    hist_ft = train_model(model_ft,
                          criterion,
                          optimizer,
                          exp_lr_scheduler,
                          fn_best_model,
                          num_epochs=epochs)

    # fn_hist = os.path.join(results_dir, 'hist_{}_{}_{}.npy'.format(net, now_str, init_mode))
    # np.save(fn_hist, hist_ft)

    # np.savetxt(txt_path, training_progress)

    now = datetime.now()
    now_stamp = now.strftime('%Y-%m-%d %H:%M:%S')
    print('#' * 15 + ' Training {} completed at {} '.format(init_mode, now_stamp) + '#' * 15)
    print('This model is {}'.format(now_str))