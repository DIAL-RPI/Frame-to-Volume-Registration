import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import numpy as np
import os.path as path
import os
import sys
import cv2
import tools
import SimpleITK as sitk
import imageio
import io
from scipy.special import gammainc

def draw_one_frame(ax, corner_pts, name, colorRGB=(255, 0, 0), line_width=3, constant=True, alpha=1):
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

    frame_id = 0
    for pt_id in range(-1, 3):
        xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id, pt_id + 1, 0]
        ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id, pt_id + 1, 1]
        zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id, pt_id + 1, 2]
        if pt_id == -1:
            ax.plot(xs, ys, zs, color=tuple(colors[frame_id, :]), lw=line_width, zorder=1, label=name, alpha=alpha)
        else:
            ax.plot(xs, ys, zs, color=tuple(colors[frame_id, :]), lw=line_width, zorder=1, alpha=alpha)

    return ax

def draw_one_cube(ax, corner_pts, name, colorRGB=(255, 0, 0), line_width=3, ls='dashed', constant=True, alpha=1):
    colorRGB = tuple(channel/255 for channel in colorRGB)
    # edges = [[0,1], [1,2], [2,3], [3,0],
    #          [4,5], [5,6], [6,7], [7,4],
    #          [0,4], [1,5], [2,6], [3,7]]
    
    # for edge_id, pts in enumerate(edges):
    #     pt1, pt2 = pts
    #     xs = corner_pts[pt1, 0], corner_pts[pt2, 0]
    #     ys = corner_pts[pt1, 1], corner_pts[pt2, 1]
    #     zs = corner_pts[pt1, 2], corner_pts[pt2, 2]
    #     if edge_id == 7:
    #         ax.plot(xs, ys, zs, color=colorRGB, lw=line_width, zorder=1, label=name, alpha=alpha)
    #     else:
    #         ax.plot(xs, ys, zs, color=colorRGB, lw=line_width, zorder=1, alpha=alpha)

    # sys.exit()
    for frame_id in range(corner_pts.shape[0]):
        if frame_id == 0:
            """ First frame draw full bounds"""
            for pt_id in range(-1, 3):
                xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id, pt_id + 1, 0]
                ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id, pt_id + 1, 1]
                zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id, pt_id + 1, 2]
                ax.plot(xs, ys, zs, color=colorRGB, lw=line_width, zorder=1, alpha=0.5, linestyle=ls)
        elif frame_id == corner_pts.shape[0] - 1:
            """ Connect to the former frame """
            for pt_id in range(-1, 3):
                xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id - 1, pt_id, 0]
                ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id - 1, pt_id, 1]
                zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id - 1, pt_id, 2]
                ax.plot(xs, ys, zs, color=colorRGB, lw=line_width, alpha=0.5, linestyle=ls)
            """ Last frame draw full bounds"""
            for pt_id in range(-1, 3):
                xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id, pt_id + 1, 0]
                ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id, pt_id + 1, 1]
                zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id, pt_id + 1, 2]
                ax.plot(xs, ys, zs, color=colorRGB, lw=line_width, alpha=0.5, linestyle=ls)
                if pt_id == -1:
                    ax.plot(xs, ys, zs, color=colorRGB, lw=line_width, label=name, alpha=0.5)
        else:
            """ Connect to the former frame """
            for pt_id in range(-1, 3):
                xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id - 1, pt_id, 0]
                ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id - 1, pt_id, 1]
                zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id - 1, pt_id, 2]
                ax.plot(xs, ys, zs, color=colorRGB, lw=line_width, zorder=1, alpha=0.5, linestyle=ls)
    return ax

def get_bounds(all_pts):
    # print('all_pts.shape {}'.format(all_pts.shape))
    # sys.exit()

    x_min, x_max = np.min(all_pts[:, :, 0]), np.max(all_pts[:, :, 0])
    y_min, y_max = np.min(all_pts[:, :, 1]), np.max(all_pts[:, :, 1])
    z_min, z_max = np.min(all_pts[:, :, 2]), np.max(all_pts[:, :, 2])

    big_range = max(x_max-x_min, y_max-y_min, z_max-z_min) * 1.2 / 2
    center = [(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2]
    # print('center {}, range {}'.format(center, big_range))

    x_range = [center[0]-big_range, center[0]+big_range]
    y_range = [center[1]-big_range, center[1]+big_range]
    z_range = [center[2]-big_range, center[2]+big_range]
    # print('x_range {}'.format(x_range))
    # print('y_range {}'.format(y_range))
    # print('z_range {}'.format(z_range))

    # sys.exit()
    return x_range, y_range, z_range

def draw_fig(init_mat, gt_mats, pd_mats, frame_ids, frames_num=None, 
             mr_vol_info=None, landmarks=None, title='Case', caption=None):
    init_mat = np.expand_dims(init_mat, axis=0)
    # print('init {}, gt {}, pd {}'.format(init_mat.shape, gt_mats.shape, pd_mats.shape))
    # sys.exit()

    gt_pts = tools.mats2pts_correction(gt_mats)
    pd_pts = tools.mats2pts_correction(pd_mats)
    init_pts = tools.mats2pts_correction(init_mat)
    x_range, y_range, z_range = get_bounds(np.concatenate((gt_pts, pd_pts)))

    if mr_vol_info is not None:
        mr_corner_pts = tools.get_mr_pts(mr_vol_info['img_size'],
                                         mr_vol_info['img_spacing'],
                                         mr_vol_info['img_origin'])
        x_range, y_range, z_range = get_bounds(np.concatenate((gt_pts, pd_pts, mr_corner_pts)))

    frame_num = gt_pts.shape[0]

    fig_list = []
    for frame_id in range(pd_mats.shape[0]):

        fig = plt.figure(figsize=(6, 6), dpi=128)
        ax = fig.gca(projection='3d')
        ax.view_init(elev=7, azim=17)
        ax.view_init(elev=13, azim=165)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_zlim(z_range[0], z_range[1])
        
        gt_frame_pts = np.expand_dims(gt_pts[frame_id, :, :], 0)
        pd_frame_pts = np.expand_dims(pd_pts[frame_id, :, :], 0)
        # print('{}/{}: gt_frame_pts {}, pd_frame_pts {}'.format(frame_ids[frame_id]+1, frames_num, gt_frame_pts.shape, pd_frame_pts.shape))
        if mr_vol_info is not None:
            # print('mr_corner_pts.shape {}'.format(mr_corner_pts.shape))
            draw_one_cube(ax, mr_corner_pts, name='MRVol', colorRGB=(0,0,255), alpha=0.5)
            # sys.exit()
        
        if landmarks is not None:
            # print('landmarks {}'.format(landmarks.shape))
            ax.scatter(landmarks[:,0], landmarks[:,1], landmarks[:,2], 
                       color='purple', marker='*', label='landmarks')
            # sys.exit()

        draw_one_frame(ax, init_pts, name='Init', colorRGB=(255,0,0), alpha=0.2)
        draw_one_frame(ax, gt_frame_pts, name='GT', colorRGB=(0,255,0))
        draw_one_frame(ax, pd_frame_pts, name='PD', colorRGB=(255,0,0))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.title('{}: frame {:04}\n{}'.format(title, frame_ids[frame_id], caption))
        plt.legend(loc='lower right')

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # print('data.shape {}'.format(data.shape))
        # sys.exit()

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', dpi=128)
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()


        # cv2.imshow('win', img_arr)
        # cv2.waitKey(0)
        # sys.exit()

        # plt.savefig('tmp/tmp_3d.jpg')
        # plt.show()
        # sys.exit()
        plt.close()

        # fig_img = cv2.imread('tmp/tmp_3d.jpg')
        # fig_img = cv2.cvtColor(fig_img, cv2.COLOR_BGR2RGB)
        # fig_list.append(fig_img)
        fig_list.append(img_arr[:, :, :3])
        # plt.imshow(fig_img)
        # plt.show()
        # sys.exit()
    
    # imageio.mimsave(path.join(fig_folder, '{}.gif'.format(case_id)), fig_list, duration=0.1)
    # imageio.mimsave('tmp/gif/{}_frames.gif'.format(case_id), fig_list, duration=0.1)
    # fig_list = np.asarray(fig_list)
    # print('fig_list shape {}'.format(fig_list.shape))
    # sys.exit()
    return fig_list

def sample(center,radius,n_per_sphere):
    r = radius
    ndim = center.size
    x = np.random.normal(size=(n_per_sphere, ndim))
    ssq = np.sum(x**2,axis=1)
    fr = r*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n_per_sphere,1),(1,ndim))
    p = center + np.multiply(x,frtiled)
    return p

def generate_landmarks_ball(landmark_sample, n=10):
    print(landmark_sample)

    center = np.mean(landmark_sample, axis=0)
    radius = np.linalg.norm(landmark_sample[0,:] - landmark_sample[1,:]) / 2
    # print('radius {}'.format(radius))
    points = sample(center, radius, n)
    # print(p)
    # sys.exit()
    return points

def draw_fig_vis(case_id, frames_num=None, mr_vol_info=None, landmarks=None, title='Case', caption=None):
    # case_id = 'Case0008'
    mats_gt = np.load('tmp/vis_fig/{}/mats_gt.npy'.format(case_id))
    mats_0 = np.load('tmp/vis_fig/{}/mats_0.npy'.format(case_id))
    mats_1 = np.load('tmp/vis_fig/{}/mats_1.npy'.format(case_id))
    mats_2 = np.load('tmp/vis_fig/{}/mats_2.npy'.format(case_id))
    mats_3 = np.load('tmp/vis_fig/{}/mats_3.npy'.format(case_id))
    print(mats_0.shape)
    # sys.exit()
    init_mat = mats_gt[0,:,:]
    init_mat = np.expand_dims(init_mat, axis=0)

    landmarks = np.load('tmp/vis_fig/{}/landmarks.npy'.format(case_id), allow_pickle=True)
    mr_vol_info = np.load('tmp/vis_fig/{}/mr_vol_info.npy'.format(case_id), allow_pickle=True).item()
    evaluations_0 = np.load('tmp/vis_fig/{}/evaluations_0.npy'.format(case_id), allow_pickle=True).item()
    evaluations_1 = np.load('tmp/vis_fig/{}/evaluations_1.npy'.format(case_id), allow_pickle=True).item()
    evaluations_2 = np.load('tmp/vis_fig/{}/evaluations_2.npy'.format(case_id), allow_pickle=True).item()
    evaluations_3 = np.load('tmp/vis_fig/{}/evaluations_3.npy'.format(case_id), allow_pickle=True).item()
    print('landmarks:\n{}'.format(landmarks.shape))
    print('mr_vol_info:\n{}'.format(mr_vol_info))

    landmarks_gen = generate_landmarks_ball(landmarks, n=100)
    # sys.exit()



    gt_pts = tools.mats2pts_correction(mats_gt)
    pd_pts0 = tools.mats2pts_correction(mats_0)
    pd_pts1 = tools.mats2pts_correction(mats_1)
    pd_pts2 = tools.mats2pts_correction(mats_2)
    pd_pts3 = tools.mats2pts_correction(mats_3)
    # print(gt_pts.shape)
    # sys.exit()
    init_pts = tools.mats2pts_correction(init_mat)
    x_range, y_range, z_range = get_bounds(np.concatenate((gt_pts, pd_pts0, pd_pts2, pd_pts3)))

    if mr_vol_info is not None:
        mr_corner_pts = tools.get_mr_pts(mr_vol_info['img_size'],
                                         mr_vol_info['img_spacing'],
                                         mr_vol_info['img_origin'])
        x_range, y_range, z_range = get_bounds(np.concatenate((gt_pts, pd_pts0, pd_pts2, pd_pts3, mr_corner_pts)))

    frame_num = gt_pts.shape[0]
    print(frame_num)

    fig_list = []
    for frame_id in range(mats_gt.shape[0]):

        fig = plt.figure(figsize=(6, 6), dpi=128)
        ax = fig.gca(projection='3d')
        ax.view_init(elev=7, azim=17)
        ax.view_init(elev=13, azim=165)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_zlim(z_range[0], z_range[1])
        
        gt_frame_pts = np.expand_dims(gt_pts[frame_id, :, :], 0)
        pd_frame_pts0 = np.expand_dims(pd_pts0[frame_id, :, :], 0)
        pd_frame_pts1 = np.expand_dims(pd_pts1[frame_id, :, :], 0)
        pd_frame_pts2 = np.expand_dims(pd_pts2[frame_id, :, :], 0)
        pd_frame_pts3 = np.expand_dims(pd_pts3[frame_id, :, :], 0)
        # print('{}/{}: gt_frame_pts {}, pd_frame_pts {}'.format(frame_ids[frame_id]+1, frames_num, gt_frame_pts.shape, pd_frame_pts.shape))
        if mr_vol_info is not None:
            # print('mr_corner_pts.shape {}'.format(mr_corner_pts.shape))
            draw_one_cube(ax, mr_corner_pts, name='MRVol', colorRGB=(0,0,255), alpha=0.5)
            # sys.exit()
        
        if landmarks is not None:
            # print('landmarks {}'.format(landmarks.shape))
            ax.scatter(landmarks[0,0], landmarks[0,1], landmarks[0,2], 
                       color='purple', marker='*', label='Bladder Neck')
            ax.scatter(landmarks[1,0], landmarks[1,1], landmarks[1,2], 
                       color='crimson', marker='*', label='Urethra Exit')

            ax.scatter(landmarks_gen[:,0], landmarks_gen[:,1], landmarks_gen[:,2], 
                       color='green', marker='.', label='Fake Landmarks')
            # sys.exit()

        draw_one_frame(ax, init_pts, name='Init', colorRGB=(255,0,0), alpha=0.2)
        line_alpha = 0.7
        draw_one_frame(ax, gt_frame_pts, name='GT', colorRGB=(0,0,0), alpha=line_alpha)
        # draw_one_frame(ax, pd_frame_pts0, name='Method 0 ({:.2f}mm)'.format(evaluations_0['err'][frame_id]), colorRGB=(230,159,0), alpha=line_alpha)
        # draw_one_frame(ax, pd_frame_pts1, name='Method 1 ({:.2f}mm)'.format(evaluations_1['err'][frame_id]), colorRGB=(0,114,178), alpha=line_alpha)
        # draw_one_frame(ax, pd_frame_pts2, name='Method 2 ({:.2f}mm)'.format(evaluations_2['err'][frame_id]), colorRGB=(255,0,0), alpha=line_alpha)
        # draw_one_frame(ax, pd_frame_pts3, name='Method 3 ({:.2f}mm)'.format(evaluations_3['err'][frame_id]), colorRGB=(0,158,115), alpha=line_alpha)
        draw_one_frame(ax, pd_frame_pts0, name='F2F', alpha=line_alpha)
        draw_one_frame(ax, pd_frame_pts1, name='F2F + F2S', colorRGB=(0,114,178), alpha=line_alpha)
        draw_one_frame(ax, pd_frame_pts3, name='FVReg', colorRGB=(0,158,115), alpha=line_alpha)
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.tight_layout(True)
        # plt.title('{}: frame {:04}\n{}'.format(title, frame_ids[frame_id], caption))
        plt.legend(loc='lower right')
        # plt.legend(loc='upper left')

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # print('data.shape {}'.format(data.shape))
        # sys.exit()

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', dpi=128)
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.close()


        # cv2.imshow('win', img_arr)
        # cv2.waitKey(0)
        # sys.exit()

        plt.savefig('tmp/vis_fig/{}/fig_{}.jpg'.format(case_id, frame_id))
        plt.show()
        sys.exit()
        plt.close()
        # sys.exit()
        print('{}/{} finished'.format(frame_id+1, frame_num))

        # fig_img = cv2.imread('tmp/tmp_3d.jpg')
        # fig_img = cv2.cvtColor(fig_img, cv2.COLOR_BGR2RGB)
        # fig_list.append(fig_img)
        fig_list.append(img_arr[:, :, :3])
        # plt.imshow(fig_img)
        # plt.show()
        # sys.exit()
    
    imageio.mimsave('tmp/vis_fig/{}/fig.gif'.format(case_id), fig_list, duration=0.1)

    return fig_list

def gen_gifs(case_id, frame_num, end, adjust):
    file_dir = 'tmp/vis_fig/{}'.format(case_id)
    img_list = []
    for i in range(end-frame_num+1, end+1):
        img = cv2.imread(path.join(file_dir, '{:04}_{}.jpg'.format(i, adjust)))
        img = cv2.resize(img, (300, 300))
        img_list.append(img)
        # print(i)
        # plt.imshow(img)
        # plt.show()
        # sys.exit()
    gif_path = path.join(file_dir, 'gif_{}.gif'.format(adjust))    
    imageio.mimsave(gif_path, img_list, duration=0.1)
    print(len(img_list))
    # sys.exit()
    return 0

if __name__ == '__main__':

    # gen_gifs(case_id='Case0008', frame_num=100, end=112, adjust='gt')
    # gen_gifs(case_id='Case0008', frame_num=100, end=112, adjust='pd_3')
    gen_gifs(case_id='Case0012', frame_num=123, end=138, adjust='gt')
    gen_gifs(case_id='Case0012', frame_num=123, end=138, adjust='pd_3')
    sys.exit()

    draw_fig_vis(case_id='Case0008')
    sys.exit()

    test_ids = tools.read_list('infos/sets/fvr_ready/test.txt')

    for case_id in test_ids[8:]:
        case_id = 'Case0009'
        print('******* {} *******'.format(case_id))
        alignment_dir = '/zion/guoh9/US_recon/alignment'
        case_recon_vol_fn = path.join(alignment_dir, case_id, 'USVol_myrecon_lined_gt_mr.mhd')
        case_us = sitk.ReadImage(case_recon_vol_fn)

        gt_mats = np.load('tmp/fvrmats/{}_gt_mats.npy'.format(case_id))
        pd_mats = np.load('tmp/fvrmats/{}_pd_mats.npy'.format(case_id))

        draw_fig(init_mat=gt_mats[30, :, :], gt_mats=gt_mats, pd_mats=pd_mats)
        sys.exit()

        print('gt {}, pd {}'.format(gt_mats.shape, pd_mats.shape))

        gt_pts = tools.mats2pts_correction(gt_mats)
        pd_pts = tools.mats2pts_correction(pd_mats)
        x_range, y_range, z_range = get_bounds(np.concatenate((gt_pts, pd_pts)))

        print('gt {}, pd {}'.format(gt_pts.shape, pd_pts.shape))

        fig_folder = 'tmp/frame_figs/{}'.format(case_id)
        if not os.path.isdir(fig_folder):
            os.mkdir(fig_folder)

        mid_id = gt_pts.shape[0] // 2
        mid_frame_pts = np.expand_dims(gt_pts[mid_id, :, :], 0)

        fig_list = []
        for frame_id in range(gt_pts.shape[0]):

            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca(projection='3d')
            ax.view_init(elev=7., azim=17)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # ax.set_xlim(-100, 100)
            # ax.set_ylim(-100, 100)
            # ax.set_zlim(-50, 150)

            ax.set_xlim(x_range[0], x_range[1])
            ax.set_ylim(y_range[0], y_range[1])
            ax.set_zlim(z_range[0], z_range[1])
            
            # frame_id = 10
            gt_frame_pts = np.expand_dims(gt_pts[frame_id, :, :], 0)
            pd_frame_pts = np.expand_dims(pd_pts[frame_id, :, :], 0)
            print('gt_frame_pts {}, pd_frame_pts {}'.format(gt_frame_pts.shape, pd_frame_pts.shape))

            draw_one_frame(ax, mid_frame_pts, name='Init', colorRGB=(0,0,255))
            draw_one_frame(ax, gt_frame_pts, name='GT', colorRGB=(0,255,0))
            draw_one_frame(ax, pd_frame_pts, name='PD', colorRGB=(255,0,0))
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.title('{}: frame {:04}'.format(case_id, frame_id))
            plt.legend(loc='lower right')
            plt.savefig(path.join(fig_folder, '{:04}.jpg'.format(frame_id)))
            plt.show()
            sys.exit()

            plt.close()

            fig_img = cv2.imread(path.join(fig_folder, '{:04}.jpg'.format(frame_id)))
            fig_img = cv2.cvtColor(fig_img, cv2.COLOR_BGR2RGB)
            fig_list.append(fig_img)
        
        # imageio.mimsave(path.join(fig_folder, '{}.gif'.format(case_id)), fig_list, duration=0.1)
        imageio.mimsave('tmp/gif/{}_frames.gif'.format(case_id), fig_list, duration=0.1)
        # sys.exit()
    





