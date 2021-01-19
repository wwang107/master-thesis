from skimage.feature import peak_local_max
from .multiview import findFundamentalMat
from numba import vectorize, float32, float64, jit, boolean
from math import sqrt
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numba as nb
import numpy.linalg as la
import numpy as np
import torch
import cv2



@vectorize([float64(float64,float64,float64,float64,float64)])
def line_to_point_distance(a,b,c,x,y):
    return abs(a*x + b*y + c) / sqrt(a**2 + b**2)

def triangulate_viewpair(hm1, hm2, cam1, cam2, epi_distance=1.0, ax1=None, ax2=None):
    """ triangulate view pair to 3d
    :param epi_distance: maximum distance in pixel
    """
    peaks1 = peak_local_max(hm1, threshold_abs=0.3)
    peaks2 = peak_local_max(hm2, threshold_abs=0.3)
    
    values1 = hm1[peaks1[:, 0], peaks1[:, 1]]
    values2 = hm2[peaks2[:, 0], peaks2[:, 1]]
    
    F_12 = findFundamentalMat(cam1,cam2).float()
    F_21 = findFundamentalMat(cam2,cam1).float()
    
    points3d = []
    values = []
    
    if len(peaks1) > 0 and len(peaks2) > 0:
        peaks1 = torch.from_numpy(np.ascontiguousarray(peaks1)).float()
        peaks2 = torch.from_numpy(np.ascontiguousarray(peaks2)).float()
        peaks1[:, [0, 1]] = peaks1[:, [1, 0]]
        peaks2[:, [0, 1]] = peaks2[:, [1, 0]]
        
        # image1 --> image2
        kpt1_homo = torch.cat([peaks1,torch.ones_like(peaks1[:,0:1])], dim=1)
        kpt1_homo = kpt1_homo.transpose(0,1)
        l2 = torch.matmul(F_12, kpt1_homo).squeeze(dim=0)
        l2_t = l2.transpose(0, 1)
        
        # image2 --> image1
        kpt2_homo = torch.cat([peaks2,torch.ones_like(peaks2[:,0:1])], dim=1)
        kpt2_homo = kpt2_homo.transpose(0,1)
        l1 = torch.matmul(F_21, kpt2_homo).squeeze(dim=0)
        l1_t = l1.transpose(0, 1)
        
        # prepare triangulation
        point_pairs_1 = []
        point_pairs_2 = []
        values = []
        for pt1, (a1, b1, c1), v1, in zip(peaks1, l2_t, values1):
            for pt2, (a2, b2, c2), v2 in zip(peaks2, l1_t, values2):
                d1 = line_to_point_distance(a1, b1, c1, pt2[0], pt2[1]).item()
                d2 = line_to_point_distance(a2, b2, c2, pt1[0], pt1[1]).item()
                
                if d1 < epi_distance and d2 < epi_distance:
                    point_pairs_1.append(pt1.numpy())
                    point_pairs_2.append(pt2.numpy())
                    values.append((v1+v2)/2)
            
        point_pairs_1 = np.array(point_pairs_1).transpose()
        point_pairs_2 = np.array(point_pairs_2).transpose()
        values = np.array(values)
        
        P1 = cam1.numpy()
        P2 = cam2.numpy()
        
        if len(point_pairs_1) > 0:
            try:
                pts3d_homo = cv2.triangulatePoints(P1, P2, point_pairs_1, point_pairs_2)
                points3d = (pts3d_homo/pts3d_homo[3])[:3].transpose()
            except ValueError:
                print('point_pairs_1', point_pairs_1.shape)
                print('point_pairs_2', point_pairs_2.shape)
                raise ValueError("nope, byebye")
        
            if ax1 is not None and ax2 is not None:
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # optional: draw stuff if axis are being passed
                pt1_in_im2 = lambda x: (l2[0,:] * x + l2[2,:]) / (-l2[1,:])
                y1_a = pt1_in_im2(0).numpy(); y1_b = pt1_in_im2(10000).numpy()
                pt2_in_im1 = lambda x: (l1[0,:] * x + l1[2,:]) / (-l1[1,:])
                y2_a = pt2_in_im1(0).numpy(); y2_b = pt2_in_im1(10000).numpy()
                ax2.plot([0, 10000], [y1_a, y1_b])
                ax1.plot([0, 10000], [y2_a, y2_b])
                ax2.scatter(peaks2[:,0],peaks2[:,1], color='yellow')
                ax1.scatter(peaks1[:,0],peaks1[:,1], color='yellow')

                pts2d_1 = P1 @ pts3d_homo
                pts2d_1 = (pts2d_1/pts2d_1[2])[:2]
                ax1.scatter(pts2d_1[0], pts2d_1[1], color='red', alpha=0.5)

                pts2d_2 = P2 @ pts3d_homo
                pts2d_2 = (pts2d_2/pts2d_2[2])[:2]
                ax2.scatter(pts2d_2[0], pts2d_2[1], color='red', alpha=0.5)

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            points3d = []
            values = []
    return points3d, values


def generate_3d_cloud(HMs, Cameras, Axs=None):
    assert len(HMs) == len(Cameras)
    n_cams = len(Cameras)
    points3d = []
    values = []
    for a in range(n_cams-1):
        for b in range(a+1, n_cams):
            pts3d, val = triangulate_viewpair(
                HMs[a], HMs[b], Cameras[a], Cameras[b]
            )
            if len(pts3d) > 0:
                points3d.append(pts3d)
                values.append(val)
            
    if len(points3d) > 0:
        try:
            points3d = np.concatenate(points3d)
            values = np.concatenate(values)
        except:
            print('---')
            for p in points3d:
                print('p', p)
            raise ValueError("nope..")
    
    if Axs is not None:
        assert len(Axs) == n_cams
        points3d_pt = torch.from_numpy(points3d)
        points3d_homo = torch.cat([points3d_pt,torch.ones_like(points3d_pt[:,0:1])], dim=1)
        points3d_homo = points3d_homo.transpose(0,1)
        
        for cid, P in enumerate(Cameras):
            ax = Axs[cid]
            pts2d = P.numpy() @ points3d_homo.numpy()
            pts2d = (pts2d/pts2d[2])[:2]
            ax.scatter(pts2d[0], pts2d[1], color='red', alpha=0.5)
        
    return points3d, values

@nb.njit(nb.float64(
    nb.float64[:], nb.float64[:], nb.float64))
def K(x, x_pr, gamma):
    """kernel"""
    return np.exp(- np.linalg.norm(x-x_pr)**2 / (2 * gamma*2))


@nb.njit(nb.float64[:, :](
    nb.float64[:, :], nb.float64[:], nb.float64, nb.float64
), nogil=True)

def meanshift(points3d, values, neighborhood_size_mm, gamma):
    """
    :param points3d: [n_points x 3]
    :param points3d: [n_points]
    """
    meanshift_convergence = 0.00001
    final_positions = points3d.copy()
    n_points = points3d.shape[0]
    for i in range(n_points):
        
        for step in range(10000):  # max meanshift steps
            x = final_positions[i]
            mx = np.zeros_like(x)
            sum_Kx = 0.0
            total_neighbors = 0
            for j in range(n_points):
                if j == i:
                    xi = x.copy()
                else:
                    xi = points3d[j]
                
                d = la.norm(x - xi)
                if d < neighborhood_size_mm:
                    Kx = K(x, xi, gamma)
                    Kx = Kx * values[j]  # weight based on the pred
                    sum_Kx += Kx
                    mx += Kx * xi
                    total_neighbors += 1
                    
            mx = mx/sum_Kx
            meanshift = mx - x
            if total_neighbors == 1:
                # no neighbors - end optimization
                break
            if la.norm(meanshift) < meanshift_convergence:
                # does not update enough anymore - break
                break
            
            final_positions[i] = mx
            
    return final_positions

def merge_points_based_on_distance(points3d, values, max_dist, n_iterations):
    """
    """
    for _ in range(n_iterations):
        n_points = len(points3d)
        flags = np.ones(n_points)
        
        for i in range(n_points-1):
            if flags[i] == 0:
                # ignore this
                continue
            a = points3d[i]
            wa = values[i]
            for j in range(i+1, n_points):
                b = points3d[j]
                wb = values[j]
                d = la.norm(a-b)
                if d < max_dist:
                    flags[j] = 0
                    
                    wa_ = wa/(wa+wb)
                    wb_ = wb/(wa+wb)
                    a = a*wa_ + b*wb_
                    points3d[i] = a
                    wa = (wa + wb)/2
                    values[i] = wa
        
        idc = np.nonzero(flags)
        points3d = points3d[idc]

    return points3d, values

parameters = {
    'scale_to_mm': 10.0,
    'meanshift_radius_in_mm': 200.0,
    'meanshift_gamma': 10000.0,
    'points3d_merge_distance_mm': 50.0
}

def cluster_3d_cloud(points3d, values, params=parameters, Cameras=None, Axs=None):
    """ merge point cloud to smaller set """
    assert len(points3d) == len(values)
    scale2mm = params['scale_to_mm']
    ms_radius = params['meanshift_radius_in_mm']
    gamma = params['meanshift_gamma']
    points3d_merge_distance_mm = params['points3d_merge_distance_mm']
    points3d_ms = meanshift(
        points3d.astype('float64')*scale2mm, 
        values.astype('float64'),
        ms_radius, gamma)
    points3d_ms, values = merge_points_based_on_distance(
        points3d_ms, values, points3d_merge_distance_mm, n_iterations=10
    )
    points3d_ms = points3d_ms/scale2mm
    
    if Axs is not None and Cameras is not None: 
        points3d_pt = torch.from_numpy(points3d_ms)
        points3d_homo = torch.cat([points3d_pt,torch.ones_like(points3d_pt[:,0:1])], dim=1)
        points3d_homo = points3d_homo.transpose(0,1)
        
        points3d_pt_old = torch.from_numpy(points3d)
        points3d_homo_old = torch.cat([points3d_pt_old,torch.ones_like(points3d_pt_old[:,0:1])], dim=1)
        points3d_homo_old = points3d_homo_old.transpose(0,1)
        
        for cid, P in enumerate(Cameras):
            ax = Axs[cid]
            pts2d = P.numpy() @ points3d_homo.numpy()
            pts2d = (pts2d/pts2d[2])[:2]
            
            pts2d_old = P.numpy() @ points3d_homo_old.numpy()
            pts2d_old = (pts2d_old/pts2d_old[2])[:2]
            
            ax = Axs[cid]
            
            ax.scatter(pts2d_old[0], pts2d_old[1], color='red', alpha=0.5)
            ax.scatter(pts2d[0], pts2d[1], color='blue', alpha=1, s=2)
    
    return points3d_ms, values

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def project(P, points3d):
    points3d_pt = torch.from_numpy(points3d)
    points3d_homo = torch.cat([points3d_pt,torch.ones_like(points3d_pt[:,0:1])], dim=1)
    points3d_homo = points3d_homo.transpose(0,1)
    pts2d = P.numpy() @ points3d_homo.numpy()
    pts2d = (pts2d/pts2d[2])[:2]
    return pts2d.transpose(1, 0)

class Pose:
    def __init__(self):
        self.u_arm_left = None
        self.u_arm_right = None
        self.l_arm_left = None
        self.l_arm_right = None
        self.l_side = None
        self.r_side = None
        self.u_leg_left = None
        self.u_leg_right = None
        self.l_leg_left = None
        self.l_leg_right = None
        self.hip = None
        self.shoulder = None
    
    def plot(self, ax, cam, color):
        self.color = color
        if self.u_arm_left != None:
            self.plot_limb(ax, cam, self.u_arm_left)
        if self.u_arm_right != None:
            self.plot_limb(ax, cam, self.u_arm_right)
        if self.l_arm_left != None:
            self.plot_limb(ax, cam, self.l_arm_left)
        if self.l_arm_right != None:
            self.plot_limb(ax, cam, self.l_arm_right)
        if self.l_side != None:
            self.plot_limb(ax, cam, self.l_side)
        if self.r_side != None:
            self.plot_limb(ax, cam, self.r_side)
        if self.u_leg_left != None:
            self.plot_limb(ax, cam, self.u_leg_left)
        if self.u_leg_right != None:
            self.plot_limb(ax, cam, self.u_leg_right)
        if self.l_leg_left != None:
            self.plot_limb(ax, cam, self.l_leg_left)
        if self.l_leg_right != None:
            self.plot_limb(ax, cam, self.l_leg_right)
        if self.hip != None:
            self.plot_limb(ax, cam, self.hip)
        if self.shoulder != None:
            self.plot_limb(ax, cam, self.shoulder)
    
    def plot_limb(self, ax, cam, limb):
        s2d = project(cam, np.expand_dims(limb.start3d, axis=0))
        e2d = project(cam, np.expand_dims(limb.end3d, axis=0))
        sx, sy = np.squeeze(s2d)
        ex, ey = np.squeeze(e2d)
        ax.scatter(sx, sy, s=20, color='white')
        ax.scatter(ex, ey, s=20, color='white')
        ax.plot([sx, ex], [sy, ey], c=self.color)
    
    def count_limbs(self):
        count = 0
        if self.u_arm_left != None:
            count += 1
        if self.u_arm_right != None:
            count += 1
        if self.l_arm_left != None:
            count += 1
        if self.l_arm_right != None:
            count += 1
        if self.l_side != None:
            count += 1
        if self.r_side != None:
            count += 1
        if self.u_leg_left != None:
            count += 1
        if self.u_leg_right != None:
            count += 1
        if self.l_leg_left != None:
            count += 1
        if self.l_leg_right != None:
            count += 1
        if self.hip != None:
            count += 1
        if self.shoulder != None:
            count += 1
        return count

from collections import namedtuple

LimbStats = namedtuple('LimbStats', [
    'start', 'end', 'mid1', 'mid2', 'min_length', 'max_length'
])


class Limb:
    
    def __init__(self, start3d, end3d, scale2mm, stats):
        self.start3d = start3d
        self.end3d = end3d
        self.scale2mm = scale2mm
        self.stats = stats
    
    def max_dist(self, other_limb):
        d1 = la.norm(self.start3d - other_limb.start3d) * self.scale2mm
        d2 = la.norm(self.end3d - other_limb.end3d) * self.scale2mm
        return max(d1, d2)
    
    def merge(self, other_limb):
        s3d = (self.start3d + other_limb.start3d) / 2
        e3d = (self.end3d + other_limb.end3d) / 2
        return Limb(s3d, e3d, self.scale2mm, self.stats)
        

upper_arm_left = LimbStats(
    start=5, end=7, mid1=33, mid2=34, min_length=150, max_length=400)
lower_arm_left = LimbStats(
    start=7, end=9, mid1=37, mid2=38, min_length=150, max_length=350)

upper_arm_right = LimbStats(
    start=6, end=8, mid1=35, mid2=36, min_length=150, max_length=400)
lower_arm_right = LimbStats(
    start=8, end=10, mid1=39, mid2=40, min_length=150, max_length=350)

left_side = LimbStats(
    start=5, end=11, mid1=27, mid2=28, min_length=200, max_length=600)
right_side = LimbStats(
    start=6, end=12, mid1=28, mid2=29, min_length=200, max_length=600)

hip_lr = LimbStats(
    start=11, end=12, mid1=25, mid2=26, min_length=150, max_length=400)

shoulder_lr = LimbStats(
    start=5, end=6, mid1=31, mid2=32, min_length=150, max_length=500)

left_upper_leg = LimbStats(
    start=11, end=13, mid1=20, mid2=19, min_length=100, max_length=450)
right_upper_leg = LimbStats(
    start=12, end=14, mid1=24, mid2=23, min_length=100, max_length=450)

left_lower_leg = LimbStats(
    start=13, end=15, mid1=17, mid2=18, min_length=100, max_length=600)
right_lower_leg = LimbStats(
    start=14, end=16, mid1=21, mid2=22, min_length=100, max_length=600)

# limbs = []
body = [
    upper_arm_left, lower_arm_left, upper_arm_right, lower_arm_right,
    left_side, right_side, hip_lr, shoulder_lr, left_upper_leg, right_upper_leg,
    left_lower_leg, right_lower_leg
]

# ========================================
def extract_limbs(Points3d, limb_stats, scale2mm, Axs=None, Cameras=None):
        """
        """
        if len(Points3d[limb_stats.start]) == 0 or len(Points3d[limb_stats.end]) == 0 \
            or len(Points3d[limb_stats.mid2]) == 0 or len(Points3d[limb_stats.mid1]) == 0:
            return []

        start_p3d, start_values = Points3d[limb_stats.start]
        end_p3d, end_values = Points3d[limb_stats.end]
        mid2_p3d, mid2_values = Points3d[limb_stats.mid2]
        mid1_p3d, mid1_values = Points3d[limb_stats.mid1]
        start_p3d = start_p3d.copy() * scale2mm
        end_p3d = end_p3d.copy() * scale2mm
        mid1_p3d = mid1_p3d.copy() * scale2mm
        mid2_p3d = mid2_p3d.copy() * scale2mm

        n_start = len(start_p3d)
        n_end = len(end_p3d)

        limbs = []

        MAX_DIST = 9999999
        Cost = np.ones((n_start, n_end)) * MAX_DIST

        for s in range(n_start):
            s3d = start_p3d[s]
            for e in range(n_end):
                e3d = end_p3d[e]
                d = la.norm(s3d - e3d)
                if d > limb_stats.min_length and d < limb_stats.max_length:
                    difs1 = []
                    difs2 = []
                    for m3d in mid2_p3d:
                        d1 = la.norm(m3d - s3d)
                        d2 = la.norm(m3d - e3d)
                        dif = (d1 + d2) - d
                        difs2.append(dif)
                    for m3d in mid1_p3d:
                        d1 = la.norm(m3d - s3d)
                        d2 = la.norm(m3d - e3d)
                        dif = (d1 + d2) - d
                        difs1.append(dif)

                    Cost[s, e] = min(difs1) + min(difs2)
        row_ind, col_ind = linear_sum_assignment(Cost)
        for s, e in zip(row_ind, col_ind):
            if Cost[s, e] < MAX_DIST:
                s3d = start_p3d[s]/scale2mm
                e3d = end_p3d[e]/scale2mm
                limbs.append(Limb(s3d, e3d, scale2mm, limb_stats))

        # ~~~ DEBUGGING ~~~
        if Axs is not None:
            assert Cameras is not None

            for ax, cam in zip(Axs, Cameras):
                for limb in limbs:
                    s3d = np.expand_dims(limb.start3d, axis=0)
                    s2d = project(cam, np.expand_dims(limb.start3d, axis=0))
                    e2d = project(cam, np.expand_dims(limb.end3d, axis=0))

                    sx, sy = np.squeeze(s2d)
                    ex, ey = np.squeeze(e2d)
                    ax.scatter(sx, sy, s=20, color='red')
                    ax.scatter(ex, ey, s=20, color='blue')
                    ax.plot([sx, ex], [sy, ey], color='yellow')

    #     survived_limbs = []
    #     deleted_limbids = set()

    #     for i in range(len(limbs)-1):
    #         limb = limbs[i]
    #         if i in deleted_limbids:
    #             print('skip', i)
    #             continue
    #         for j in range(i+1, len(limbs)):
    #             o_limb = limbs[j]
    #             if o_limb.max_dist(limb) < 90:
    #                 deleted_limbids.add(j)
    #                 limb = o_limb.merge(limb)

    #                 print('o', o_limb.start3d, o_limb.end3d)
    #                 print('m', limb.start3d, limb.end3d)

    #                 print("MERGE", j, i)
    #         survived_limbs.append(limb)

    #     if Axs is not None:
    #         assert Cameras is not None

    #         for ax, cam in zip(Axs, Cameras):
    #             for limb in survived_limbs:
    #                 s3d = np.expand_dims(limb.start3d, axis=0)
    #                 s2d = project(cam, np.expand_dims(limb.start3d, axis=0))
    #                 e2d = project(cam, np.expand_dims(limb.end3d, axis=0))

    #                 sx, sy = np.squeeze(s2d)
    #                 ex, ey = np.squeeze(e2d)
    #                 ax.scatter(sx, sy, s=20, color='orange')
    #                 ax.scatter(ex, ey, s=20, color='cornflowerblue')
    #                 ax.plot([sx, ex], [sy, ey], color='green')

    #     return survived_limbs
        return limbs

import matplotlib.pyplot as plt

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


class Pose:
    def __init__(self):
        self.u_arm_left = None
        self.u_arm_right = None
        self.l_arm_left = None
        self.l_arm_right = None
        self.l_side = None
        self.r_side = None
        self.u_leg_left = None
        self.u_leg_right = None
        self.l_leg_left = None
        self.l_leg_right = None
        self.hip = None
        self.shoulder = None
    
    def convert_pose_to_joints(self):
        joints = np.zeros((4, 17)) # x,y,z,visibitlity (1: visible, 0: unvisible)
        if self.u_arm_left != None:
            joints[3,5] = 1
            joints[3,7] = 1
            joints[0:3,5] = self.u_arm_left.start3d
            joints[0:3,7] = self.u_arm_left.end3d
        
        if self.l_arm_left != None:
            joints[3,7] = 1
            joints[3,9] = 1
            joints[0:3,7] = self.l_arm_left.start3d
            joints[0:3,9] = self.l_arm_left.end3d
        
        if self.u_arm_right != None:
            joints[3,6] = 1
            joints[3,8] = 1
            joints[0:3,6] = self.u_arm_right.start3d
            joints[0:3,8] = self.u_arm_right.end3d

        if self.l_arm_right != None:
            joints[3,8] = 1
            joints[3,10] = 1
            joints[0:3,8] = self.l_arm_right.start3d
            joints[0:3,10] = self.l_arm_right.end3d
        
        if self.l_side != None:
            joints[3,5] = 1
            joints[3,11] = 1
            joints[0:3,5] = self.l_side.start3d
            joints[0:3,11] = self.l_side.end3d

        if self.r_side != None:
            joints[3,6] = 1
            joints[3,12] = 1
            joints[0:3,6] = self.r_side.start3d
            joints[0:3,12] = self.r_side.end3d

        if self.hip != None:
            joints[3,11] = 1
            joints[3,12] = 1
            joints[0:3,11] = self.hip.start3d
            joints[0:3,12] = self.hip.end3d
        
        if self.shoulder != None:
            joints[3,5] = 1
            joints[3,6] = 1
            joints[0:3,5] = self.shoulder.start3d
            joints[0:3,6] = self.shoulder.end3d

        if self.u_leg_left != None:
            joints[3,11] = 1
            joints[3,13] = 1
            joints[0:3,11] = self.u_leg_left.start3d
            joints[0:3,13] = self.u_leg_left.end3d
        
        if self.u_leg_right != None:
            joints[3,12] = 1
            joints[3,14] = 1
            joints[0:3,12] = self.u_leg_right.start3d
            joints[0:3,14] = self.u_leg_right.end3d
        
        if self.l_leg_left != None:
            joints[3,13] = 1
            joints[3,15] = 1
            joints[0:3,13] = self.l_leg_left.start3d
            joints[0:3,15] = self.l_leg_left.end3d

        if self.l_leg_right != None:
            joints[3,14] = 1
            joints[3,16] = 1
            joints[0:3,14] = self.l_leg_right.start3d
            joints[0:3,16] = self.l_leg_right.end3d
        
        return joints

    def plot(self, ax, cam, color):
        self.color = color
        if self.u_arm_left != None:
            self.plot_limb(ax, cam, self.u_arm_left)
        if self.u_arm_right != None:
            self.plot_limb(ax, cam, self.u_arm_right)
        if self.l_arm_left != None:
            self.plot_limb(ax, cam, self.l_arm_left)
        if self.l_arm_right != None:
            self.plot_limb(ax, cam, self.l_arm_right)
        if self.l_side != None:
            self.plot_limb(ax, cam, self.l_side)
        if self.r_side != None:
            self.plot_limb(ax, cam, self.r_side)
        if self.u_leg_left != None:
            self.plot_limb(ax, cam, self.u_leg_left)
        if self.u_leg_right != None:
            self.plot_limb(ax, cam, self.u_leg_right)
        if self.l_leg_left != None:
            self.plot_limb(ax, cam, self.l_leg_left)
        if self.l_leg_right != None:
            self.plot_limb(ax, cam, self.l_leg_right)
        if self.hip != None:
            self.plot_limb(ax, cam, self.hip)
        if self.shoulder != None:
            self.plot_limb(ax, cam, self.shoulder)
    
    def plot_limb(self, ax, cam, limb):
        s2d = project(cam, np.expand_dims(limb.start3d, axis=0))
        e2d = project(cam, np.expand_dims(limb.end3d, axis=0))
        sx, sy = np.squeeze(s2d)
        ex, ey = np.squeeze(e2d)
        ax.scatter(sx, sy, s=20, color='white')
        ax.scatter(ex, ey, s=20, color='white')
        ax.plot([sx, ex], [sy, ey], c=self.color)
    
    def count_limbs(self):
        count = 0
        if self.u_arm_left != None:
            count += 1
        if self.u_arm_right != None:
            count += 1
        if self.l_arm_left != None:
            count += 1
        if self.l_arm_right != None:
            count += 1
        if self.l_side != None:
            count += 1
        if self.r_side != None:
            count += 1
        if self.u_leg_left != None:
            count += 1
        if self.u_leg_right != None:
            count += 1
        if self.l_leg_left != None:
            count += 1
        if self.l_leg_right != None:
            count += 1
        if self.hip != None:
            count += 1
        if self.shoulder != None:
            count += 1
        return count
        


def extract_poses(Points3d, scale2mm, merge_distance_mm=60):
    """pass"""
    global upper_arm_left, lower_arm_left, upper_arm_right, lower_arm_right
    global left_side, right_side, hip_lr, shoulder_lr, left_upper_leg, right_upper_leg
    global left_lower_leg, right_lower_leg
    
    lower_arm_left_ = extract_limbs(Points3d, lower_arm_left, scale2mm=scale2mm)
    upper_arm_left_ = extract_limbs(Points3d, upper_arm_left, scale2mm=scale2mm)
    upper_arm_right_ = extract_limbs(Points3d, upper_arm_right, scale2mm=scale2mm)
    lower_arm_right_ = extract_limbs(Points3d, lower_arm_right, scale2mm=scale2mm)
    left_side_ = extract_limbs(Points3d, left_side, scale2mm=scale2mm)
    right_side_ = extract_limbs(Points3d, right_side, scale2mm=scale2mm)
    hip_lr_ = extract_limbs(Points3d, hip_lr, scale2mm=scale2mm)
    shoulder_lr_ = extract_limbs(Points3d, shoulder_lr, scale2mm=scale2mm)
    left_upper_leg_ = extract_limbs(Points3d, left_upper_leg, scale2mm=scale2mm)
    right_upper_leg_ = extract_limbs(Points3d, right_upper_leg, scale2mm=scale2mm)
    left_lower_leg_ = extract_limbs(Points3d, left_lower_leg, scale2mm=scale2mm)
    right_lower_leg_ = extract_limbs(Points3d, right_lower_leg, scale2mm=scale2mm)
    
    Poses = []
    # == LEFT ARM ==    
    for limb in lower_arm_left_:
        pose = Pose()
        pose.l_arm_left = limb
        Poses.append(pose)
    
    new_Poses = []
    for limb in upper_arm_left_:
        pt1 = limb.end3d
        distances = []
        for pose in Poses:
            pt2 = pose.l_arm_left.start3d
            d = la.norm(pt1 - pt2) * scale2mm
            distances.append(d)
        
        best = np.argmin(distances)
        best_val = distances[best]
        if merge_distance_mm > best_val:
            best_pose = Poses[best]
            # assert best_pose.u_arm_left is None
            best_pose.u_arm_left = limb
        else:
            pose = Pose()
            pose.u_arm_left = limb
            new_Poses.append(pose)
            
    Poses = Poses + new_Poses
    
    # == SHOULDER ==
    new_Poses = []
    for shoulder in shoulder_lr_:
        pt1 = shoulder.start3d
        distances = []
        for pose in Poses:
            if pose.u_arm_left is None:
                distances.append(99999999)
                continue
            pt2 = pose.u_arm_left.start3d
            d = la.norm(pt1 - pt2) * scale2mm
            distances.append(d)
        
        best = np.argmin(distances)
        best_val = distances[best]
        if merge_distance_mm > best_val:
            best_pose = Poses[best]
            best_pose.shoulder = shoulder
        else:
            pose = Pose()
            pose.shoulder = shoulder
            new_Poses.append(pose)
    Poses = Poses + new_Poses
    
    # == RIGHT ARM ==
    new_Poses = []
    for limb in upper_arm_right_:
        pt1 = limb.start3d
        distances = []
        for pose in Poses:
            if pose.shoulder is None:
                distances.append(99999999)
                continue
            pt2 = pose.shoulder.end3d
            d = la.norm(pt1 - pt2) * scale2mm
            distances.append(d)
        
        best = np.argmin(distances)
        best_val = distances[best]
        if merge_distance_mm > best_val:
            best_pose = Poses[best]
            # assert best_pose.u_arm_right is None
            best_pose.u_arm_right = limb
        else:
            pose = Pose()
            pose.u_arm_right = limb
            new_Poses.append(pose)
    
    Poses = Poses + new_Poses
    new_Poses = []
    for limb in lower_arm_right_:
        pt1 = limb.start3d
        distances = []
        for pose in Poses:
            if pose.u_arm_right is None:
                distances.append(99999999)
                continue
            pt2 = pose.u_arm_right.end3d
            d = la.norm(pt1 - pt2) * scale2mm
            distances.append(d)
        
        best = np.argmin(distances)
        best_val = distances[best]
        if merge_distance_mm > best_val:
            best_pose = Poses[best]
            # assert best_pose.l_arm_right is None
            best_pose.l_arm_right = limb
        else:
            pose = Pose()
            pose.l_arm_right = limb
            new_Poses.append(pose)
            
    Poses = Poses + new_Poses
    
    # == LEFT BODY SIDE ==
    new_Poses = []
    for limb in left_side_:
        pt1 = limb.start3d
        distances = []
        for pose in Poses:
            if pose.shoulder is None:
                distances.append(99999999)
                continue
            pt2 = pose.shoulder.start3d
            d = la.norm(pt1 - pt2) * scale2mm
            distances.append(d)
        
        best = np.argmin(distances)
        best_val = distances[best]
        if merge_distance_mm > best_val:
            best_pose = Poses[best]
            # assert best_pose.l_side is None
            best_pose.l_side = limb
        else:
            pose = Pose()
            pose.l_side = limb
            new_Poses.append(pose)
            
    Poses = Poses + new_Poses
    
    # == RIGHT BODY SIDE ==
    new_Poses = []
    for limb in right_side_:
        pt1 = limb.start3d
        distances = []
        for pose in Poses:
            if pose.shoulder is None:
                distances.append(99999999)
                continue
            pt2 = pose.shoulder.end3d
            d = la.norm(pt1 - pt2) * scale2mm
            distances.append(d)
        
        best = np.argmin(distances)
        best_val = distances[best]
        if merge_distance_mm > best_val:
            best_pose = Poses[best]
            # assert best_pose.r_side is None
            best_pose.r_side = limb
        else:
            pose = Pose()
            pose.r_side = limb
            new_Poses.append(pose)
            
    Poses = Poses + new_Poses
    
    # == HIP ==
    new_Poses = []
    for limb in hip_lr_:  
        pt1_left = limb.start3d
        pt1_right = limb.end3d
        
        distances = []
        for pose in Poses:
            if pose.l_side is None and pose.r_side is None:
                distances.append(99999999)
                continue
            elif pose.l_side is None:
                pt2 = pose.r_side.end3d
                d = la.norm(pt1_right - pt2) * scale2mm
                distances.append(d)
            elif pose.r_side is None:
                pt2 = pose.l_side.end3d
                d = la.norm(pt1_left - pt2) * scale2mm
                distances.append(d)
            else:
                pt2_right = pose.r_side.end3d
                d_right = la.norm(pt1_right - pt2_right) * scale2mm
                pt2_left = pose.l_side.end3d
                d_left = la.norm(pt1_left - pt2_left) * scale2mm
                d = (d_left+d_right)/2
                distances.append(d * 0.7)
        best = np.argmin(distances)
        best_val = distances[best]
        if merge_distance_mm > best_val:
            best_pose = Poses[best]
            # assert best_pose.hip is None
            best_pose.hip = limb
        else:
            pose = Pose()
            pose.hip = limb
            new_Poses.append(pose)
            
    Poses = Poses + new_Poses
    
    # == LEFT LEG ==
    new_Poses = []
    for limb in left_upper_leg_:
        pt1 = limb.start3d
        distances = []
        for pose in Poses:
            if pose.hip is None:
                distances.append(99999999)
                continue
            pt2 = pose.hip.start3d
            d = la.norm(pt1 - pt2) * scale2mm
            distances.append(d)
        
        best = np.argmin(distances)
        best_val = distances[best]
        if merge_distance_mm > best_val:
            best_pose = Poses[best]
            # assert best_pose.u_leg_left is None
            best_pose.u_leg_left = limb
        else:
            pose = Pose()
            pose.u_leg_left = limb
            new_Poses.append(pose)
    Poses = Poses + new_Poses
    
    new_Poses = []
    for limb in left_lower_leg_:
        pt1 = limb.start3d
        distances = []
        for pose in Poses:
            if pose.u_leg_left is None:
                distances.append(99999999)
                continue
            pt2 = pose.u_leg_left.end3d
            d = la.norm(pt1 - pt2) * scale2mm
            distances.append(d)
        
        best = np.argmin(distances)
        best_val = distances[best]
        if merge_distance_mm > best_val:
            best_pose = Poses[best]
            # assert best_pose.l_leg_left is None
            best_pose.l_leg_left = limb
        else:
            pose = Pose()
            pose.l_leg_left = limb
            new_Poses.append(pose)
    Poses = Poses + new_Poses
    
    # == RIGHT LEG ==
    new_Poses = []
    for limb in right_upper_leg_:
        pt1 = limb.start3d
        distances = []
        for pose in Poses:
            if pose.hip is None:
                distances.append(99999999)
                continue
            pt2 = pose.hip.end3d
            d = la.norm(pt1 - pt2) * scale2mm
            distances.append(d)
        
        best = np.argmin(distances)
        best_val = distances[best]
        if merge_distance_mm > best_val:
            best_pose = Poses[best]
            # assert best_pose.u_leg_right is None
            best_pose.u_leg_right = limb
        else:
            pose = Pose()
            pose.u_leg_right = limb
            new_Poses.append(pose)
    Poses = Poses + new_Poses
    
    new_Poses = []
    for limb in right_lower_leg_:
        pt1 = limb.start3d
        distances = []
        for pose in Poses:
            if pose.u_leg_right is None:
                distances.append(99999999)
                continue
            pt2 = pose.u_leg_right.end3d
            d = la.norm(pt1 - pt2) * scale2mm
            distances.append(d)
        
        best = np.argmin(distances)
        best_val = distances[best]
        if merge_distance_mm > best_val:
            best_pose = Poses[best]
            # assert best_pose.l_leg_right is None
            best_pose.l_leg_right = limb
        else:
            pose = Pose()
            pose.l_leg_right = limb
            new_Poses.append(pose)
    Poses = Poses + new_Poses

    final_poses = []
    # discard poses has less than 5 limbs
    for pose in Poses:
        if pose.count_limbs() > 5:
            final_poses.append(pose)

    return final_poses

def calculate_pckh3d(gt_poses, est_poses):
    tp = np.zeros(17)
    for i, gt_pose in enumerate(gt_poses):
        head_segment_length = np.linalg.norm(gt_pose[:,1] - gt_pose[:,2])
        diff = est_poses[i] - gt_pose
        diff = np.linalg.norm(diff, axis=0)
        tp += diff <= head_segment_length 
    
    return tp

def find_nearest_pose(gt_joints, estimated_poses):
    num_person = gt_joints.shape[0]
    cost_mat = np.zeros((num_person, len(estimated_poses)))
    pred_joints = np.zeros((len(estimated_poses), 4, 17))
    for id, pose in enumerate(estimated_poses):
        pred_joints[id,:,:] = pose.convert_pose_to_joints()
    
    for i, gt_joint in enumerate(gt_joints):
        for j, pred_joint in enumerate(pred_joints):
            vis_joint = pred_joint[3]>0
            diff = gt_joint[0:3] - pred_joint[0:3]
            diff = diff[:,vis_joint]
            diff = np.linalg.norm(diff,axis=0)
            cost_mat[i,j] = np.sum(diff)/np.sum(vis_joint)
    
    gt_ind, est_ind = linear_sum_assignment(cost_mat)
    return gt_joints[gt_ind] ,pred_joints[est_ind]


        