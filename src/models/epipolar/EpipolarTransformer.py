import torch
import cv2
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from core import cfg
from utils.multiview import normalize, findFundamentalMat, de_normalize, coord2pix  

visualize_prob = 0.1

class Epipolar(nn.Module):
    def __init__(self, debug=False):
        super(Epipolar, self).__init__()
        self.debug = debug
        
        self.feat_h, self.feat_w = cfg.KEYPOINT.HEATMAP_SIZE
        self.sample_size = cfg.EPIPOLAR.SAMPLESIZE
        self.epsilon = 0.001 # for avoiding floating point error

        y = torch.arange(0, self.feat_h, dtype=torch.float) # 0 .. 128
        x = torch.arange(0, self.feat_w, dtype=torch.float) # 0 .. 84

        grid_y, grid_x = torch.meshgrid(y, x)
        self.grid = torch.stack((grid_x, grid_y, torch.ones_like(grid_x))).view(3, -1)

        self.xmin = x[0]
        self.ymin = y[0]
        self.xmax = x[-1]
        self.ymax = y[-1]

        self.sample_steps = torch.range(0, 1, 1./(self.sample_size-1)).view(-1, 1, 1, 1)
        self.tmp_tensor = torch.tensor([True, True, False, False])
        self.outrange_tensor = torch.tensor([
            self.xmin-10000, self.ymin-10000, 
            self.xmin-10000, self.ymin-10000]).view(2, 2)

    def forward(self, feat1, feat2, P1, P2, img1=None, img2=None, keypt1=None, keypt2=None):
        """ 
        Args:
            feat1         : N x C x H x W
            feat2         : N x C x H x W
            P1          : N x 3 x 4
            P2          : N x 3 x 4
        1. Compute epipolar lines: NHW x 3 (http://users.umiacs.umd.edu/~ramani/cmsc828d/lecture27.pdf)
        2. Compute intersections with the image: NHW x 2 x 2
            4 intersections with each boundary of the image NHW x 4 x 2
            Convert to (-1, 1)
            find intersections on the rectangle NHW x 4 T/F, NHW x 2 x 2
            sample N*sample_size x H x W x 2
                if there's no intersection, the sample points are out of (-1, 1), therefore ignored by pytorch
        3. Sample points between the intersections: sample_size x N x H x W x 2
        4. grid_sample: sample_size*N x C x H x W -> sample_size x N x C x H x W
            trick: compute feat1 feat2 dot product first: N x HW x H x W
        5. max pooling/attention: N x C x H x W
        """

        assert feat1.size(0) == feat2.size(0)
        assert feat1.size(1) == feat2.size(1)
        assert feat1.size(2) == feat2.size(2)
        assert feat1.size(3) == feat2.size(3)
        N,C,H,W = feat1.size()
        
        ref_view = feat1
        src_view = feat2.expand(self.sample_size, N, C, H, W)       
         
        sample_locs = self.grid2sample_locs(self.grid, P1, P2, H, W)
        sample_locs = sample_locs.to(src_view)

        batch = []
        corr_pos = []
        for i in range(N):
            src_sampled = F.grid_sample(src_view[:,i]  , sample_locs[:, i])
            sim = self.epipolar_similarity(ref_view[i], src_sampled)
            idx = sim.argmax(0)
            with torch.no_grad():
                    # H x W x 2
                    pos = torch.gather(sample_locs[:, i], 0, idx.view(1, H, W, 1).expand(-1, -1, -1, 2)).squeeze()
                    pos = de_normalize(pos, H, W)
                    corr_pos.append(pos)
            if self.debug:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(12, 8))
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                img1_resized = cv2.resize(cv2.cvtColor(img1[i].cpu().permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB),(64,64))
                img2_resized = cv2.resize(cv2.cvtColor(img2[i].cpu().permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB),(64,64))
                ax1.imshow(img1_resized/255)
                ax2.imshow(img2_resized/255)
                ax1.set_ylim([64, 0])
                ax1.set_xlim([0, 64])
                ax2.set_ylim([64, 0])
                ax2.set_xlim([0, 64])
                for j in range(0,17):
                    cx, cy = int(coord2pix(keypt1[i,2,j,0], 1)), int(coord2pix(keypt1[i,2,j,1], 1))
                    xy = corr_pos[0][cy,cx].cpu().numpy()

                    line_start1 = de_normalize(sample_locs[1,0][int(cy)][int(cx)], H, W)
                    line_start2 = de_normalize(sample_locs[127,0][int(cy)][int(cx)], H, W)
                    
                    ax1.scatter(cx,cy,color='yellow')
                    ax2.plot([line_start1[0], line_start2[0]], [line_start1[1], line_start2[1]], alpha=0.5, color='b', zorder=1)
                    ax2.scatter(xy[0],xy[1],color='red')
                plt.show()

            idx = idx.view(1, 1, H, W).expand(-1, C, -1, -1)
            # C x H x W
            tmp = src_sampled.max(dim=0, keepdim=True)[0].squeeze()
            # tmp = (src_sampled * sim.view(-1, 1, H, W)).sum(0)
            batch.append(tmp)
        out = torch.stack(batch)
        return out
    
    def epipolar_similarity(self, feat1, sampled_feat2, epipolar_similarity = 'cos', epipolar_attension = 'avg'):
        """ 
        Args:
            fea1: C, H, W
            sampled_feat2: sample_size, C, H, W
        Return:
            sim: sample_size H W
        """
    
        C, H, W = feat1.shape
        sample_size = sampled_feat2.shape[0]
        if epipolar_attension == 'max':
            # sample_size H W
            sim = F.cosine_similarity(
                feat1.view(1, C, H, W).expand(sample_size, -1, -1, -1),
                sampled_feat2, 1)
        elif epipolar_attension == 'avg':
            if epipolar_similarity == 'prior':
                return self.prior[(cam1, cam2)].to(feat1)
            elif epipolar_similarity == 'cos':
                sim = F.cosine_similarity(
                    feat1.view(1, C, H, W).expand(sample_size, -1, -1, -1),
                    sampled_feat2, 1)
            elif epipolar_similarity == 'dot':
                sim = (sampled_feat2 * feat1.view(1, C, H, W).expand(sample_size, -1, -1, -1)).sum(1)
            else:
                raise NotImplementedError
            sim[sim==0] = -1e10

            sim /= sample_size
            
        return sim

    def grid2sample_locs(self, grid, P1, P2, H, W):
        """ 
        Args:
            grid: 3 x HW, from the reference view
        Return:
            sample_locs: sample_size x N x H x W x 2, float xy (-1, 1)
        """
        N = P1.shape[0]

        F_mat = findFundamentalMat(P1, P2)
        grid = torch.unsqueeze(grid, dim=0).to(F_mat)

        l2 = torch.matmul(F_mat,grid)
        l2 = l2.transpose(1, 2)

        xmin = self.xmin.to(l2)
        xmax = self.xmax.to(l2)
        ymin = self.ymin.to(l2)
        ymax = self.ymax.to(l2)

        #numerical stability
        EPS = torch.tensor(self.epsilon).to(l2)
        by1 = -(xmin * l2[..., 0] + l2[..., 2]) / (torch.sign(l2[..., 1]) * torch.max(torch.abs(l2[..., 1]), EPS))
        by2 = -(xmax * l2[..., 0] + l2[..., 2]) / (torch.sign(l2[..., 1]) * torch.max(torch.abs(l2[..., 1]), EPS))
        bx0 = -(ymin * l2[..., 1] + l2[..., 2]) / (torch.sign(l2[..., 0]) * torch.max(torch.abs(l2[..., 0]), EPS))
        bx3 = -(ymax * l2[..., 1] + l2[..., 2]) / (torch.sign(l2[..., 0]) * torch.max(torch.abs(l2[..., 0]), EPS))
        # N x HW x 4
        intersections = torch.stack((
            bx0,
            by1,
            by2,
            bx3,
            ), -1)
        # N x HW x 4 x 2
        intersections = intersections.view(N, H*W, 4, 1).repeat(1, 1, 1, 2)
        intersections[..., 0, 1] = ymin
        intersections[..., 1, 0] = xmin
        intersections[..., 2, 0] = xmax
        intersections[..., 3, 1] = ymax
        # N x HW x 4
        mask = torch.stack((
            (bx0 >= xmin + self.epsilon) & (bx0 <  xmax - self.epsilon),
            (by1 >  ymin + self.epsilon) & (by1 <= ymax - self.epsilon),
            (by2 >= ymin + self.epsilon) & (by2 <  ymax - self.epsilon),
            (bx3 >  xmin + self.epsilon) & (bx3 <= xmax - self.epsilon),
            ), -1)
        # N x HW
        Nintersections = mask.sum(-1)
        # rule out all lines have no intersections
        mask[Nintersections < 2] = 0
        tmp_mask = mask.clone()
        tmp_mask[Nintersections < 2] = self.tmp_tensor.to(tmp_mask)
        # assert (Nintersections <= 2).all().item(), intersections[Nintersections > 2]
        # N x HW x 2 x 2
        valid_intersections = intersections[tmp_mask].view(N, H*W, 2, 2)
        valid_intersections[Nintersections < 2] = self.outrange_tensor.to(valid_intersections)
        # N x HW x 2
        start = valid_intersections[..., 0, :]
        vec = valid_intersections[..., 1, :] - start
        vec = vec.view(1, N, H*W, 2)
        # sample_size x N x HW x 2
        sample_locs = start.view(1, N, H*W, 2) + vec * self.sample_steps.to(vec)
        # sample_size*N x H x W x 2
        sample_locs = normalize(sample_locs, H, W).view(-1, H, W, 2)
        sample_locs = sample_locs.view(self.sample_size, N, H, W, 2)
        # if self.debug:
        #     return sample_locs, intersections, mask, valid_intersections, start, vec
        return sample_locs