import torch
import torch.nn as nn
import numpy as np
import numba

@numba.jit(nopython=True, parallel=True)
def calculate(spatial_feature, pillars, indices):
    for i in numba.prange(spatial_feature.shape[0]):
        for j in numba.prange(indices.shape[0]):
            spatial_feature[i][indices[j]] = pillars[i][j]
    return spatial_feature

#ENABLE_NUMBA=True
ENABLE_NUMBA=False
if ENABLE_NUMBA:
    numba.config.THREADING_LAYER = 'tbb'

ENABLE_NUMBA_DEBUG_PRINT=False

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        if batch_size != 1:
            raise NotImplementedError

        spatial_feature = np.zeros((self.num_bev_features,self.nz * self.nx * self.ny), dtype=np.float32)
        batch_mask = coords[:, 0] == 0
        this_coords = coords[batch_mask, :]
        indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
        
        indices = indices.type(torch.long)
        # print(f'indices : {indices}')
        batchmask = batch_mask.numpy()
        len_mask = batchmask.shape[0]
        if len_mask < 100:
            batchmask_pad = np.pad(batchmask, (0,100-len_mask), 'constant',constant_values=False)
        else:
            batchmask_pad = batchmask[:100]
        batch_mask_tensor = torch.from_numpy(batchmask_pad)
        pillars = pillar_features[batch_mask_tensor, :]
        pillars = pillars.t()
        # print(f'pillars shape ; {pillars.shape}')

        if ENABLE_NUMBA is True:
            spatial_feature = calculate(spatial_feature, pillars.numpy(), indices.numpy())
            spatial_feature = torch.from_numpy(spatial_feature)
            if ENABLE_NUMBA_DEBUG_PRINT is True:
                print("Threading layer chosen: %s" % numba.threading_layer())
                calculate.parallel_diagnostics(level=4)
                exit(0)
        else:
            spatial_feature = torch.from_numpy(spatial_feature)
            # spatial_feature[:, indices] = pillars
            spatial_feature[:, indices[:100]] = pillars

        batch_spatial_features = spatial_feature.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
