import os
import math
import torch
import torch.nn as nn

def assign_visual_adapter(cfg, module, **kwargs):
    vtype = cfg.visual_modality_type
    
    if vtype == "3dpc":
        if cfg.pc_tokenizer == "pointbert":
            from open_clip.modal_3d.models.pointbert.point_encoder import PointTokenizer
            module.visual_adapter = PointTokenizer(config=cfg)
        elif cfg.pc_tokenizer == "pnsa":
            from open_clip.modal_3d.models.pointnet.pointnet_util import PointNSATokenizer
            module.visual_adapter = PointNSATokenizer(config=cfg)
        else:
            raise NotImplementedError
        
    elif vtype == "3dpc_raw":
        module.visual_adapter = nn.Identity()
    
    else:
        raise NotImplementedError
    
    return


def get_visual_adapter(cfg, **kwargs):
    vtype = cfg.visual_modality_type
    
    if vtype == "3dpc":
        if cfg.pc_tokenizer == "pointbert":
            from open_clip.modal_3d.models.pointbert.point_encoder import PointTokenizer
            return PointTokenizer(config=cfg)
        elif cfg.pc_tokenizer == "pnsa":
            from open_clip.modal_3d.models.pointnet.pointnet_util import PointNSATokenizer
            return PointNSATokenizer(config=cfg)
        else:
            NotImplementedError
              
    elif vtype == "3dpc_raw":
        return nn.Identity()
    
    else:
        raise NotImplementedError