import copy
from easydict import EasyDict as edict

def get_default_image_cfg():
    cfg = edict()
    cfg.use_perceiver = False
    cfg.use_visual_adapter = False
    cfg.visual_modality_type = "image"
    cfg.perceiver_cfg = None
    cfg.visual_adapter_cfg = None
    
    return cfg


def set_default_image_cfg(cfg):
    _default_dict={
        "use_perceiver": False,
        "use_visual_adapter": False,
        "visual_modality_type": "image",
        "perceiver_cfg": None,
        "visual_adapter_cfg": None,
        "unlock_cls": False,
        "skip_trans_first_n_layers": None,
        "unlock_trans_first_n_layers": None,
    }
    cfg_ = copy.deepcopy(cfg)
    if isinstance(cfg_, dict):
        cfg_.update(_default_dict)
    else:
        for k,v in _default_dict.items():
            if hasattr(cfg_, k):
                setattr(cfg_, k, v)
    return cfg_
    

def get_perceiver_cfg(args):
    cfg = edict()
    cfg.use_perceiver = args.use_perceiver
    cfg.input_chan = args.perceiver_input_chan
    cfg.input_axis = args.perceiver_input_axis
    cfg.num_freq_bands = args.perceiver_num_freq_bands
    cfg.max_freq = args.perceiver_max_freq
    cfg.depth = args.perceiver_depth
    cfg.num_latents = args.perceiver_num_latents
    cfg.latent_dim = args.perceiver_latent_dim
    cfg.cross_heads = args.perceiver_cross_heads
    cfg.latent_heads = args.perceiver_latent_heads
    cfg.cross_dim_head = args.perceiver_cross_dim_head
    cfg.latent_dim_head = args.perceiver_latent_dim_head
    cfg.num_classes = args.perceiver_num_classes
    cfg.attn_dropout = args.perceiver_attn_dropout
    cfg.ff_dropout = args.perceiver_ff_dropout
    cfg.weight_tie_layers = args.perceiver_weight_tie_layers
    cfg.fourier_encode_data = args.perceiver_fourier_encode_data
    cfg.self_per_cross_attn = args.perceiver_self_per_cross_attn
    
    return cfg


def get_input_adapter_cfg(args):
    cfg = edict()
    cfg.use_visual_adapter = args.use_visual_adapter
    cfg.visual_modality_type = args.visual_modality_type
    cfg.disable_orig_pos = args.disable_orig_pos
    
    # case: visual modality
    if args.visual_modality_type == "image":
        pass
    elif args.visual_modality_type == "3dpc":
        cfg.pc_tokenizer = args.pc_tokenizer
        cfg.trans_dim = args.pc_trans_dim
        cfg.group_size = args.pc_group_size
        cfg.num_group = args.pc_num_group
        cfg.encoder_dims = args.pc_encoder_dims
        cfg.radius = args.pc_radius
        cfg.in_dim = args.pc_in_channel
        
    elif args.visual_modality_type == "3dpc_raw":
        pass
    else:
        raise NotImplementedError
        
    return cfg


def get_pointbert_cfg(args):
    cfg = edict()
    cfg.trans_dim = args.pc_trans_dim
    cfg.depth = args.pointbert_depth
    cfg.drop_path_rate = args.pointbert_drop_path_rate
    cfg.num_heads = args.pointbert_num_heads
    cfg.group_size = args.pc_group_size
    cfg.num_group = args.pc_num_group
    cfg.encoder_dims = args.pc_encoder_dims
    cfg.do_cat = not args.pointbert_disable_cat
    
    return cfg

