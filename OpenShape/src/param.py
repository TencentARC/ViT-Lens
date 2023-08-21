import argparse

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/train.yaml",
    )
    parser.add_argument(
        "--horovod",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--trial_name",
        type=str,
        default="try",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="./exp",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--train",
        default=True,
        action="store_true",
        help="train a model."
    )
    parser.add_argument(
        "--resume", 
        default=None, 
        help="path to the weights to be resumed"
    )
    parser.add_argument(
        "--autoresume",
        default=False,
        action="store_true",
        help="auto back-off on failure"
    )
    parser.add_argument(
        "--ngpu", 
        default=1, 
        type=int,
        help="number of gpu used"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/path/to/ckpt/pretrained",
        help="cache dir for model",
    ) 
    parser.add_argument(
        "--clip-model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--lock-visual",
        default=False,
        action='store_true',
        help="Lock full visual tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-visual-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n visual tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-visual-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in visual tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--disable_orig_pos",
        default=False,
        action="store_true",
        help="whether disable original pos encoding"
    )
    parser.add_argument(
        "--disable_pt_vit",
        default=False,
        action="store_true",
        help="whether disable using pretrained ViT model init."
    )
    parser.add_argument(
        "--visual_arch",
        default="perceiver_vit",
        type=str,
        help="visual model arch [name]"
    )
    parser.add_argument(
        "--use_eva_pt_lin",
        default=False,
        action="store_true",
        help="whether use eva clip pt linear weight for binding."
    )
    # ===== Config for Perceiver =====
    parser.add_argument(
        "--use_perceiver",
        default=False,
        action="store_true",
        help="whether use perceiver",
    )
    parser.add_argument(
        "--perceiver_as_identity",
        action="store_true",
        default=False,
        help="use for abalation. whether use identity for perceiver."
    )
    parser.add_argument(
        "--perceiver_input_chan",
        default=384,
        type=int,
        help="number of channels for each token of the input"
    )
    parser.add_argument(
        "--perceiver_input_axis",
        default=1,
        type=int,
        help="number of axis for input data (1 for plain seq, 2 for images, 3 for video)"
    )
    parser.add_argument(
        "--perceiver_num_freq_bands",
        default=32,
        type=int,
        help="number of freq bands, with original value (2 * K + 1)"
    )
    parser.add_argument(
        "--perceiver_max_freq",
        default=10.,
        type=float,
        help="maximum frequency, hyperparameter depending on how fine the data is"
    )
    parser.add_argument(
        "--perceiver_depth",
        default=4,
        type=int,
        help="depth of net. The shape of the final attention mechanism will be: epth * (cross attention -> self_per_cross_attn * self attention)"
    )
    parser.add_argument(
        "--perceiver_num_latents",
        default=196,
        type=int,
        help="number of latents, or induced set points, or centroids. different papers giving it different names"
    )
    parser.add_argument(
        "--perceiver_latent_dim",
        default=768,
        type=int,
        help="perceiver latent dimension"
    )
    parser.add_argument(
        "--perceiver_cross_heads",
        default=1,
        type=int,
        help="number of heads for cross attention. paper said 1"
    )
    parser.add_argument(
        "--perceiver_latent_heads",
        default=12,
        type=int,
        help="Number of heads for latent self attention, 8"
    )
    parser.add_argument(
        "--perceiver_cross_dim_head",
        default=64,
        type=int,
        help="number of dimensions per cross attention head"
    )
    parser.add_argument(
        "--perceiver_latent_dim_head",
        default=64,
        type=int,
        help="number of dimensions per latent self-attention head"
    )
    parser.add_argument(
        "--perceiver_num_classes",
        default=1000,
        type=int,
        help="number of dimensions per latent self-attention head"
    )
    parser.add_argument(
        "--perceiver_attn_dropout",
        default=0.,
        type=float,
    )
    parser.add_argument(
        "--perceiver_ff_dropout",
        default=0.,
        type=float,
    )
    parser.add_argument(
        "--perceiver_weight_tie_layers",
        default=False,
        action="store_true",
        help="whether to weight tie layers (optional, as indicated in the diagram)"
    )
    parser.add_argument(
        "--perceiver_fourier_encode_data",
        default=False,
        action="store_true",
        help="whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself"
    )
    parser.add_argument(
        "--perceiver_self_per_cross_attn",
        default=6,
        type=int,
        help="number of self attention blocks per cross attention"
    )
    # ====== end of perceiver config
    
    parser.add_argument(
        "--visual_modality_type",
        default="3dpc",
        type=str,
        help="type of visual modality used for exp"
    )
    parser.add_argument(
        "--use_visual_adapter",
        default=False,
        action="store_true",
        help="whether use visual adapter"
    )
    
    # ==== 3d pc config
    parser.add_argument(
        "--pc_in_channel",
        type=int,
        default=3,
        choices=[3,6],
        help="point cloud input channels"
    )
    parser.add_argument(
        "--pc_tokenizer",
        type=str,
        default="pointbert",
        choices=["pointbert", "pnsa"]
    )
    parser.add_argument(
        "--pc_trans_dim",
        default=384,
        type=int,
        help="point cloud transformer dim, final dim of tokenizer"
    )
    parser.add_argument(
        "--pc_group_size",
        default=32,
        type=int,
        help="point cloud group size"
    )
    parser.add_argument(
        "--pc_num_group",
        default=512,
        type=int,
        help="point cloud group size"
    )
    parser.add_argument(
        "--pc_encoder_dims",
        default=256,
        type=int,
        help="point cloud tokenizer intermediate dim"
    )
    parser.add_argument(
        "--pc_npoints",
        default=None,
        type=int,
        help="number of points sampled for pc."
    )
    parser.add_argument(
        "--pc_radius",
        type=float,
        default=0.2,
        help="radius of pc for sampling."
    )
    parser.add_argument(
        "--skip-trans-first-n-layers",
        default=None,
        type=int,
        help="skip first-n-layers of the ViT transformer."
    )
    parser.add_argument(
        "--unlock-trans-first-n-layers",
        default=None,
        type=int,
        help="unlock training of first-n-layers of the ViT transformer."
    )
    parser.add_argument(
        "--unlock-cls",
        default=False,
        action="store_true",
        help="whether unlock the cls in original transformer for training."
    )

    args, extras = parser.parse_known_args()
    return args, extras