import argparse
import ast


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_tower",
        type=int,
        default=2,
        help="number of towers in the model arch, default to be 2 in the orginal CLIP model."
    )
    
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto", "3dpc", "video", "video-text"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        default=False,
        help="whether save model ckpt with the best metric."
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
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
        "--load_ckpt_strict",
        default=False,
        action='store_true',
        help="whether load original ckpt strictly."
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
        "--visual-stat-flops",
        default=False,
        action="store_true",
        help="caculate the flops of the model"
    )
    parser.add_argument(
        "--unlock-cls",
        default=False,
        action="store_true",
        help="specify whether unlock cls when lock visual. default to be False"
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
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
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
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    # arguments for distributed training
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
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
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
        "--log-every-n-steps",
        type=int,
        default=10,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/path/to/ckpt/pretrained",
        help="cache dir for model",
    )
    
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    
    # add by leiwx
    parser.add_argument(
        "--v_key",
        choices=["image", "pc"],
        default="pc",
        help="visual key for fetching data in batch",
        
    )
    # ===== specific prompt template =====
    parser.add_argument(
        "--disable_orig_pos",
        default=False,
        action="store_true",
        help="whether disable original pos encoding"
    )
    
    parser.add_argument(
        "--train_data_prompt",
        default=None,
        type=str,
        help='Prompt template for training set, if any'
    )
    parser.add_argument(
        "--val_data_prompt",
        default=None,
        type=str,
        help='Prompt template for val set, if any'
    )
    # ===== end of specific prompt template =====

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
        default="image",
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
    
    #==
    parser.add_argument(
        "--use_eva_pt_lin",
        default=False,
        action="store_true",
        help="whether use eva clip pt linear weight for binding."
    )

    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
