# 🚀 Training and Inference

## Note
- Most training related scripts are located under [vitlens/src/training](vitlens/src/training). 
- Evaluations across different modalities are mainly located at [vitlens/src/training/zero_shot.py](vitlens/src/training/zero_shot.py).
- For data processing, modality embedding module and dataset implementation, please refer to [vitlens/src/open_clip/modal_*](vitlens/src/open_clip/).
- For core model implementation, please refer to [vitlens/src/open_clip/transformer.py](vitlens/src/open_clip/transformer.py) and [vitlens/src/open_clip/model.py](vitlens/src/open_clip/model.py).
- For using OpenShape triplets, training/inference scripts are located under [VitLens-OpenShape](VitLens-OpenShape).



## 3D Point Cloud

<details>
  <summary>Train vitlensL on ULIP-ShapeNet-Trplets (click to expand)</summary>


```shell
cd vitlens/
# you may change the path accordingly
# train with 16 V100, total training time: ~8 hours
# You may change --accum-freq arg if using less GPUs
python -m torch.distributed.run $@ ./src/training/point_cloud/pc_tri_main.py \
    --cache_dir /path_to/cache_dir \
    --train-data shapenet --val-data modelnet40 --train_data_prompt shapenet_64 --val_data_prompt modelnet40_64 \
    --visual_modality_type 3dpc --dataset-type 3dpc --v_key pc --pc_npoints 8192 \
    --n_tower 3 \
    --use_perceiver --perceiver_cross_dim_head 64 --perceiver_latent_dim 1024 --perceiver_latent_dim_head 64 --perceiver_latent_heads 16 --perceiver_num_latents 256 --perceiver_self_per_cross_attn 1 --perceiver_weight_tie_layers \
    --use_visual_adapter \
    --batch-size 32 --accum-freq 1 \
    --model ViT-L-14 --pretrained datacomp_xl_s13b_b90k --name pc/vitlensL_ShapeNet \
    --save-frequency 1 --delete-previous-checkpoint --resume latest --save-best \
    --epochs 200
```
</details>



<details>
  <summary>Evaluate vitlensL (trained on ULIP-ShapeNet-Trplets) on ModelNet40 (click to expand)</summary>

Download [vitlensL-pc-ShapeNet](https://huggingface.co/TencentARC/ViT-Lens/blob/main/Datacomp_L14_ShapeNet.pt) checkpoint.

```shell
cd vitlens/
# you may change the path accordingly
torchrun --nproc_per_node=1 ./src/training/point_cloud/pc_tri_main.py \
    --cache_dir /path_to/cache_dir \
    --val-data modelnet40 --val_data_prompt modelnet40_64 \
    --visual_modality_type 3dpc --dataset-type 3dpc --v_key pc --pc_npoints 8192 \
    --n_tower 3 \
    --use_perceiver --perceiver_cross_dim_head 64 --perceiver_latent_dim 1024 --perceiver_latent_dim_head 64 --perceiver_latent_heads 16 --perceiver_num_latents 256 --perceiver_self_per_cross_attn 1 --perceiver_weight_tie_layers \
    --use_visual_adapter \
    --batch-size 32 \
    --model ViT-L-14 --pretrained datacomp_xl_s13b_b90k \
    --name pc/inference_vitlensL_ShapeNet \
    --resume /path_to/Datacomp_L14_ShapeNet.pt
```
</details>


<details>
  <summary>Train vitlensL on ULIP2-Objaverse-Trplets (click to expand)</summary>

```shell
cd vitlens/
# you may change the path accordingly
# train with 16 V100, total training time: ~100 hours
# You may change --accum-freq arg if using less GPUs
python -m torch.distributed.run $@  ./src/training/point_cloud/pc_tri_main.py \
  --cache_dir /path_to/cache \
  --train-data objverse --val-data modelnet40 --val_data_prompt modelnet40_64 \
  --visual_modality_type 3dpc --dataset-type 3dpc --pc_npoints 8192 \
  --n_tower 3 \
  --use_perceiver --perceiver_cross_dim_head 64 --perceiver_latent_dim 1024 --perceiver_latent_dim_head 64 --perceiver_latent_heads 16 --perceiver_num_latents 256 --perceiver_self_per_cross_attn 1 \
  --use_visual_adapter \
  --batch-size 32 --accum-freq 1 \
  --model ViT-L-14 --pretrained datacomp_xl_s13b_b90k --name pc/vitlensL_Objaverse \
  --save-frequency 1 --delete-previous-checkpoint --resume latest --save-best \
  --epochs 200
```
</details>


<details>
  <summary>Evaluate vitlensL (trained on ULIP2-Objaverse-Triplets) on ModelNet40 (click to expand)</summary>

Download [vitlensL-pc-Objaverse](https://huggingface.co/TencentARC/ViT-Lens/blob/main/Datacomp_L14_Objaverse.pt) checkpoint.

```shell
cd vitlens/
# you may change the path accordingly
torchrun --nproc_per_node=1 ./src/training/point_cloud/pc_tri_main.py \
  --cache_dir /path_to/cache_dir \
  --val-data modelnet40 --val_data_prompt modelnet40_64 \
  --visual_modality_type 3dpc --dataset-type 3dpc --v_key pc --pc_npoints 8192 \
  --n_tower 3 \
  --use_perceiver --perceiver_cross_dim_head 64 --perceiver_latent_dim 1024 --perceiver_latent_dim_head 64 --perceiver_latent_heads 16 --perceiver_num_latents 256 --perceiver_self_per_cross_attn 1 \
  --use_visual_adapter \
  --batch-size 32 \
  --model ViT-L-14 --pretrained datacomp_xl_s13b_b90k \
  --name pc/inference_vitlensL_Objaverse \
  --resume /path_to/Datacomp_L14_Objaverse.pt
```
</details>

<details>
  <summary>Train vitlensG on OpenShape-Trplets (click to expand)</summary>

```shell
cd VitLens-OpenShape/
# you may change the path accordingly
# train with 32 V100, total training time: ~150 hours
# you may change --accum-freq arg if using less GPUs
python -m torch.distributed.run $@ ./src/main.py \
 --trial_name vitlensG_OpenShapeAll --clip-model ViT-bigG-14 --pretrained laion2b_s39b_b160k \
 --visual_modality_type 3dpc --pc_tokenizer pnsa \
 --use_perceiver --use_visual_adapter \
 --pc_in_channel 6 --pc_radius 0.2 --pc_npoints 10000 --pc_num_group 512 --pc_group_size 64 --pc_trans_dim 256 \
 --perceiver_input_chan 256 --perceiver_cross_dim_head 104 --perceiver_latent_dim 1664 --perceiver_latent_dim_head 104 --perceiver_latent_heads 16 \
 --perceiver_num_latents 256 --perceiver_self_per_cross_attn 1 --perceiver_depth 4 \
 --lock-visual --unlock-trans-first-n-layers 2 \
 --skip-trans-first-n-layers 16 \
 --accum-freq 1 \
 dataset.train_batch_size=16 dataset.test_batch_size=32 \
 openshape_data_dir=/path_to/openshape_data_dir \
 model.name=clipbind \
 model.use_dense=True \
 training.use_openclip_loss=True training.use_openclip_optimizer_scheduler=True \
 training.lr=0.0001 \
 training.lr_decay_rate=0.967 \
 training.grad_clip_norm=10.0
```
</details>

<details>
  <summary>Evaluate vitlensG (trained on OpenShape-Trplets) on ModelNet40, Objaverse-LVIS and ScanObjectNN (click to expand)</summary>

Download [vitlensG-pc-OpenShapeAll](https://huggingface.co/TencentARC/ViT-Lens/blob/main/bigG14_sk16_openshape_all.pt) checkpoint.

```shell
cd VitLens-OpenShape/
# you may change the path accordingly
# evaluate with 8 V100
# you may change --accum-freq arg if using less GPUs
torchrun --nproc_per_node=8 ./src/inference.py \
 --trial_name inference_vitlensG_OpenShapeAll --clip-model ViT-bigG-14 --pretrained laion2b_s39b_b160k \
 --use_perceiver --use_visual_adapter \
 --visual_modality_type 3dpc --pc_tokenizer pnsa \
 --pc_in_channel 6 --pc_radius 0.2 --pc_npoints 10000 --pc_num_group 512 --pc_group_size 64 --pc_trans_dim 256 \
 --perceiver_input_chan 256 --perceiver_cross_dim_head 104 --perceiver_latent_dim 1664 --perceiver_latent_dim_head 104 --perceiver_latent_heads 16 \
 --perceiver_num_latents 256 --perceiver_self_per_cross_attn 1 --perceiver_depth 4 \
 --resume /path_to/bigG14_sk16_openshape_all.pt \
 --skip-trans-first-n-layers 16 \
 --lock-visual --unlock-trans-first-n-layers 2 --unlock-cls \
 dataset.train_batch_size=16 dataset.test_batch_size=32 \
 openshape_data_dir=/path_to/openshape_data_dir \
 model.name=clipbind \
 model.use_dense=True 
```
</details>


<details>
  <summary>Train vitlensG on OpenShape-Trplets-NOLVIS (click to expand)</summary>

```shell
cd VitLens-OpenShape/
# you may change the path accordingly
# train with 32 V100, total training time: ~150 hours
# you may change --accum-freq arg if using less GPUs
python -m torch.distributed.run $@ ./src/main.py \
 --trial_name vitlensG_OpenShapeNOLVIS --clip-model ViT-bigG-14 --pretrained laion2b_s39b_b160k \
 --use_perceiver --use_visual_adapter \
 --visual_modality_type 3dpc --pc_tokenizer pnsa \
 --pc_in_channel 6 --pc_radius 0.2 --pc_npoints 10000 --pc_num_group 512 --pc_group_size 64 --pc_trans_dim 256 \
 --perceiver_input_chan 256 --perceiver_cross_dim_head 104 --perceiver_latent_dim 1664 --perceiver_latent_dim_head 104 --perceiver_latent_heads 16 \
 --perceiver_num_latents 256 --perceiver_self_per_cross_attn 1 --perceiver_depth 2 \
 --lock-visual --unlock-trans-first-n-layers 2 --unlock-cls \
 --accum-freq 1 \
 openshape_data_dir=/path_to/openshape_data_dir \
 dataset.train_split=/path_to/openshape_data_dir/meta_data/split/train_no_lvis.json \
 dataset.train_batch_size=16 dataset.test_batch_size=32 \
 model.name=clipbind \
 model.use_dense=True \
 training.use_openclip_loss=True training.use_openclip_optimizer_scheduler=True \
 training.lr=0.0001 \
 training.lr_decay_rate=0.967 \
 training.grad_clip_norm=10.0
```
</details>

<details>
  <summary>Evaluate vitlensG (trained on OpenShape-Trplets-NOLVIS) on ModelNet40, Objaverse-LVIS and ScanObjectNN (click to expand)</summary>

Download [vitlensG-pc-OpenShapeNOLVIS](https://huggingface.co/TencentARC/ViT-Lens/blob/main/bigG14_openshape_nolvis.pt) checkpoint.

```shell
cd VitLens-OpenShape/
# you may change the path accordingly
# evaluate with 8 V100
# you may change --accum-freq arg if using less GPUs
torchrun --nproc_per_node=8 ./src/inference.py \
 --trial_name inference_vitlensG_OpenShapeNOLVIS --clip-model ViT-bigG-14 --pretrained laion2b_s39b_b160k \
 --use_perceiver --use_visual_adapter \
 --visual_modality_type 3dpc --pc_tokenizer pnsa \
 --pc_in_channel 6 --pc_radius 0.2 --pc_npoints 10000 --pc_num_group 512 --pc_group_size 64 --pc_trans_dim 256 \
 --perceiver_input_chan 256 --perceiver_cross_dim_head 104 --perceiver_latent_dim 1664 --perceiver_latent_dim_head 104 --perceiver_latent_heads 16 \
 --perceiver_num_latents 256 --perceiver_self_per_cross_attn 1 --perceiver_depth 2 \
 --lock-visual --unlock-trans-first-n-layers 2 --unlock-cls \
 --precision fp32 \
 --resume /path_to/bigG14_openshape_nolvis.pt \
 dataset.train_batch_size=16 dataset.test_batch_size=32 \
 openshape_data_dir=/path_to/openshape_data_dir \
 model.name=clipbind \
 model.use_dense=True 
```
</details>

## Depth
<details>
  <summary>Train vitlensL on SUN-RGBD (click to expand)</summary>

```shell
cd vitlens/
# you may change the path accordingly
# train with 8 V100, total training time: ~16 hours
# You may change --accum-freq arg if using less GPUs
torchrun --nproc_per_node=8 ./src/training/depth/depth_tri_main.py \
  --cache_dir /path_to/cache \
  --train-data sun-rgbd  --val-data sun-rgbd::nyu-depth-v2-val1::nyu-depth-v2-val2 \
  --visual_modality_type depth --dataset-type depth --v_key depth \
  --n_tower 3 \
  --use_perceiver  --perceiver_cross_dim_head 64 --perceiver_latent_dim 1024 --perceiver_latent_dim_head 64 --perceiver_latent_heads 16 \
  --perceiver_num_latents 256 --perceiver_as_identity \
  --use_visual_adapter \
  --batch-size 64 --lr 0.0002 \
  --lock-image --lock-text --lock-visual --unlock-trans-first-n-layers 4 \
  --model ViT-L-14 --pretrained datacomp_xl_s13b_b90k \
  --name depth/vitlensL_SUNRGBD \
  --save-frequency 1 --delete-previous-checkpoint --save-best --resume latest \
  --epochs 100
```
</details>


<details>
  <summary>Evaluate vitlensL on SUN-Depth-only and NYU Depth-only (click to expand)</summary>

Download [vitlensL-depth](https://huggingface.co/TencentARC/ViT-Lens/blob/main/vitlensL_depth.pt) checkpoint.

```shell
cd vitlens/
# you may change the path accordingly
torchrun --nproc_per_node=1 ./src/training/depth/depth_tri_main.py \
  --cache_dir /path_to/cache \
  --val-data sun-rgbd::nyu-depth-v2-val1::nyu-depth-v2-val2 \
  --visual_modality_type depth --dataset-type depth --v_key depth \
  --n_tower 3 \
  --use_perceiver  --perceiver_cross_dim_head 64 --perceiver_latent_dim 1024 --perceiver_latent_dim_head 64 --perceiver_latent_heads 16 \
  --perceiver_num_latents 256 --perceiver_as_identity \
  --use_visual_adapter \
  --batch-size 64 \
  --lock-image --lock-text --lock-visual --unlock-trans-first-n-layers 4 \
  --model ViT-L-14 --pretrained datacomp_xl_s13b_b90k \
  --name depth/inference_vitlensL_perf \
  --resume /path_to/vitlensL_depth.pt
```
</details>



## Audio

<details>
  <summary>Train vitlensL on Audioset (5-sec clips) (click to expand)</summary>

```shell
cd vitlens/
# you may change the path accordingly
# train with 32 V100, total batch size is 2048, total training time: ~ 150 hours
# you may change --accum-freq arg if using less GPUs
python -m torch.distributed.run $@ ./src/training/audio/audio_tri_main.py \
  --cache_dir /path_to/cache \
  --train-data audioset@audioset_train_all --val-data "audioset@val::vggsound@val::esc50@val-all::esc50@val-fold-1::clotho@val::clotho@test::audiocaps@val::audiocaps@test" \
  --visual_modality_type audio --dataset-type audio --v_key audio \
  --n_tower 3 \
  --use_perceiver --perceiver_depth 2 --perceiver_input_chan 1024 --perceiver_self_per_cross_attn 3 \
  --perceiver_cross_dim_head 64 --perceiver_latent_dim 1024 --perceiver_latent_dim_head 64 --perceiver_latent_heads 16 --perceiver_num_latents 256 \
  --use_visual_adapter --audio_load_vision \
  --n_frames 1 \
  --audio_sampling_rate 16000 --audio_clip_duration 5.0 --audio_target_length 512 --audio_mel_bins 128 --audio_fstride 10 --audio_tstride 10 --audio_freqm 12 --audio_timem 48 \
  --audio_noise_aug --audio_mix_up --audio_mix_up_p 0.5 \
  --batch-size 32 --lr 0.0002 --accum-freq 2 \
  --lock-image --lock-text --lock-visual --unlock-cls \
  --model ViT-L-14 --pretrained datacomp_xl_s13b_b90k \
  --name audio/vitlensL_AS_dur5 \
  --save-frequency 1 --delete-previous-checkpoint --resume latest --save-best \
  --epochs 80
```
</details>


<details>
  <summary>Evaluate vitlensL(trained on 5-sec clips as reported in arXiv paper) on audio benchmarks. (click to expand)</summary>

Download [vitlensL-audio](https://huggingface.co/TencentARC/ViT-Lens/blob/main/vitlensL_audio.pt) checkpoint.

```shell
cd vitlens/
# you may change the path accordingly
torchrun --nproc_per_node=1  ./src/training/audio/audio_tri_main.py \
  --cache_dir /path_to/cache \
  --val-data "audioset@val::vggsound@val::esc50@val-all::clotho@val::clotho@test::audiocaps@val::audiocaps@test::audiocaps@test_ib" \
  --visual_modality_type audio --dataset-type audio --v_key audio \
  --n_tower 3 \
  --use_perceiver --perceiver_depth 2 --perceiver_input_chan 1024 --perceiver_self_per_cross_attn 3 \
  --perceiver_cross_dim_head 64 --perceiver_latent_dim 1024 --perceiver_latent_dim_head 64 --perceiver_latent_heads 16 --perceiver_num_latents 256 \
  --use_visual_adapter \
  --n_frames 1 \
  --audio_sampling_rate 16000 --audio_clip_duration 5.0 --audio_target_length 512 --audio_mel_bins 128 --audio_fstride 10 --audio_tstride 10 --audio_freqm 12 --audio_timem 48 \
  --audio_noise_aug --audio_mix_up --audio_mix_up_p 0.5 \
  --batch-size 16 \
  --model ViT-L-14 --pretrained datacomp_xl_s13b_b90k \
  --name audio/infer_vitlensL_perf \
  --resume /path_to/vitlensL_audio.pt
```
</details>


## Tactile

<details>
  <summary>Train vitlensL Touch-and-Go (click to expand)</summary>

```shell
cd vitlens/
# you may change the path accordingly
# train with 8 V100, total training time: ~16 hours
# you may change --accum-freq arg if using less GPUs
torchrun --nproc_per_node=8 ./src/training/tactile/tactile_tri_main.py \
  --cache_dir /path_to/cache \
  --train-data tag@pretrain --val-data "tag@test_material::tag@test_hard::tag@test_rough" \
  --visual_modality_type tactile --dataset-type tactile --v_key tactile \
  --n_tower 3 \
  --batch-size 64 --lr 0.0002 \
  --perceiver_num_latents 256 \
  --lock-image --lock-text --lock-visual --lock-visual-unlocked-groups 5 --unlock_from_head \
  --model ViT-L-14 --pretrained datacomp_xl_s13b_b90k \
  --name tactile/vitlensL_tag \
  --save-frequency 1 --delete-previous-checkpoint --resume latest --save-best \
  --epochs 80
```
</details>

<details>
  <summary>Evaluate vitlensL on Touch-and-Go Material, Hard/Soft, Rough/Smooth (click to expand)</summary>

Download [vitlensL-tactile](https://huggingface.co/TencentARC/ViT-Lens/blob/main/vitlensL_tactile.pt) checkpoint.

```shell
cd vitlens/
# you may change the path accordingly
torchrun --nproc_per_node=1 ./src/training/tactile/tactile_tri_main.py \
  --cache_dir /path_to/cache \
  --val-data "tag@test_material::tag@test_hard::tag@test_rough" \
  --visual_modality_type tactile --dataset-type tactile --v_key tactile \
  --n_tower 3 \
  --batch-size 64 \
  --perceiver_num_latents 256 \
  --lock-image --lock-text --lock-visual --lock-visual-unlocked-groups 5 --unlock_from_head \
  --model ViT-L-14 --pretrained datacomp_xl_s13b_b90k \
  --name tactile/inference_vitlensL_tag \
  --resume /path_to/vitlens_tactile.pt
```
</details>


## EEG
<details>
  <summary>Train vitlensL on ImageNet EEG (click to expand)</summary>

```shell
cd vitlens/
# you may change the path accordingly
# train with 8 V100, total training time: ~8 hours
# you may change --accum-freq arg if using less GPUs
torchrun --nproc_per_node=8 ./src/training/eeg/eeg_tri_main.py \
  --cache_dir /path_to/cache \
  --train-data eeg@train --val-data "eeg@val::eeg@test" \
  --visual_modality_type eeg --dataset-type eeg --v_key eeg \
  --eeg_window_size 1 --eeg_stride 1 \
  --use_perceiver --perceiver_depth 1 --perceiver_input_chan 1024 --perceiver_self_per_cross_attn 1 \
  --perceiver_cross_dim_head 64 --perceiver_latent_dim 1024 --perceiver_latent_dim_head 64 --perceiver_latent_heads 16 --perceiver_num_latents 256 \
  --use_visual_adapter \
  --n_tower 3 \
  --batch-size 64 --lr 0.0002 \
  --lock-image --lock-text --lock-visual --unlock-cls  \
  --model ViT-L-14 --pretrained datacomp_xl_s13b_b90k \
  --name eeg/vitlens_INEEG \
  --save-frequency 1 --delete-previous-checkpoint --resume latest --save-best \
  --epochs 40
```
</details>

<details>
  <summary>Evaluate vitlensL on ImageNet EEG (click to expand)</summary>

Download [vitlensL-eeg](https://huggingface.co/TencentARC/ViT-Lens/blob/main/vitlensL_eeg.pt) checkpoint.

```shell
cd vitlens/
# you may change the path accordingly
torchrun --nproc_per_node=1 ./src/training/eeg/eeg_tri_main.py \
  --cache_dir /path_to/cache \
  --val-data "eeg@val::eeg@test" \
  --visual_modality_type eeg --dataset-type eeg --v_key eeg \
  --eeg_window_size 1 --eeg_stride 1 \
  --use_perceiver --perceiver_depth 1 --perceiver_input_chan 1024 --perceiver_self_per_cross_attn 1 \
  --perceiver_cross_dim_head 64 --perceiver_latent_dim 1024 --perceiver_latent_dim_head 64 --perceiver_latent_heads 16 --perceiver_num_latents 256 \
  --use_visual_adapter \
  --n_tower 3 \
  --batch-size 64 \
  --lock-image --lock-text --lock-visual --unlock-cls \
  --model ViT-L-14 --pretrained datacomp_xl_s13b_b90k \
  --name eeg/vitlens_INEEG \
  --resume /path_to/vitlensL_eeg.pt 
```
</details>
