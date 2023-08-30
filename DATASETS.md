## 📦 Datasets

### 3D Pretraining Datasets
- ULIP-ShapeNet Triplets: follow [ULIP](https://github.com/salesforce/ULIP) to download and prepare the data.
- ULIP2-Objaverse Triplets: follow [ULIP](https://github.com/salesforce/ULIP) to download and prepare the data.
- OpenShape Triplets: follow [OpenShape](https://github.com/Colin97/OpenShape_code) to download and prepare the data.

### 3D Zero-shot Datasets
- ModelNet40: we prepare two versions following [ULIP](https://github.com/salesforce/ULIP) and [OpenShape](https://github.com/Colin97/OpenShape_code).
- ScanObjectNN: follow [OpenShape](https://github.com/Colin97/OpenShape_code) to download and prepare the data.
- Objaverse LVIS: follow [OpenShape](https://github.com/Colin97/OpenShape_code) to download and prepare the data.

### ⏰ Change Hard-coded Paths
Do note that you may change the hard-coded paths in the following files. Will make it configurable in the future.
- [ModelNet40.yaml](open_clip/src/open_clip/modal_3d/data/ModelNet40.yaml)
- [Objverse.yaml](open_clip/src/open_clip/modal_3d/data/Objverse.yaml)
- [ShapeNet-55.yaml](open_clip/src/open_clip/modal_3d/data/ShapeNet-55.yaml)
- [train.yaml for OpenShape](/group/30042/weixian/code/release/ViT-Lens/OpenShape/src/configs/train.yaml)
