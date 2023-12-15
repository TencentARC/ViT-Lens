import os
import torch

from open_clip import ModalityType
from mm_vit_lens import ViTLens

here = os.path.abspath(os.path.dirname(__file__))

model = ViTLens(modality_loaded=[ModalityType.IMAGE, ModalityType.AUDIO, ModalityType.TEXT, ModalityType.PC])
model = model.to("cuda:0")

# Example 1
images = [
    os.path.join(here, "assets/example/image_bird.jpg"),
    os.path.join(here, "assets/example/image_fire.jpg"),
    os.path.join(here, "assets/example/image_dog.jpg"),
    os.path.join(here, "assets/example/image_beach.jpg"),
]
audios = [
    os.path.join(here, "assets/example/audio_chirping_birds.flac"),
    os.path.join(here, "assets/example/audio_crackling_fire.flac"),
    os.path.join(here, "assets/example/audio_dog.flac"),
    os.path.join(here, "assets/example/audio_sea_wave.flac"),
]
texts = [
    "a bird",
    "crackling fire",
    "a dog",
    "sea wave",
]
inputs_1 = {
    ModalityType.IMAGE: images,
    ModalityType.AUDIO: audios,
    ModalityType.TEXT: texts,
}

with torch.no_grad(), torch.cuda.amp.autocast():
    outputs_1 = model.encode(inputs_1, normalize=True)

sim_at = torch.softmax(100 * outputs_1[ModalityType.AUDIO] @ outputs_1[ModalityType.TEXT].T, dim=-1)
print(
    "Audio x Text:\n",
    sim_at
)
# Audio x Text:
#  tensor([[9.9998e-01, 9.3977e-07, 2.1545e-05, 9.3642e-08],
#         [3.8017e-09, 1.0000e+00, 3.1551e-09, 6.9498e-10],
#         [9.4895e-03, 1.3270e-06, 9.9051e-01, 2.5545e-07],
#         [9.7020e-06, 6.4767e-07, 2.8860e-06, 9.9999e-01]], device='cuda:0')

sim_ai = torch.softmax(100 * outputs_1[ModalityType.AUDIO] @ outputs_1[ModalityType.IMAGE].T, dim=-1)
print(
    "Audio x Image:\n",
    sim_ai
)
# Audio x Image:
#  tensor([[1.0000e+00, 1.5798e-06, 2.0614e-06, 1.6502e-07],
#         [2.3712e-09, 1.0000e+00, 1.4446e-10, 1.2260e-10],
#         [4.9333e-03, 1.2942e-02, 9.8212e-01, 1.8582e-06],
#         [6.8347e-04, 1.0547e-02, 1.3476e-05, 9.8876e-01]], device='cuda:0')


# Example 2
pcs = [
    os.path.join(here, "assets/example/pc_car_0260.npy"),
    os.path.join(here, "assets/example/pc_guitar_0243.npy"),
    os.path.join(here, "assets/example/pc_monitor_0503.npy"),
    os.path.join(here, "assets/example/pc_person_0102.npy"),
    os.path.join(here, "assets/example/pc_piano_0286.npy"),
]
text_pcs = ["a car", "a guitar", "a monitor", "a person", "a piano"]
inputs_2 = {
    ModalityType.PC: pcs,
    ModalityType.TEXT: text_pcs,
}
with torch.no_grad(), torch.cuda.amp.autocast():
    outputs_2 = model.encode(inputs_2, normalize=True)
sim_pc_t = torch.softmax(100 * outputs_2[ModalityType.PC] @ outputs_2[ModalityType.TEXT].T, dim=-1)
print(
    "PointCould x Text:\n",
    sim_pc_t
)
# PointCould x Text:
#  tensor([[9.9945e-01, 1.0483e-05, 1.4904e-04, 2.3988e-05, 3.7041e-04],
#         [1.2574e-09, 1.0000e+00, 6.8450e-09, 2.6463e-08, 3.3659e-07],
#         [6.2730e-09, 1.9918e-06, 9.9999e-01, 6.7161e-06, 4.9279e-06],
#         [1.8846e-06, 7.4831e-06, 4.4594e-06, 9.9998e-01, 7.9092e-06],
#         [1.2218e-08, 1.5571e-06, 1.8991e-07, 1.7521e-08, 1.0000e+00]],
#        device='cuda:0')