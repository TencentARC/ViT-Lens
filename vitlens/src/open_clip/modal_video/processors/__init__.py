from .vt_processors import (
    AIOBeitv1VisProcessor, AIOBeitv1VisProcessorEval,
    AIOBeitv2VisProcessor, AIOBeitv2VisProcessorEval,
    AIOEVAClipVisProcessor, AIOEVAClipVisProcessorEval,
    AIOLavisVisProcessor, AIOLavisVisProcessorEval,
    AIOV1VisProcessor, AIOV1VisProcessorEval,
    BlipCaptionProcessor, BlipQuestionProcessor,
    OpenClip_VisProcessor, OpenClip_VisProcessorEval
)

PROCESSOR_CLS = {
    "aio_beitv1_train": AIOBeitv1VisProcessor,
    "aio_beitv1_eval": AIOBeitv1VisProcessorEval,
    "aio_beitv2_train": AIOBeitv2VisProcessor,
    "aio_beitv2_eval": AIOBeitv2VisProcessorEval,
    "aio_eva_train": AIOEVAClipVisProcessor,
    "aio_eva_eval": AIOEVAClipVisProcessorEval,
    "aio_lavis_train": AIOLavisVisProcessor,
    "aio_lavis_eval": AIOLavisVisProcessorEval,
    "aio_v1_train": AIOV1VisProcessor,
    "aio_v1_eval": AIOV1VisProcessorEval,
    "blip_caption": BlipCaptionProcessor,
    "blip_question": BlipQuestionProcessor,
    "openclip_train": OpenClip_VisProcessor,
    "openclip_eval": OpenClip_VisProcessorEval
}