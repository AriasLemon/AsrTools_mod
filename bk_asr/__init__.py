from .WhisperASR import WhisperASR
from .BcutASR import BcutASR
from .JianYingASR import JianYingASR
from .KuaiShouASR import KuaiShouASR

__all__ = ["WhisperASR", "BcutASR", "JianYingASR", "KuaiShouASR"]


def transcribe(audio_file, platform="WhisperASR", **kwargs):
    assert platform in __all__, f"不支持的平台: {platform}"
    asr = globals()[platform](audio_file, **kwargs)
    return asr.run()
