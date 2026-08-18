import logging
import os
import shutil
import sys
import tempfile
from typing import List, Optional, Union

import imageio_ffmpeg
from pywhispercpp.model import Model

from .ASRData import ASRDataSeg
from .BaseASR import BaseASR


def _ensure_ffmpeg_in_path():
    """确保 ffmpeg 在系统环境变量 PATH 中，供 pywhispercpp 或 subprocess 使用"""
    if shutil.which("ffmpeg") is not None:
        return
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
        if not os.path.exists(target_ffmpeg):
            shutil.copyfile(ffmpeg_exe, target_ffmpeg)
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:
        logging.warning(f"无法自动配置 ffmpeg: {e}")


def get_whisper_models_dir() -> str:
    """获取 resources/whisper 文件夹的绝对路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "..", "resources", "whisper"),
        os.path.join(os.getcwd(), "resources", "whisper"),
        os.path.join(os.path.dirname(sys.executable), "resources", "whisper"),
    ]
    for p in candidates:
        abs_p = os.path.abspath(p)
        if os.path.isdir(abs_p):
            return abs_p
    # 默认创建并返回 resources/whisper 目录
    default_p = os.path.abspath(os.path.join(current_dir, "..", "resources", "whisper"))
    os.makedirs(default_p, exist_ok=True)
    return default_p


def list_whisper_models() -> List[str]:
    """列出 resources/whisper 目录下的所有可用 Whisper 模型文件"""
    models_dir = get_whisper_models_dir()
    if not os.path.isdir(models_dir):
        return []
    valid_exts = (".bin", ".gguf")
    models = [
        f
        for f in os.listdir(models_dir)
        if f.lower().endswith(valid_exts)
        and os.path.isfile(os.path.join(models_dir, f))
    ]
    return sorted(models)


def _find_default_model_path(model_path: Optional[str] = None) -> str:
    """查找本地 resources/whisper 路径下的 Whisper 模型文件"""
    models_dir = get_whisper_models_dir()

    if model_path:
        # 如果传入的是已存在的完整文件路径
        if os.path.isfile(model_path):
            return os.path.abspath(model_path)
        # 如果传入的是文件名，去 resources/whisper 目录下查找
        candidate = os.path.join(models_dir, model_path)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    # 优先在 resources/whisper 目录下查找
    models = list_whisper_models()
    if models:
        # 若存在 ggml-base.bin 则优先使用，否则使用列表中的第一个模型
        if "ggml-base.bin" in models:
            return os.path.join(models_dir, "ggml-base.bin")
        return os.path.join(models_dir, models[0])

    # 备选向后兼容查找
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "..", "..", "ggml-base.bin"),
        os.path.join(current_dir, "..", "ggml-base.bin"),
        os.path.join(os.getcwd(), "ggml-base.bin"),
    ]
    for p in candidates:
        abs_p = os.path.abspath(p)
        if os.path.isfile(abs_p):
            return abs_p

    raise FileNotFoundError(
        f"未在 '{models_dir}' 路径下找到 Whisper 模型文件 (.bin / .gguf)。\n"
        f"请将下载好的模型文件（例如 ggml-base.bin）放入 resources/whisper 文件夹中，方便随时替换和使用。"
    )


class WhisperASR(BaseASR):
    """基于 Whisper.cpp (pywhispercpp) 的本地离线语音识别"""

    # 全局模型缓存，避免每次识别重复加载相同模型
    _model_instance = None
    _loaded_model_path = None

    def __init__(
        self,
        audio_path: Union[str, bytes],
        model_path: Optional[str] = None,
        language: str = "zh",
        initial_prompt: str = "以下是普通话的句子，请使用简体中文。",
        n_threads: int = 4,
        use_cache: bool = False,
    ):
        super().__init__(audio_path, use_cache=use_cache)
        _ensure_ffmpeg_in_path()

        self.model_path = _find_default_model_path(model_path)
        self.language = language
        self.initial_prompt = initial_prompt
        self.n_threads = n_threads

        self._init_model()

    def _init_model(self):
        """初始化或复用已加载的 Whisper 模型"""
        if (
            WhisperASR._model_instance is None
            or WhisperASR._loaded_model_path != self.model_path
        ):
            logging.info(f"正在从 resources/whisper 加载本地模型: {self.model_path}")
            WhisperASR._model_instance = Model(
                self.model_path,
                n_threads=self.n_threads,
            )
            WhisperASR._loaded_model_path = self.model_path
        self.model = WhisperASR._model_instance

    def _get_key(self) -> str:
        model_name = os.path.basename(self.model_path)
        return f"{self.__class__.__name__}-{model_name}-{self.language}-{self.crc32_hex}"

    def _run(self) -> dict:
        """执行本地识别"""
        temp_audio_file = None
        if isinstance(self.audio_path, bytes) or not isinstance(self.audio_path, str):
            # 将二进制写入临时文件
            temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_audio.write(self.file_binary)
            temp_audio.close()
            temp_audio_file = temp_audio.name
            target_path = temp_audio_file
        else:
            target_path = self.audio_path

        try:
            logging.info(f"开始使用 Whisper.cpp 本地转写: {target_path} (模型: {os.path.basename(self.model_path)})")
            segments = self.model.transcribe(
                target_path,
                language=self.language,
                initial_prompt=self.initial_prompt,
            )

            results = []
            for seg in segments:
                # pywhispercpp 的 t0, t1 单位为 10ms (厘秒)，转换为毫秒需乘以 10
                start_ms = seg.t0 * 10
                end_ms = seg.t1 * 10
                text = seg.text.strip()
                if text:
                    results.append({
                        "text": text,
                        "start": start_ms,
                        "end": end_ms,
                    })

            logging.info(f"Whisper.cpp 本地转写完成，生成 {len(results)} 个语音片段")
            return {"segments": results}
        finally:
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    os.remove(temp_audio_file)
                except Exception:
                    pass

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        return [
            ASRDataSeg(u["text"], u["start"], u["end"])
            for u in resp_data.get("segments", [])
        ]
