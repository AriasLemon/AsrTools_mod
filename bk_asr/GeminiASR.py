import base64
import json
import logging
import os
import re
import socket
from typing import Dict, List, Optional, Union

import requests

from .ASRData import ASRData, ASRDataSeg
from .BaseASR import BaseASR


# 默认硬编码的 Google AI Studio API Key（公开发布版本保持为空，由用户自行配置）
DEFAULT_GEMINI_API_KEY = ""

# Gemini 语音转录专用模型
GEMINI_TRANSCRIBE_MODEL = "gemini-3.5-transcribe"

# Interactions API 端点
INTERACTIONS_API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


def detect_local_proxy() -> Optional[Dict[str, str]]:
    """检测本地可用代理（Clash / Mihomo 常用端口 7897、7890 等）"""
    # 优先使用系统或环境变量中已有的代理配置
    env_http = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
    env_https = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if env_http or env_https:
        return {
            "http": env_http or env_https,
            "https": env_https or env_http,
        }

    # 常见本地代理端口探活
    probe_ports = [7897, 7890, 10808, 10809]
    for port in probe_ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.15)
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    proxy_url = f"http://127.0.0.1:{port}"
                    return {"http": proxy_url, "https": proxy_url}
        except Exception:
            continue
    return None


class GeminiASR(BaseASR):
    """Google Gemini 3.5 Transcribe 语音识别接口 (G 接口)"""

    def __init__(
        self,
        audio_path: Union[str, bytes],
        use_cache: bool = False,
        api_key: Optional[str] = None,
        model_name: str = GEMINI_TRANSCRIBE_MODEL,
        max_chars_per_line: int = 25,
    ):
        super().__init__(audio_path, use_cache=use_cache)
        self.api_key = self._resolve_api_key(api_key)
        self.model_name = model_name
        self.max_chars_per_line = max_chars_per_line
        self.proxies = detect_local_proxy()

    def _resolve_api_key(self, api_key: Optional[str] = None) -> str:
        """
        按以下优先级获取 Gemini API Key：
        1. 构造函数显式传入的 api_key
        2. 环境变量 GEMINI_API_KEY 或 GOOGLE_API_KEY
        3. 项目根目录或当前工作目录下的 gemini_key.txt 文件
        4. 本模块硬编码的 DEFAULT_GEMINI_API_KEY
        """
        if api_key and api_key.strip():
            return api_key.strip()

        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()

        candidate_dirs = [
            os.getcwd(),
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ]
        for c_dir in candidate_dirs:
            key_file = os.path.join(c_dir, "gemini_key.txt")
            if os.path.exists(key_file):
                try:
                    with open(key_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            return content
                except Exception:
                    pass

        return DEFAULT_GEMINI_API_KEY or ""

    def _get_mime_type(self) -> str:
        """获取音频 MIME 类型"""
        if isinstance(self.audio_path, str):
            ext = self.audio_path.split(".")[-1].lower()
            mime_map = {
                "mp3": "audio/mp3",
                "wav": "audio/wav",
                "flac": "audio/flac",
                "m4a": "audio/m4a",
                "aac": "audio/aac",
                "ogg": "audio/ogg",
            }
            return mime_map.get(ext, "audio/mp3")
        return "audio/mp3"

    def _run(self) -> dict:
        """调用 Google AI Studio Interactions API 进行语音识别"""
        if not self.api_key:
            raise ValueError(
                "未检测到有效的 Gemini API Key！\n"
                "请通过以下任一方式配置：\n"
                "1. 在项目根目录下创建 gemini_key.txt 文件并填入您的 API Key（推荐）\n"
                "2. 在 bk_asr/GeminiASR.py 中设置 DEFAULT_GEMINI_API_KEY = '您的Key'\n"
                "3. 设置系统或终端环境变量 GEMINI_API_KEY\n"
                "4. 在代码中传入 api_key 参数：GeminiASR(audio_path, api_key='...')\n"
                "可在 Google AI Studio 免费申请 API Key: https://aistudio.google.com/app/apikey"
            )

        if not self.file_binary:
            raise ValueError("音频数据为空，无法上传")

        base64_audio = base64.b64encode(self.file_binary).decode("utf-8")
        mime_type = self._get_mime_type()

        payload = {
            "model": self.model_name,
            "input": [
                {
                    "type": "audio",
                    "data": base64_audio,
                    "mime_type": mime_type,
                }
            ],
            "generation_config": {
                "transcription_config": {
                    "mode": {
                        "type": "verbatim",
                        "timestamp_granularities": ["word"],
                    }
                }
            },
        }

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        logging.info(f"[GeminiASR] 开始向 Google AI Studio 请求转写: 模型={self.model_name}, 代理={self.proxies}")

        try:
            resp = requests.post(
                INTERACTIONS_API_URL,
                headers=headers,
                json=payload,
                proxies=self.proxies,
                timeout=180,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"请求 Google AI Studio API 发生网络错误: {e}")

        if resp.status_code != 200:
            err_text = resp.text
            try:
                err_json = resp.json()
                err_msg = err_json.get("error", {}).get("message", err_text)
            except Exception:
                err_msg = err_text
            raise RuntimeError(f"Google AI Studio API 调用失败 (HTTP {resp.status_code}): {err_msg}")

        resp_data = resp.json()
        logging.info("[GeminiASR] 成功获取识别结果")
        return resp_data

    @staticmethod
    def _parse_offset_to_ms(offset_val: Union[str, int, float]) -> int:
        """将 API 返回的时间戳偏移转换为毫秒整数"""
        if offset_val is None:
            return 0
        if isinstance(offset_val, (int, float)):
            # 如果值很小（如 <= 3600），可能是秒单位；否则是毫秒
            if offset_val < 3600:
                return int(round(offset_val * 1000))
            return int(round(offset_val))
        val_str = str(offset_val).strip()
        if val_str.endswith("ms"):
            return int(round(float(val_str[:-2])))
        if val_str.endswith("s"):
            return int(round(float(val_str[:-1]) * 1000))
        try:
            val_float = float(val_str)
            return int(round(val_float * 1000)) if val_float < 3600 else int(round(val_float))
        except ValueError:
            return 0

    def _extract_words(self, resp_data: dict) -> List[dict]:
        """从 API 响应中提取词级时间戳列表"""
        words = []

        # 结构 1: Interactions API steps[].content[].annotations[] (word_info)
        steps = resp_data.get("steps", [])
        for step in steps:
            contents = step.get("content", [])
            for c in contents:
                # 检查 annotations 里的 word_info
                annotations = c.get("annotations", [])
                for anno in annotations:
                    if anno.get("type") == "word_info":
                        text = anno.get("text", "")
                        start_ms = self._parse_offset_to_ms(anno.get("start_offset"))
                        end_ms = self._parse_offset_to_ms(anno.get("end_offset"))
                        if text:
                            words.append({
                                "text": text,
                                "start_ms": start_ms,
                                "end_ms": max(end_ms, start_ms + 10),
                            })

        # 结构 2: 如果 steps 中未提取到，检查 GenerateContent 风格的 audio_transcription 或 candidates
        if not words:
            candidates = resp_data.get("candidates", [])
            for cand in candidates:
                parts = cand.get("content", {}).get("parts", [])
                for part in parts:
                    audio_trans = part.get("audio_transcription", {})
                    word_list = audio_trans.get("words", [])
                    for w in word_list:
                        text = w.get("text") or w.get("word", "")
                        start_ms = self._parse_offset_to_ms(w.get("start_offset"))
                        end_ms = self._parse_offset_to_ms(w.get("end_offset"))
                        if text:
                            words.append({
                                "text": text,
                                "start_ms": start_ms,
                                "end_ms": max(end_ms, start_ms + 10),
                            })

        # 结构 3: 顶层 words 字段
        if not words and "words" in resp_data:
            for w in resp_data["words"]:
                text = w.get("text") or w.get("word", "")
                start_ms = self._parse_offset_to_ms(w.get("start_offset", w.get("start_time", 0)))
                end_ms = self._parse_offset_to_ms(w.get("end_offset", w.get("end_time", 0)))
                if text:
                    words.append({
                        "text": text,
                        "start_ms": start_ms,
                        "end_ms": max(end_ms, start_ms + 10),
                    })

        # 兜底：如果 API 仅返回全文文本而无词级信息，按标点切句
        if not words:
            full_text = ""
            if "output_text" in resp_data:
                full_text = resp_data["output_text"]
            elif steps:
                for step in steps:
                    for c in step.get("content", []):
                        if c.get("type") == "text":
                            full_text += c.get("text", "") + " "
            if full_text.strip():
                logging.warning("[GeminiASR] 未检测到显式词级标注，使用全文回退解析")
                words = [{"text": full_text.strip(), "start_ms": 0, "end_ms": 1000}]

        return words

    @staticmethod
    def _format_word_sequence(word_list: List[dict]) -> str:
        """合理拼接词列表，西文单词间补充空格，中文/全角标点紧凑连接"""
        if not word_list:
            return ""

        result = []
        for i, w in enumerate(word_list):
            cur_text = w["text"].strip()
            if not cur_text:
                continue
            if not result:
                result.append(cur_text)
                continue

            prev_text = result[-1]
            prev_char = prev_text[-1]
            cur_char = cur_text[0]

            # 若前后均为西文字母/数字，添加空格隔开
            if prev_char.isascii() and prev_char.isalnum() and cur_char.isascii() and cur_char.isalnum():
                result.append(" " + cur_text)
            else:
                result.append(cur_text)

        return "".join(result).strip()

    def _aggregate_words_to_segments(self, words: List[dict]) -> List[ASRDataSeg]:
        """
        聚合词级时间戳为字幕块（SRT段落）：
        1. 句子语义通顺：依据标点符号（。！？，、；）与语音停顿断句
        2. 硬性限制：每行文本块严格不超过 max_chars_per_line（默认 25 字）
        """
        if not words:
            return []

        # 拆分处理过长的单个词（如果单词本身 > max_chars_per_line）
        normalized_words = []
        for w in words:
            text = w["text"]
            start_ms = w["start_ms"]
            end_ms = w["end_ms"]
            if len(text) > self.max_chars_per_line:
                step = self.max_chars_per_line
                total_len = len(text)
                duration = max(end_ms - start_ms, 10)
                for idx in range(0, total_len, step):
                    chunk_text = text[idx : idx + step]
                    chunk_start = start_ms + int(duration * (idx / total_len))
                    chunk_end = start_ms + int(duration * (min(idx + step, total_len) / total_len))
                    normalized_words.append({
                        "text": chunk_text,
                        "start_ms": chunk_start,
                        "end_ms": chunk_end,
                    })
            else:
                normalized_words.append(w)

        segments = []
        current_chunk: List[dict] = []

        sentence_enders = ("。", "！", "？", "!", "?", "…")
        clause_breaks = ("，", "、", "；", ";", "：", ":", ",")

        for w in normalized_words:
            if not current_chunk:
                current_chunk.append(w)
                continue

            last_word = current_chunk[-1]
            pause_ms = w["start_ms"] - last_word["end_ms"]

            candidate_chunk = current_chunk + [w]
            candidate_text = self._format_word_sequence(candidate_chunk)
            candidate_len = len(candidate_text)

            current_text = self._format_word_sequence(current_chunk)
            current_len = len(current_text)

            should_break = False

            # 规则 1：硬性限制，累计长度不能超过 max_chars_per_line
            if candidate_len > self.max_chars_per_line:
                should_break = True

            # 规则 2：前一词以句末终结符（。！？!?）结尾，且当前块有一定字数（>= 5 字）
            elif any(last_word["text"].endswith(p) for p in sentence_enders) and current_len >= 5:
                should_break = True

            # 规则 3：语音停顿长（>= 700ms 且字数 >= 8，或 >= 1200ms）
            elif (pause_ms >= 700 and current_len >= 8) or pause_ms >= 1200:
                should_break = True

            # 规则 4：前一词以分句逗号等停顿标点结尾，且当前块字数已经较长（>= 14 字），继续添加易超长
            elif any(last_word["text"].endswith(p) for p in clause_breaks) and current_len >= 14:
                should_break = True

            if should_break:
                seg_text = self._format_word_sequence(current_chunk)
                seg_start = current_chunk[0]["start_ms"]
                seg_end = current_chunk[-1]["end_ms"]
                if seg_text:
                    segments.append(ASRDataSeg(seg_text, seg_start, seg_end))
                current_chunk = [w]
            else:
                current_chunk.append(w)

        if current_chunk:
            seg_text = self._format_word_sequence(current_chunk)
            seg_start = current_chunk[0]["start_ms"]
            seg_end = current_chunk[-1]["end_ms"]
            if seg_text:
                segments.append(ASRDataSeg(seg_text, seg_start, seg_end))

        return segments

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        """将 API 结果转换为 ASRDataSeg 列表"""
        words = self._extract_words(resp_data)
        return self._aggregate_words_to_segments(words)
