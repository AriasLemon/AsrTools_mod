import base64
import json
import logging
import os
import re
import socket
import unicodedata
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
        max_chars_per_line: int = 18,
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

    @staticmethod
    def _clean_subtitle_punctuation(text: str) -> str:
        """
        文本块换行处理后，字幕中除了顿号、括号、数学符号和·，其他标点符号换为空格。
        保留项：
        - 顿号：、
        - 括号：()（）[]【】{}｛｝〔〕〈〉《》«»‹›〖〗
        - 数学符号：+-*/=<>%^~±×÷≠≤≥≈≡‰°√∑∏∫∞＋－×÷＝＜＞％ 以及数字之间的小数点（如 3.14）
        - ·（间隔号）：·・•●
        其他所有标点符号均替换为空格，并合并连续空格及去除首尾空格。
        """
        if not text:
            return ""

        # 1. 保护数字间的点（如 3.14）
        placeholder = "\uE000"
        text_protected = re.sub(r'(?<=\d)\.(?=\d)', placeholder, text)

        # 2. 保留标点白名单：顿号、括号、数学符号、·
        pause_comma = {"、"}
        brackets = set("()（）[]【】{}｛｝〔〕〈〉《》«»‹›〖〗")
        math_symbols = set("+-*/=<>%^~±×÷≠≤≥≈≡‰°√∑∏∫∞＋－×÷＝＜＞％")
        middle_dots = set("·・•●")
        allowed = pause_comma | brackets | math_symbols | middle_dots

        cleaned_chars = []
        for ch in text_protected:
            if ch == placeholder:
                cleaned_chars.append(".")
            elif ch.isalnum() or ch.isspace() or ch in allowed:
                cleaned_chars.append(ch)
            elif unicodedata.category(ch).startswith("P") or unicodedata.category(ch).startswith("S"):
                cleaned_chars.append(" ")
            else:
                cleaned_chars.append(ch)

        return re.sub(r" +", " ", "".join(cleaned_chars)).strip()

    @staticmethod
    def _is_period_ended(text: str) -> bool:
        """判断文本是否以句号结尾（。、｡ 或非小数点的 .）"""
        if not text:
            return False
        if text.endswith("。") or text.endswith("｡"):
            return True
        if text.endswith(".") and not re.search(r'\d\.\d?$', text):
            return True
        return False

    def _split_words_by_period_and_length(self, words: List[dict]) -> List[dict]:
        """预处理词列表：若单词内部包含句号，在句号处切分；若单词长度超过 max_chars_per_line，按长度切分"""
        result = []
        for w in words:
            text = w["text"]
            start_ms = w["start_ms"]
            end_ms = w["end_ms"]
            duration = max(end_ms - start_ms, 10)

            # 1. 检查内部句号切分（例如 "逻辑。首先" -> "逻辑。" 与 "首先"）
            parts = []
            cur_idx = 0
            for m in re.finditer(r'(?:[。｡]|(?<!\d)\.(?!\d))', text):
                end_pos = m.end()
                if end_pos < len(text):
                    part_str = text[cur_idx:end_pos]
                    if part_str:
                        parts.append(part_str)
                    cur_idx = end_pos
            if cur_idx < len(text):
                parts.append(text[cur_idx:])

            if not parts:
                parts = [text]

            # 计算切分后的时间戳
            cur_start = start_ms
            total_len = len(text)
            for part in parts:
                part_duration = int(duration * (len(part) / total_len)) if total_len > 0 else 0
                part_end = cur_start + part_duration

                # 2. 如果单个分块长度仍然 > max_chars_per_line，进行长度切分
                if len(part) > self.max_chars_per_line:
                    step = self.max_chars_per_line
                    p_len = len(part)
                    p_dur = max(part_end - cur_start, 10)
                    for i in range(0, p_len, step):
                        sub_text = part[i : i + step]
                        sub_start = cur_start + int(p_dur * (i / p_len))
                        sub_end = cur_start + int(p_dur * (min(i + step, p_len) / p_len))
                        result.append({
                            "text": sub_text,
                            "start_ms": sub_start,
                            "end_ms": sub_end,
                        })
                else:
                    result.append({
                        "text": part,
                        "start_ms": cur_start,
                        "end_ms": part_end,
                    })
                cur_start = part_end
        return result

    def _aggregate_words_to_segments(self, words: List[dict]) -> List[ASRDataSeg]:
        """
        聚合词级时间戳为字幕块（SRT段落）：
        1. 换行规则：每行最多 18 个字符，超过就换行；遇到句号也换行。
        2. 标点清洗规则：文本块换行处理后，字幕中除了顿号、括号、数学符号和·，其他标点符号换为空格。
        """
        if not words:
            return []

        normalized_words = self._split_words_by_period_and_length(words)

        segments = []
        current_chunk: List[dict] = []

        for w in normalized_words:
            if not current_chunk:
                current_chunk.append(w)
                continue

            last_word = current_chunk[-1]

            candidate_chunk = current_chunk + [w]
            candidate_text = self._format_word_sequence(candidate_chunk)
            candidate_len = len(candidate_text)

            should_break = False

            # 规则 1：遇到句号换行（前一个词以句号结尾，换到下一行）
            if self._is_period_ended(last_word["text"]):
                should_break = True

            # 规则 2：每行最多 18 个字符，超过就换行
            elif candidate_len > self.max_chars_per_line:
                should_break = True

            if should_break:
                seg_raw_text = self._format_word_sequence(current_chunk)
                seg_text = self._clean_subtitle_punctuation(seg_raw_text)
                seg_start = current_chunk[0]["start_ms"]
                seg_end = current_chunk[-1]["end_ms"]
                if seg_text:
                    segments.append(ASRDataSeg(seg_text, seg_start, seg_end))
                current_chunk = [w]
            else:
                current_chunk.append(w)

        if current_chunk:
            seg_raw_text = self._format_word_sequence(current_chunk)
            seg_text = self._clean_subtitle_punctuation(seg_raw_text)
            seg_start = current_chunk[0]["start_ms"]
            seg_end = current_chunk[-1]["end_ms"]
            if seg_text:
                segments.append(ASRDataSeg(seg_text, seg_start, seg_end))

        return segments

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        """将 API 结果转换为 ASRDataSeg 列表"""
        words = self._extract_words(resp_data)
        return self._aggregate_words_to_segments(words)
