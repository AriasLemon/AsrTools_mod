import os
import sys

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from bk_asr.GeminiASR import GeminiASR
from bk_asr.ASRData import ASRDataSeg, ASRData


def test_gemini_asr_init():
    print("[-] 测试 GeminiASR 初始化与配置...")
    audio_path = os.path.join(os.path.dirname(__file__), "resources", "test.mp3")
    asr = GeminiASR(audio_path)
    assert asr.model_name == "gemini-3.5-transcribe", "模型名必须为 gemini-3.5-transcribe"
    assert asr.max_chars_per_line == 25, "每行最大字数默认必须为 25"

    # 测试显式传入 key
    custom_asr = GeminiASR(audio_path, api_key="test_custom_api_key")
    assert custom_asr.api_key == "test_custom_api_key", "显式传入的 API Key 必须正确生效"
    print("  [PASS] 初始化配置验证通过")


def test_word_extraction():
    print("[-] 测试 API 响应词级时间戳提取解析...")
    audio_path = os.path.join(os.path.dirname(__file__), "resources", "test.mp3")
    asr = GeminiASR(audio_path)

    mock_resp = {
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": "今天我们讨论数字化转型，首先了解它的基本逻辑。",
                        "annotations": [
                            {"type": "word_info", "text": "今天", "start_offset": "0.100s", "end_offset": "0.400s"},
                            {"type": "word_info", "text": "我们", "start_offset": "0.410s", "end_offset": "0.700s"},
                            {"type": "word_info", "text": "讨论", "start_offset": "0.710s", "end_offset": "1.100s"},
                            {"type": "word_info", "text": "数字化", "start_offset": "1.110s", "end_offset": "1.600s"},
                            {"type": "word_info", "text": "转型", "start_offset": "1.610s", "end_offset": "2.000s"},
                            {"type": "word_info", "text": "，", "start_offset": "2.000s", "end_offset": "2.050s"},
                            {"type": "word_info", "text": "首先", "start_offset": "2.800s", "end_offset": "3.200s"},
                            {"type": "word_info", "text": "了解", "start_offset": "3.210s", "end_offset": "3.600s"},
                            {"type": "word_info", "text": "它的", "start_offset": "3.610s", "end_offset": "3.900s"},
                            {"type": "word_info", "text": "基本", "start_offset": "3.910s", "end_offset": "4.200s"},
                            {"type": "word_info", "text": "逻辑", "start_offset": "4.210s", "end_offset": "4.600s"},
                            {"type": "word_info", "text": "。", "start_offset": "4.600s", "end_offset": "4.650s"},
                        ]
                    }
                ]
            }
        ]
    }

    words = asr._extract_words(mock_resp)
    assert len(words) == 12, f"应提取出 12 个词，实际为 {len(words)}"
    assert words[0]["start_ms"] == 100
    assert words[0]["end_ms"] == 400
    assert words[-1]["text"] == "。"
    assert words[-1]["end_ms"] == 4650
    print("  [PASS] 词级时间戳提取验证通过")
    return asr, words


def test_segment_aggregation(asr, words):
    print("[-] 测试词级时间戳聚合为 SRT 段落...")
    segments = asr._aggregate_words_to_segments(words)
    print(f"  聚合得到 {len(segments)} 条字幕:")
    for i, seg in enumerate(segments, 1):
        print(f"    {i}. [{seg.start_time}ms -> {seg.end_time}ms]: {seg.text} (字数: {len(seg.text)})")
        assert len(seg.text) <= 25, f"字幕段不能超过 25 字: '{seg.text}' (长度 {len(seg.text)})"

    asr_data = ASRData(segments)
    srt_text = asr_data.to_srt()
    print("  生成的 SRT 格式:")
    print("\n".join("    " + line for line in srt_text.strip().split("\n")))
    assert "-->" in srt_text, "SRT 必须包含时间戳箭头 -->"
    print("  [PASS] 聚合及 SRT 格式生成验证通过")


def test_long_sentence_constraint():
    print("[-] 测试超长连续文本自动按 <=25 字切分约束...")
    audio_path = os.path.join(os.path.dirname(__file__), "resources", "test.mp3")
    asr = GeminiASR(audio_path)

    # 构造一段连续超过 50 字但没有标点的词列表
    long_words = []
    text_corpus = "在企业推进全面数字化的进程当中业务流程重构以及技术架构升级是至关重要的两项核心任务需要全员协同"
    for i, ch in enumerate(text_corpus):
        long_words.append({
            "text": ch,
            "start_ms": i * 200,
            "end_ms": (i + 1) * 200,
        })

    segs = asr._aggregate_words_to_segments(long_words)
    assert len(segs) >= 2, "超长文本必须被切分为多个段落"
    for s in segs:
        assert len(s.text) <= 25, f"切分后的段落不能超过 25 字: '{s.text}' (字数: {len(s.text)})"
        assert s.end_time > s.start_time, "时间戳必须递增"
        print(f"    切分块: '{s.text}' ({len(s.text)}字)")
    print("  [PASS] 超长连续文本切分约束验证通过")


def test_english_and_mixed():
    print("[-] 测试中英混排与空格排版...")
    audio_path = os.path.join(os.path.dirname(__file__), "resources", "test.mp3")
    asr = GeminiASR(audio_path)
    mixed_words = [
        {"text": "使用", "start_ms": 0, "end_ms": 300},
        {"text": "Google", "start_ms": 310, "end_ms": 700},
        {"text": "Gemini", "start_ms": 710, "end_ms": 1100},
        {"text": "API", "start_ms": 1110, "end_ms": 1400},
        {"text": "转录", "start_ms": 1410, "end_ms": 1800},
        {"text": "字幕", "start_ms": 1810, "end_ms": 2200},
        {"text": "。", "start_ms": 2200, "end_ms": 2250},
    ]
    segs = asr._aggregate_words_to_segments(mixed_words)
    assert len(segs) == 1
    assert segs[0].text == "使用Google Gemini API转录字幕。"
    print(f"    混排结果: '{segs[0].text}'")
    print("  [PASS] 中英混排排版验证通过")


if __name__ == "__main__":
    test_gemini_asr_init()
    asr, words = test_word_extraction()
    test_segment_aggregation(asr, words)
    test_long_sentence_constraint()
    test_english_and_mixed()
    print("\n[ALL PASS] 所有 GeminiASR 单元测试均顺利通过！")
