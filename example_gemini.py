import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from bk_asr import GeminiASR

if __name__ == '__main__':
    audio_file = "resources/test.mp3"
    print(f"正在使用 Google Gemini 3.5 Transcribe (G接口) 转录音频: {audio_file}")
    
    # API Key 配置方式说明（三选一）：
    # 1. 在项目根目录下创建 gemini_key.txt 并填入您的 API Key（推荐）
    # 2. 设置环境变量 GEMINI_API_KEY
    # 3. 在实例化时直接传入: GeminiASR(audio_file, api_key="你的API_KEY")
    # 可在 Google AI Studio 免费申请 API Key: https://aistudio.google.com/app/apikey
    
    try:
        asr = GeminiASR(audio_file)
        result = asr.run()
        srt_content = result.to_srt()
        print("--- 识别生成的 SRT 字幕 ---")
        print(srt_content)
    except Exception as e:
        print(f"转录失败: {e}")
