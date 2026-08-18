from bk_asr import WhisperASR


if __name__ == '__main__':
    audio_file = "resources/test.mp3"
    asr = WhisperASR(audio_file)
    result = asr.run()
    srt_content = result.to_srt()
    print("--- 识别生成的 SRT 字幕 ---")
    print(srt_content)