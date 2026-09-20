# 🎤 AsrTools

基于 AsrTools 修改：
- 修复 J 接口
- 加入 Whisper 本地模式
- 新增 G 接口（Google Gemini 3.5 语音转写）

## 🌟 **使用**
下载项目点击 `启动ASR工具.bat`（或运行 `python asr_gui.py`）

### 界面操作

- **引擎选择** 下拉框：支持 `Whisper.cpp (本地)`、`J 接口`、`B 接口`、`G 接口`
- **Whisper模型** 下拉框：选择本地模式时，自动列出 `resources\whisper\` 下所有可用模型
- **刷新模型** 按钮：放入新模型文件后点击，即时刷新列表
- **打开模型目录** 按钮：一键打开模型文件夹，方便放入新模型
- **B接口 / J接口**：保持联网直接使用
- **G接口**：基于 Google Gemini 3.5 Transcribe 语音转录，需配置 API KEY（详见下方说明）

### G 接口 API KEY 配置说明

使用 G 接口需要配置 Google AI Studio API Key（可前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 免费获取）。

**填 API KEY 的位置（任选以下一种方式即可）：**

- **方式一（推荐，最简便）**：
  在项目根目录下新建一个名为 `gemini_key.txt` 的文本文件，将获取到的 API Key 粘贴进去保存即可（已加入 `.gitignore`，不会被提交）。

- **方式二（修改配置文件）**：
  打开 `bk_asr/GeminiASR.py` 文件，在第 16 行填入：
  ```python
  DEFAULT_GEMINI_API_KEY = "你的API_KEY"
  ```

- **方式三（环境变量）**：
  在系统中设置环境变量 `GEMINI_API_KEY` 为你的 Key。

- **方式四（代码调用）**：
  若在 Python 脚本中使用，可在实例化时直接传入：
  ```python
  from bk_asr import GeminiASR
  asr = GeminiASR("audio.mp3", api_key="你的API_KEY")
  ```

> **注**：G 接口原生获取词级时间戳，单行字幕长度限制在 18 个字符以内（超过即换行），遇到句号自动换行；文本块换行后，字幕中除顿号（`、`）、括号、数学符号及间隔号（`·`）外，其余标点符号自动转换为空格。

### 模型下载

可从 HuggingFace 下载 ggml 格式的 Whisper 模型：
- https://huggingface.co/ggerganov/whisper.cpp/tree/main

**主界面截图示例：**

<img src="resources/main_window-mod.png" width="80%" alt="主界面">



