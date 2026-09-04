import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import webbrowser

import imageio_ffmpeg

# FIX: 修复中文路径报错 https://github.com/WEIFENG2333/AsrTools/issues/18  设置QT_QPA_PLATFORM_PLUGIN_PATH 
plugin_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

from PyQt5.QtCore import Qt, QRunnable, QThreadPool, QObject, pyqtSignal as Signal, pyqtSlot as Slot, QSize, QThread, \
    pyqtSignal
from PyQt5.QtGui import QCursor, QColor, QFont
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog,
                             QTableWidgetItem, QHeaderView, QSizePolicy)
from qfluentwidgets import (ComboBox, PushButton, LineEdit, TableWidget, FluentIcon as FIF,
                            Action, RoundMenu, InfoBar, InfoBarPosition,
                            FluentWindow, BodyLabel, MessageBox)

from bk_asr.WhisperASR import WhisperASR, list_whisper_models, get_whisper_models_dir
from bk_asr.BcutASR import BcutASR
from bk_asr.JianYingASR import JianYingASR
from bk_asr.KuaiShouASR import KuaiShouASR

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

_ffmpeg_gui_lock = threading.Lock()


def ensure_ffmpeg_available() -> str:
    """获取可用的 ffmpeg 执行文件路径并设置环境变量（线程安全）"""
    if shutil.which("ffmpeg") is not None:
        return "ffmpeg"
    with _ffmpeg_gui_lock:
        if shutil.which("ffmpeg") is not None:
            return "ffmpeg"
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
            target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
            if not os.path.exists(target_ffmpeg):
                shutil.copyfile(ffmpeg_exe, target_ffmpeg)
            if ffmpeg_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            return target_ffmpeg
        except Exception as e:
            logging.warning(f"获取 ffmpeg 失败: {e}")
            return "ffmpeg"


class WorkerSignals(QObject):
    finished = Signal(str, str)
    errno = Signal(str, str)


class ASRWorker(QRunnable):
    """ASR处理工作线程"""
    def __init__(self, file_path, asr_engine, export_format, whisper_model=None):
        super().__init__()
        # 禁用 QThreadPool 对 QRunnable 的自动析构，防止 Python 引用与 Qt 跨线程信号派发冲突导致崩溃
        self.setAutoDelete(False)
        self.file_path = file_path
        self.asr_engine = asr_engine
        self.export_format = export_format
        self.whisper_model = whisper_model
        self.signals = WorkerSignals()

        self.audio_path = None

    @Slot()
    def run(self):
        try:
            use_cache = True
            
            # 检查文件类型,如果不是音频则通过 ffmpeg 提取 mp3 音频
            logging.info("[+] 正在检查文件类型并准备音频")
            audio_exts = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma']
            if not any(self.file_path.lower().endswith(ext) for ext in audio_exts):
                logging.info(f"[+] 正在使用 ffmpeg 从视频中抽取音频: {self.file_path}")
                temp_audio = self.file_path.rsplit(".", 1)[0] + ".mp3"
                if not video2audio(self.file_path, temp_audio):
                    raise Exception("视频音频抽取失败，请确保视频包含有效音频轨且 ffmpeg 正常工作")
                if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
                    raise Exception("提取生成的音频文件为空或无效")
                self.audio_path = temp_audio
            else:
                self.audio_path = self.file_path
            
            # 根据选择的 ASR 引擎实例化相应的类
            if 'whisper' in self.asr_engine.lower():
                asr = WhisperASR(self.audio_path, model_path=self.whisper_model, use_cache=use_cache)
            elif self.asr_engine == 'B 接口':
                asr = BcutASR(self.audio_path, use_cache=use_cache)
            elif self.asr_engine == 'J 接口':
                asr = JianYingASR(self.audio_path, use_cache=use_cache)
            elif self.asr_engine == 'K 接口':
                asr = KuaiShouASR(self.audio_path, use_cache=use_cache)
            else:
                asr = WhisperASR(self.audio_path, model_path=self.whisper_model, use_cache=use_cache)

            logging.info(f"开始处理文件: {self.file_path} 使用引擎: {self.asr_engine}")
            result = asr.run()
            
            # 根据导出格式选择转换方法
            save_ext = self.export_format.lower()
            if save_ext == 'srt':
                result_text = result.to_srt()
            elif save_ext == 'ass':
                result_text = result.to_ass()
            elif save_ext == 'txt':
                result_text = result.to_txt()
            else:
                result_text = result.to_srt()
                
            logging.info(f"完成处理文件: {self.file_path} 使用引擎: {self.asr_engine}")
            save_path = self.file_path.rsplit(".", 1)[0] + "." + save_ext
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(result_text)
            self.signals.finished.emit(self.file_path, result_text)
        except Exception as e:
            logging.error(f"处理文件 {self.file_path} 时出错: {str(e)}")
            self.signals.errno.emit(self.file_path, f"处理时出错: {str(e)}")

class UpdateCheckerThread(QThread):
    msg = pyqtSignal(str, str, str)  # 用于发送消息的信号

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        try:
            from check_update import check_update, check_internet_connection
            # 检查互联网连接
            if not check_internet_connection():
                self.msg.emit("错误", "无法连接到互联网，请检查网络连接。", "")
                return
            # 检查更新
            config = check_update(self)
            if config:
                if config['fource']:
                    self.msg.emit("更新", "检测到新版本，请下载最新版本。", config['update_download_url'])
                else:
                    self.msg.emit("可更新", "检测到新版本，请下载最新版本。", config['update_download_url'])
        except Exception as e:
            pass


class ASRWidget(QWidget):
    """ASR处理界面"""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.max_threads = 3  # 设置最大线程数
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(self.max_threads)
        self.processing_queue = []
        self.workers = {}  # 维护文件路径到worker的映射


    def init_ui(self):
        layout = QVBoxLayout(self)

        # ASR引擎选择区域
        engine_layout = QHBoxLayout()
        engine_label = BodyLabel("选择引擎:", self)
        engine_label.setFixedWidth(85)
        self.combo_box = ComboBox(self)
        self.combo_box.addItems(['Whisper.cpp (本地)', 'J 接口', 'B 接口', 'K 接口'])
        self.combo_box.currentTextChanged.connect(self.on_engine_changed)
        engine_layout.addWidget(engine_label)
        engine_layout.addWidget(self.combo_box)
        layout.addLayout(engine_layout)

        # Whisper 模型选择区域 (读取 resources\whisper 目录)
        self.model_layout = QHBoxLayout()
        self.model_label = BodyLabel("Whisper模型:", self)
        self.model_label.setFixedWidth(85)
        self.model_combo = ComboBox(self)
        self.refresh_model_btn = PushButton("刷新模型", self)
        self.refresh_model_btn.setFixedWidth(80)
        self.refresh_model_btn.clicked.connect(self.refresh_whisper_models)
        self.open_model_dir_btn = PushButton("打开模型目录", self)
        self.open_model_dir_btn.setFixedWidth(100)
        self.open_model_dir_btn.clicked.connect(self.open_models_directory)

        self.model_layout.addWidget(self.model_label)
        self.model_layout.addWidget(self.model_combo)
        self.model_layout.addWidget(self.refresh_model_btn)
        self.model_layout.addWidget(self.open_model_dir_btn)
        layout.addLayout(self.model_layout)

        # 导出格式选择区域 
        format_layout = QHBoxLayout()
        format_label = BodyLabel("导出格式:", self)
        format_label.setFixedWidth(85)
        self.format_combo = ComboBox(self)
        self.format_combo.addItems(['SRT', 'TXT', 'ASS'])
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        layout.addLayout(format_layout)

        # 文件选择区域
        file_layout = QHBoxLayout()
        self.file_input = LineEdit(self)
        self.file_input.setPlaceholderText("拖拽文件或文件夹到这里")
        self.file_input.setReadOnly(True)
        self.file_button = PushButton("选择文件", self)
        self.file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.file_button)
        layout.addLayout(file_layout)

        # 文件列表表格
        self.table = TableWidget(self)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['文件名', '状态'])
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.table)

        # 设置表格列的拉伸模式
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 100)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 处理按钮
        self.process_button = PushButton("开始处理", self)
        self.process_button.clicked.connect(self.process_files)
        self.process_button.setEnabled(False)  # 初始禁用
        layout.addWidget(self.process_button)

        self.setAcceptDrops(True)
        self.refresh_whisper_models()

    def on_engine_changed(self, engine_text):
        is_whisper = 'whisper' in engine_text.lower()
        self.model_label.setEnabled(is_whisper)
        self.model_combo.setEnabled(is_whisper)
        self.refresh_model_btn.setEnabled(is_whisper)
        self.open_model_dir_btn.setEnabled(is_whisper)

    def refresh_whisper_models(self):
        """刷新 resources/whisper 目录中的模型文件"""
        models = list_whisper_models()
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
            if "ggml-base.bin" in models:
                self.model_combo.setCurrentText("ggml-base.bin")
        else:
            self.model_combo.addItem("未检测到模型 (请放入 resources/whisper)")

    def open_models_directory(self):
        """打开 resources/whisper 文件夹，方便用户放置新模型"""
        models_dir = get_whisper_models_dir()
        try:
            if platform.system() == "Windows":
                os.startfile(models_dir)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", models_dir])
            else:
                subprocess.Popen(["xdg-open", models_dir])
        except Exception as e:
            InfoBar.error(
                title='无法打开目录',
                content=str(e),
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )

    def select_file(self):
        """选择文件对话框"""
        files, _ = QFileDialog.getOpenFileNames(self, "选择音频或视频文件", "",
                                                "Media Files (*.mp3 *.wav *.ogg *.flac *.m4a *.mp4 *.avi *.mov *.ts *.mkv *.flv *.webm)")
        for file in files:
            self.add_file_to_table(file)
        self.update_start_button_state()

    def add_file_to_table(self, file_path):
        """将文件添加到表格中"""
        if self.find_row_by_file_path(file_path) != -1:
            InfoBar.warning(
                title='文件已存在',
                content=f"文件 {os.path.basename(file_path)} 已经添加到列表中。",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            return

        row_count = self.table.rowCount()
        self.table.insertRow(row_count)
        item_filename = self.create_non_editable_item(os.path.basename(file_path))
        item_status = self.create_non_editable_item("未处理")
        item_status.setForeground(QColor("gray"))
        self.table.setItem(row_count, 0, item_filename)
        self.table.setItem(row_count, 1, item_status)
        item_filename.setData(Qt.UserRole, file_path)

    def create_non_editable_item(self, text):
        """创建不可编辑的表格项"""
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def show_context_menu(self, pos):
        """显示右键菜单"""
        current_row = self.table.rowAt(pos.y())
        if current_row < 0:
            return

        self.table.selectRow(current_row)

        menu = RoundMenu(parent=self)
        reprocess_action = Action(FIF.SYNC, "重新处理")
        delete_action = Action(FIF.DELETE, "删除任务")
        open_dir_action = Action(FIF.FOLDER, "打开文件目录")
        menu.addActions([reprocess_action, delete_action, open_dir_action])

        delete_action.triggered.connect(self.delete_selected_row)
        open_dir_action.triggered.connect(self.open_file_directory)
        reprocess_action.triggered.connect(self.reprocess_selected_file)

        menu.exec(QCursor.pos())

    def delete_selected_row(self):
        """删除选中的行"""
        current_row = self.table.currentRow()
        if current_row >= 0:
            file_path = self.table.item(current_row, 0).data(Qt.UserRole)
            if file_path in self.workers:
                worker = self.workers[file_path]
                worker.signals.finished.disconnect(self.update_table)
                worker.signals.errno.disconnect(self.handle_error)
                self.workers.pop(file_path, None)
            self.table.removeRow(current_row)
            self.update_start_button_state()

    def open_file_directory(self):
        """打开文件所在目录"""
        current_row = self.table.currentRow()
        if current_row >= 0:
            current_item = self.table.item(current_row, 0)
            if current_item:
                file_path = current_item.data(Qt.UserRole)
                directory = os.path.dirname(file_path)
                try:
                    if platform.system() == "Windows":
                        os.startfile(directory)
                    elif platform.system() == "Darwin":
                        subprocess.Popen(["open", directory])
                    else:
                        subprocess.Popen(["xdg-open", directory])
                except Exception as e:
                    InfoBar.error(
                        title='无法打开目录',
                        content=str(e),
                        orient=Qt.Horizontal,
                        isClosable=True,
                        position=InfoBarPosition.TOP,
                        duration=3000,
                        parent=self
                    )

    def reprocess_selected_file(self):
        """重新处理选中的文件"""
        current_row = self.table.currentRow()
        if current_row >= 0:
            file_path = self.table.item(current_row, 0).data(Qt.UserRole)
            status = self.table.item(current_row, 1).text()
            if status == "处理中":
                InfoBar.warning(
                    title='当前文件正在处理中',
                    content="请等待当前文件处理完成后再重新处理。",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
                return
            self.add_to_queue(file_path)

    def add_to_queue(self, file_path):
        """将文件添加到处理队列并更新状态"""
        self.processing_queue.append(file_path)
        self.process_next_in_queue()

    def process_files(self):
        """处理所有未处理的文件"""
        for row in range(self.table.rowCount()):
            if self.table.item(row, 1).text() == "未处理":
                file_path = self.table.item(row, 0).data(Qt.UserRole)
                self.processing_queue.append(file_path)
        self.process_next_in_queue()

    def process_next_in_queue(self):
        """处理队列中的下一个文件"""
        while self.thread_pool.activeThreadCount() < self.max_threads and self.processing_queue:
            file_path = self.processing_queue.pop(0)
            if file_path not in self.workers:
                self.process_file(file_path)

    def process_file(self, file_path):
        """处理单个文件"""
        selected_engine = self.combo_box.currentText()
        selected_format = self.format_combo.currentText()
        selected_model = self.model_combo.currentText() if self.model_combo.isEnabled() else None
        worker = ASRWorker(file_path, selected_engine, selected_format, whisper_model=selected_model)
        worker.signals.finished.connect(self.update_table)
        worker.signals.errno.connect(self.handle_error)
        self.thread_pool.start(worker)
        self.workers[file_path] = worker

        row = self.find_row_by_file_path(file_path)
        if row != -1:
            status_item = self.create_non_editable_item("处理中")
            status_item.setForeground(QColor("orange"))
            self.table.setItem(row, 1, status_item)
            self.update_start_button_state()

    def update_table(self, file_path, result):
        """更新表格中文件的处理状态"""
        row = self.find_row_by_file_path(file_path)
        if row != -1:
            item_status = self.create_non_editable_item("已处理")
            item_status.setForeground(QColor("green"))
            self.table.setItem(row, 1, item_status)

            InfoBar.success(
                title='处理完成',
                content=f"文件 {self.table.item(row, 0).text()} 已处理完成",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=1500,
                parent=self
            )

        self.workers.pop(file_path, None)
        self.process_next_in_queue()
        self.update_start_button_state()

    def handle_error(self, file_path, error_message):
        """处理错误信息"""
        row = self.find_row_by_file_path(file_path)
        if row != -1:
            item_status = self.create_non_editable_item("错误")
            item_status.setForeground(QColor("red"))
            self.table.setItem(row, 1, item_status)

            InfoBar.error(
                title='处理出错',
                content=error_message,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )

        self.workers.pop(file_path, None)
        self.process_next_in_queue()
        self.update_start_button_state()

    def find_row_by_file_path(self, file_path):
        """根据文件路径查找表格中的行号"""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item.data(Qt.UserRole) == file_path:
                return row
        return -1

    def update_start_button_state(self):
        """根据文件列表更新开始处理按钮的状态"""
        has_unprocessed = any(
            self.table.item(row, 1).text() == "未处理"
            for row in range(self.table.rowCount())
        )
        self.process_button.setEnabled(has_unprocessed)

    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """拖拽释放事件"""
        supported_formats = ('.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma',  # 音频格式
                           '.mp4', '.avi', '.mov', '.ts', '.mkv', '.wmv', '.flv', '.webm', '.rmvb')  # 视频格式
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for file in files:
            if os.path.isdir(file):
                for root, dirs, files_in_dir in os.walk(file):
                    for f in files_in_dir:
                        if f.lower().endswith(supported_formats):
                            self.add_file_to_table(os.path.join(root, f))
            elif file.lower().endswith(supported_formats):
                self.add_file_to_table(file)
        self.update_start_button_state()


class InfoWidget(QWidget):
    """个人信息界面"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        GITHUB_URL = "https://github.com/WEIFENG2333/AsrTools"
        REPO_DESCRIPTION = """
    🚀 本地离线高精准：基于 Whisper.cpp 本地模型进行语音识别，支持自由更换 resources/whisper 模型文件。
    🖥️ 高颜值界面：基于 PyQt5 和 qfluentwidgets，界面美观且用户友好。
    ⚡ 效率超人：多线程并发 + 批量处理，视频自动抽取音频并极速转换。
    📄 多格式支持：支持生成 .srt、.txt、.ass 字幕文件，满足不同需求。
        """
        
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignTop)

        # 标题
        title_label = BodyLabel("  ASRTools (Whisper.cpp)", self)
        title_label.setFont(QFont("Segoe UI", 30, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # 仓库描述区域
        desc_label = BodyLabel(REPO_DESCRIPTION, self)
        desc_label.setFont(QFont("Segoe UI", 12))
        main_layout.addWidget(desc_label)

        github_button = PushButton("GitHub 仓库", self)
        github_button.setIcon(FIF.GITHUB)
        github_button.setIconSize(QSize(20, 20))
        github_button.setMinimumHeight(42)
        github_button.clicked.connect(lambda _: webbrowser.open(GITHUB_URL))
        main_layout.addWidget(github_button)


class MainWindow(FluentWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ASR Processing Tool')

        # ASR 处理界面
        self.asr_widget = ASRWidget()
        self.asr_widget.setObjectName("main")
        self.addSubInterface(self.asr_widget, FIF.ALBUM, 'ASR Processing')

        # 个人信息界面
        self.info_widget = InfoWidget()
        self.info_widget.setObjectName("info")
        self.addSubInterface(self.info_widget, FIF.GITHUB, 'About')

        self.navigationInterface.setExpandWidth(200)
        self.resize(850, 620)

        self.update_checker = UpdateCheckerThread(self)
        self.update_checker.msg.connect(self.show_msg)
        self.update_checker.start()

    def show_msg(self, title, content, update_download_url):
        w = MessageBox(title, content, self)
        if w.exec() and update_download_url:
            webbrowser.open(update_download_url)
        if title == "更新":
            sys.exit(0)

def video2audio(input_file: str, output: str = "") -> bool:
    """使用 ffmpeg 将视频转换为音频（健壮防闪退版）"""
    try:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.warning(f"创建目标目录失败，尝试使用临时目录: {e}")
        output = os.path.join(tempfile.gettempdir(), Path(output).name)

    output = str(output)
    ffmpeg_bin = ensure_ffmpeg_available()

    # -vn: 禁用视频流解码（大幅降低CPU/内存开销，彻底避免 4K/8K 或异常视频编码导致解码崩溃）
    # -sn: 禁用字幕流解码
    # -ac 1: 转换单声道
    # -f mp3: 输出 mp3
    cmd = [
        ffmpeg_bin,
        '-i', input_file,
        '-vn',
        '-sn',
        '-ac', '1',
        '-f', 'mp3',
        '-af', 'aresample=async=1',
        '-y',
        output
    ]
    try:
        kwargs = {}
        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
            **kwargs
        )
        if result.returncode == 0 and Path(output).is_file() and os.path.getsize(output) > 0:
            return True
        else:
            err_msg = result.stderr.strip()[-500:] if result.stderr else "未知错误"
            logging.error(f"ffmpeg 音频抽取失败 (返回码 {result.returncode}): {err_msg}")
            return False
    except Exception as e:
        logging.error(f"调用 ffmpeg 异常: {e}")
        return False

def start():
    # enable dpi scale
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    start()
