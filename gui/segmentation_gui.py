from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from PySide6.QtCore import Qt, QProcess
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("segmentation_gui_config.yaml")
PREDICT_SCRIPT = REPO_ROOT / "segmentation" / "segment" / "predict.py"
TRAIN_SCRIPT = REPO_ROOT / "segmentation" / "segment" / "train.py"

DEFAULT_CONFIG: dict[str, object] = {
    "app": {"title": "Avalanche Segmentation Toolkit", "training_enabled": False},
    "inference": {
        "default_project": "segmentation/runs/predict-seg",
        "default_name_template": "compare_{stem}_full",
        "default_conf_threshold": 0.2,
        "default_save_json": True,
        "default_save_txt": True,
        "weights": [
            {"label": "full50 best.pt", "path": "segmentation/smoke_seg/full50/weights/best.pt"},
            {"label": "full150 best.pt", "path": "segmentation/smoke_seg/full150/weights/best.pt"},
        ],
    },
    "training": {
        "base_weights": "segmentation/yolov5n-seg.pt",
        "img_size": 320,
        "batch_size": 2,
        "workers": 0,
        "cache": "ram",
        "freeze": 10,
        "profiles": {
            "full": {"label": "Full", "epochs": 180, "noval": False},
            "medium": {"label": "Medium", "epochs": 2, "noval": True},
            "small": {"label": "Small", "epochs": 1, "noval": True},
        },
        "splits": {
            "50": {
                "full": "segmentation/data/avalanchesplit50.yaml",
                "medium": "segmentation/data/avalanchesplit50-medium.yaml",
                "small": "segmentation/data/avalanchesplit50-smoke.yaml",
            },
            "100": {
                "full": "segmentation/data/avalanchesplit100.yaml",
                "medium": "segmentation/data/avalanchesplit100.yaml",
                "small": "segmentation/data/avalanchesplit100.yaml",
            },
            "150": {
                "full": "segmentation/data/avalanchesplit150.yaml",
                "medium": "segmentation/data/avalanchesplit150.yaml",
                "small": "segmentation/data/avalanchesplit150.yaml",
            },
        },
    },
}


@dataclass(frozen=True)
class WeightOption:
    label: str
    path: str


def merge_dict(base: dict[str, object], overlay: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for key, value in base.items():
        overlay_value = overlay.get(key)
        if isinstance(value, dict):
            merged[key] = merge_dict(value, overlay_value if isinstance(overlay_value, dict) else {})
        elif isinstance(value, list):
            merged[key] = overlay_value if isinstance(overlay_value, list) else value
        else:
            merged[key] = overlay_value if overlay_value is not None else value
    for key, value in overlay.items():
        if key not in merged:
            merged[key] = value
    return merged


def load_config() -> dict[str, object]:
    if CONFIG_PATH.exists():
        loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            return merge_dict(DEFAULT_CONFIG, loaded)
    return DEFAULT_CONFIG


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def command_text(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts)


def weight_options(config: dict[str, object]) -> list[WeightOption]:
    options: list[WeightOption] = []
    for item in config["inference"]["weights"]:  # type: ignore[index]
        options.append(WeightOption(str(item["label"]), str(item["path"])))  # type: ignore[index]
    return options


class SegmentationWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.config = load_config()
        self.process = QProcess(self)
        self._auto_name = True

        self.setWindowTitle(str(self.config["app"]["title"]))  # type: ignore[index]
        self.resize(1120, 820)

        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._append_process_output)
        self.process.errorOccurred.connect(self._process_error)
        self.process.finished.connect(self._process_finished)

        tabs = QTabWidget()
        tabs.addTab(self._build_inference_tab(), "Segmentation")
        tabs.addTab(self._build_training_tab(), "Training")

        self.log_window = QPlainTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setPlaceholderText("Command output will appear here")

        root_widget = QWidget()
        root_layout = QVBoxLayout(root_widget)
        root_layout.addWidget(tabs, 4)
        root_layout.addWidget(QLabel("Log"))
        root_layout.addWidget(self.log_window, 1)
        self.setCentralWidget(root_widget)

    def _build_inference_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        description = QLabel(
            "Select an image, choose a weight file, and the GUI will generate the segmentation command for you."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form_box = QGroupBox("Segmentation command")
        form_layout = QFormLayout(form_box)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Path to the image you want to segment")
        source_browse = QPushButton("Browse")
        source_browse.clicked.connect(self._browse_source)
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_edit, 1)
        source_row.addWidget(source_browse)
        source_container = QWidget()
        source_container.setLayout(source_row)
        form_layout.addRow("Image", source_container)

        self.weights_combo = QComboBox()
        for option in weight_options(self.config):
            self.weights_combo.addItem(option.label, option.path)
        add_weight = QPushButton("Add weight")
        add_weight.clicked.connect(self._add_weight)
        weight_row = QHBoxLayout()
        weight_row.addWidget(self.weights_combo, 1)
        weight_row.addWidget(add_weight)
        weight_container = QWidget()
        weight_container.setLayout(weight_row)
        form_layout.addRow("Weights", weight_container)

        self.project_edit = QLineEdit(str(self.config["inference"]["default_project"]))  # type: ignore[index]
        project_browse = QPushButton("Browse")
        project_browse.clicked.connect(self._browse_project)
        project_row = QHBoxLayout()
        project_row.addWidget(self.project_edit, 1)
        project_row.addWidget(project_browse)
        project_container = QWidget()
        project_container.setLayout(project_row)
        form_layout.addRow("Output folder", project_container)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(str(self.config["inference"]["default_name_template"]))  # type: ignore[index]
        self.name_edit.textEdited.connect(self._mark_name_manual)
        form_layout.addRow("Output name", self.name_edit)

        self.save_json_checkbox = QCheckBox("Save JSON")
        self.save_json_checkbox.setChecked(bool(self.config["inference"]["default_save_json"]))  # type: ignore[index]
        self.save_txt_checkbox = QCheckBox("Save TXT")
        self.save_txt_checkbox.setChecked(bool(self.config["inference"]["default_save_txt"]))  # type: ignore[index]

        option_row = QHBoxLayout()
        option_row.addWidget(self.save_json_checkbox)
        option_row.addWidget(self.save_txt_checkbox)
        option_row.addStretch(1)
        option_container = QWidget()
        option_container.setLayout(option_row)
        form_layout.addRow("Exports", option_container)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(float(self.config["inference"]["default_conf_threshold"]))  # type: ignore[index]
        form_layout.addRow("Confidence", self.conf_spin)

        layout.addWidget(form_box)

        self.predict_preview = QPlainTextEdit()
        self.predict_preview.setReadOnly(True)
        self.predict_preview.setPlaceholderText("Command preview")
        layout.addWidget(self.predict_preview, 1)

        run_row = QHBoxLayout()
        self.run_predict_button = QPushButton("Run segmentation")
        self.run_predict_button.clicked.connect(self._run_predict)
        run_row.addWidget(self.run_predict_button)
        run_row.addStretch(1)
        run_container = QWidget()
        run_container.setLayout(run_row)
        layout.addWidget(run_container)

        self.source_edit.textChanged.connect(self._update_predict_preview)
        self.weights_combo.currentIndexChanged.connect(self._update_predict_preview)
        self.project_edit.textChanged.connect(self._update_predict_preview)
        self.name_edit.textChanged.connect(self._update_predict_preview)
        self.conf_spin.valueChanged.connect(self._update_predict_preview)
        self.save_json_checkbox.toggled.connect(self._update_predict_preview)
        self.save_txt_checkbox.toggled.connect(self._update_predict_preview)

        self._update_predict_preview()
        return widget

    def _build_training_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        warning = QLabel("Training is disabled in segmentation_gui_config.yaml to prevent accidental retraining.")
        warning.setWordWrap(True)
        warning.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        warning.setStyleSheet("padding: 8px;")
        layout.addWidget(warning)

        form_box = QGroupBox("Training command")
        form_layout = QFormLayout(form_box)

        self.training_profile_combo = QComboBox()
        training_profiles = self.config["training"]["profiles"]  # type: ignore[index]
        for key in ("full", "medium", "small"):
            profile = training_profiles[key]  # type: ignore[index]
            self.training_profile_combo.addItem(f'{profile["label"]} ({key})', key)
        form_layout.addRow("Profile", self.training_profile_combo)

        self.training_split_combo = QComboBox()
        for split in ("150", "100", "50"):
            self.training_split_combo.addItem(split, split)
        form_layout.addRow("Dataset split", self.training_split_combo)

        self.training_summary = QLabel()
        self.training_summary.setWordWrap(True)
        form_layout.addRow("Summary", self.training_summary)

        layout.addWidget(form_box)

        self.train_preview = QPlainTextEdit()
        self.train_preview.setReadOnly(True)
        self.train_preview.setPlaceholderText("Training command preview")
        layout.addWidget(self.train_preview, 1)

        run_row = QHBoxLayout()
        self.run_train_button = QPushButton("Run training")
        self.run_train_button.clicked.connect(self._run_training)
        run_row.addWidget(self.run_train_button)
        run_row.addStretch(1)
        run_container = QWidget()
        run_container.setLayout(run_row)
        layout.addWidget(run_container)

        self.training_profile_combo.currentIndexChanged.connect(self._update_training_preview)
        self.training_split_combo.currentIndexChanged.connect(self._update_training_preview)

        training_enabled = bool(self.config["app"]["training_enabled"])  # type: ignore[index]
        form_box.setEnabled(training_enabled)
        self.run_train_button.setEnabled(training_enabled)
        if not training_enabled:
            self.training_summary.setText(
                "Enable training_enabled in segmentation_gui_config.yaml if you want this tab to launch train.py."
            )

        self._update_training_preview()
        return widget

    def _default_output_name(self) -> str:
        source_text = self.source_edit.text().strip() if hasattr(self, "source_edit") else ""
        stem = Path(source_text).stem if source_text else "image"
        template = str(self.config["inference"]["default_name_template"])  # type: ignore[index]
        return template.format(stem=stem)

    def _mark_name_manual(self, _: str) -> None:
        self._auto_name = False
        self._update_predict_preview()

    def _browse_source(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            str(REPO_ROOT),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if file_path:
            self._auto_name = True
            self.source_edit.setText(file_path)
            self.name_edit.setText(self._default_output_name())
            self._update_predict_preview()

    def _browse_project(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select output folder", str(REPO_ROOT))
        if directory:
            self.project_edit.setText(directory)

    def _add_weight(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select weight file",
            str(REPO_ROOT),
            "PyTorch weights (*.pt);;All files (*.*)",
        )
        if not file_path:
            return
        existing_paths = {self.weights_combo.itemData(index) for index in range(self.weights_combo.count())}
        if file_path not in existing_paths:
            self.weights_combo.addItem(f"Custom: {Path(file_path).name}", file_path)
        self.weights_combo.setCurrentIndex(self.weights_combo.count() - 1)

    def _predict_command(self) -> list[str]:
        source = self.source_edit.text().strip()
        if self._auto_name and source:
            expected = self._default_output_name()
            if self.name_edit.text().strip() != expected:
                self.name_edit.blockSignals(True)
                self.name_edit.setText(expected)
                self.name_edit.blockSignals(False)

        command = [
            sys.executable,
            str(PREDICT_SCRIPT),
            "--weights",
            str(self.weights_combo.currentData()),
            "--source",
            source,
            "--conf-thres",
            f"{self.conf_spin.value():.2f}",
            "--project",
            self.project_edit.text().strip(),
            "--name",
            self.name_edit.text().strip() or self._default_output_name(),
            "--exist-ok",
        ]
        if self.save_json_checkbox.isChecked():
            command.append("--save-json")
        if self.save_txt_checkbox.isChecked():
            command.append("--save-txt")
        return command

    def _training_command(self) -> list[str]:
        profile_key = str(self.training_profile_combo.currentData())
        split = str(self.training_split_combo.currentData())
        training = self.config["training"]  # type: ignore[index]
        profile = training["profiles"][profile_key]  # type: ignore[index]
        split_config = training["splits"][split]  # type: ignore[index]
        data_path = split_config.get(profile_key, split_config["full"])
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--weights",
            str(resolve_repo_path(training["base_weights"])),  # type: ignore[index]
            "--data",
            str(resolve_repo_path(data_path)),
            "--img",
            str(training["img_size"]),
            "--epochs",
            str(profile["epochs"]),
            "--batch-size",
            str(training["batch_size"]),
            "--workers",
            str(training["workers"]),
            "--cache",
            str(training["cache"]),
            "--freeze",
            str(training["freeze"]),
        ]
        if profile.get("noval", False):
            command.append("--noval")
        return command

    def _update_predict_preview(self) -> None:
        self.predict_preview.setPlainText(command_text(self._predict_command()))

    def _update_training_preview(self) -> None:
        command = self._training_command()
        self.train_preview.setPlainText(command_text(command))
        profile_key = str(self.training_profile_combo.currentData())
        split = str(self.training_split_combo.currentData())
        training = self.config["training"]  # type: ignore[index]
        profile = training["profiles"][profile_key]  # type: ignore[index]
        split_config = training["splits"][split]  # type: ignore[index]
        data_path = split_config.get(profile_key, split_config["full"])
        self.training_summary.setText(
            f'{profile["label"]} training on split {split} uses {Path(str(data_path)).name}, '
            f'{profile["epochs"]} epochs, img {training["img_size"]}, batch {training["batch_size"]}.'
        )

    def _start_process(self, command: list[str], title: str) -> None:
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Busy", "A command is already running.")
            return
        self.log_window.clear()
        self._append_log(f"{title}: {command_text(command)}")
        self.process.setProgram(command[0])
        self.process.setArguments(command[1:])
        self.process.setWorkingDirectory(str(REPO_ROOT))
        self.process.start()

    def _run_predict(self) -> None:
        if not self.source_edit.text().strip():
            QMessageBox.warning(self, "Missing source", "Choose an image before running segmentation.")
            return
        self._start_process(self._predict_command(), "Segmentation")

    def _run_training(self) -> None:
        if not bool(self.config["app"]["training_enabled"]):  # type: ignore[index]
            QMessageBox.information(
                self,
                "Training disabled",
                "Training is disabled by configuration. Enable it in segmentation_gui_config.yaml first.",
            )
            return
        self._start_process(self._training_command(), "Training")

    def _append_log(self, text: str) -> None:
        self.log_window.appendPlainText(text)

    def _append_process_output(self) -> None:
        output = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if output.strip():
            self._append_log(output.rstrip())

    def _process_error(self, error: QProcess.ProcessError) -> None:
        self._append_log(f"Process error: {error.name}")

    def _process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._append_log(f"Finished with exit code {exit_code} ({exit_status.name})")


def main() -> None:
    app = QApplication(sys.argv)
    window = SegmentationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
