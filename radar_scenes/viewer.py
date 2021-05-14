import json
import os
import sys
import argparse
from PySide2 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import qdarkstyle
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import enum
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from pkg_resources import resource_filename

from radar_scenes.sequence import Sequence
from radar_scenes.coordinate_transformation import transform_detections_sequence_to_car
from radar_scenes.sensors import get_mounting
from radar_scenes.colors import Colors, brush_for_color
from radar_scenes.evaluation import PredictionFileSchemas
from radar_scenes.labels import Label


class LoadSequenceWorker(QtCore.QObject):
    finished = QtCore.Signal()
    loading_done = QtCore.Signal(object, object)
    loading_failed = QtCore.Signal()

    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def load(self):
        try:
            sequence = Sequence.from_json(self.filename)
            cur_timestamp = sequence.first_timestamp
            timestamps = [cur_timestamp]
            while True:
                cur_timestamp = sequence.next_timestamp_after(cur_timestamp)
                if cur_timestamp is None:
                    break
                timestamps.append(cur_timestamp)
            self.loading_done.emit(sequence, timestamps)

            self.finished.emit()
        except:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)
            self.loading_failed.emit()
            self.finished.emit()


class ColorOpts(enum.Enum):
    """
    An enum containing the possible coloring options for the detections. This enum is used to fill the option list
    in the gui.
    """
    DOPPLER = "Doppler Velocity"
    RCS = "RCS"
    SENSORID = "Sensor ID"
    PREDLABEL = "Predicted Label ID"
    PREDTRACK = "Predicted Track ID"
    TRUELABEL = "True Label ID"
    TRUETRACK = "True Track ID"
    TRUEFALSE = "True/False Prediction"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.create_ui()

        self.setWindowIcon(QtGui.QIcon("res/icon.png"))

        self.showMaximized()
        self.sequence = None
        self.current_cam_filename = None
        self.timestamps = []

        self.plot_item.plotItem.ctrlMenu = None # deactivate plot options like fft
        self.plot_widget = PlotWidget(self.plot_item.plotItem)

        self.plot_widget.info_label = self.detection_info_label

        self._reset_predictions()

        self.timeline_spinbox.valueChanged.connect(self.timeline_slider.setValue)
        self.future_frames_spinbox.valueChanged.connect(self.plot_frames)
        self.prev_frames_spinbox.valueChanged.connect(self.plot_frames)
        self.timeline_slider.valueChanged.connect(self.on_slider_value_changed)
        self.color_by_list.currentIndexChanged.connect(self.plot_frames)
        self.doppler_scale_slider.valueChanged.connect(self.plot_frames)
        self.label_text_cb.stateChanged.connect(self.plot_frames)
        self.convex_hulls_cb.stateChanged.connect(self.plot_frames)
        self.doppler_arrows_cb.stateChanged.connect(self.on_doppler_cb_clicked)

    def create_ui(self):
        """
        Setup the whole UI of the radar data viewer. Creates several member variables for sliders, comboBoxes etc.
        :return: None
        """
        self.setWindowTitle("Radar Data Viewer")

        self.main_grid_layout = QtWidgets.QGridLayout()
        self.central_widget = QtWidgets.QWidget()
        self.central_widget.setLayout(self.main_grid_layout)
        self.setCentralWidget(self.central_widget)

        self.plot_item = pg.PlotWidget()
        self.plot_item.setBackground("#0B0F14")
        self.main_grid_layout.addWidget(self.plot_item, 0, 0)

        # Options Dock Widget
        self.options_layout = QtWidgets.QVBoxLayout()
        self.options_layout.setAlignment(QtCore.Qt.AlignTop)

        self.doppler_arrows_cb = QtWidgets.QCheckBox("Doppler Velocity Arrows")
        self.doppler_arrows_cb.setChecked(True)
        self.options_layout.addWidget(self.doppler_arrows_cb)

        # scaling of doppler arrows
        self.doppler_h_layout = QtWidgets.QHBoxLayout()
        self.doppler_h_layout.setContentsMargins(20, 0, 0, 10)
        self.doppler_scale_label = QtWidgets.QLabel()
        self.doppler_scale_label.setText("Scale")
        self.doppler_scale_label.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self.doppler_h_layout.addWidget(self.doppler_scale_label)

        self.doppler_scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.doppler_scale_slider.setMinimum(1)
        self.doppler_scale_slider.setMaximum(20)
        self.doppler_scale_slider.setValue(7)
        self.doppler_scale_slider.setFixedWidth(150)
        self.doppler_h_layout.addWidget(self.doppler_scale_slider)

        self.options_layout.addLayout(self.doppler_h_layout)

        self.label_text_cb = QtWidgets.QCheckBox("Class Names of True Label")
        self.label_text_cb.setChecked(True)
        self.options_layout.addWidget(self.label_text_cb)

        self.convex_hulls_cb = QtWidgets.QCheckBox("Convex Hulls Around True Objects")
        self.options_layout.addWidget(self.convex_hulls_cb)

        self.color_by_label = QtWidgets.QLabel()
        self.color_by_label.setText("Color Detections by")
        self.color_by_label.setStyleSheet(
            """
            QLabel {
            margin-top: 10px;
            }
            """
        )
        self.options_layout.addWidget(self.color_by_label)

        self.color_by_list = QtWidgets.QComboBox()
        self.color_by_list.setStyleSheet(
            """
            QComboBox::item:checked {
            height: 12px;
            border: 1px solid #32414B;
            margin-top: 0px;
            margin-bottom: 0px;
            padding: 4px;
            padding-left: 0px;
            }
            QComboBox {
            margin-left: 20px;
            }
            """
        )
        self.color_by_list.addItems([x.value for x in ColorOpts])
        self.color_by_list.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self.color_by_list.setCurrentIndex(6)
        self.options_layout.addWidget(self.color_by_list)

        options_widget = QtWidgets.QWidget()
        options_widget.setLayout(self.options_layout)
        self.options_dock = QtWidgets.QDockWidget("Options", self)
        self.options_dock.setWidget(options_widget)
        self.options_dock.setFloating(False)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.options_dock)

        # Detection Info Dock Widget
        self.info_dock = QtWidgets.QDockWidget("Information", self)
        self.info_dock.setMinimumWidth(350)
        self.info_dock.setMinimumHeight(200)
        self.detection_info_label = QtWidgets.QTextEdit(text="No detection selected.")
        self.detection_info_label.setMinimumWidth(200)
        flags = QtCore.Qt.TextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detection_info_label.setTextInteractionFlags(flags)
        self.info_dock.setWidget(self.detection_info_label)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.info_dock)

        # Camera Dock Widget
        self.cam_graphics_layout = pg.GraphicsLayoutWidget()
        self.cam_viewbox = self.cam_graphics_layout.addViewBox()
        self.cam_viewbox.setAspectLocked(True)
        self.camera_widget = pg.ImageItem()
        self.camera_widget.rotate(-90)
        self.cam_viewbox.addItem(self.camera_widget)
        self.camera_dock = QtWidgets.QDockWidget("Camera", self)
        self.camera_dock.setMinimumWidth(300)
        self.camera_dock.setMinimumHeight(300)
        self.camera_dock.setWidget(self.cam_graphics_layout)
        self.camera_dock.setFloating(False)
        self.camera_dock.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding))
        self.camera_dock.widget().setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.camera_dock)

        # Timeline: Slider, Spinboxes, Labes
        self.timeline_grid_layout = QtWidgets.QGridLayout()
        self.timeline_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.timeline_slider.setStyleSheet("""
        .QSlider {
        min-height: 20px;
        max-height: 20px;
        }
        .QSlider::groove:horizontal {
        height: 15px;
        }
        """)

        self.timeline_spinbox = QtWidgets.QSpinBox()
        self.timeline_spinbox.setMaximum(10000)
        self.timeline_spinbox.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self.timeline_label = QtWidgets.QLabel()
        self.timeline_label.setText("Current Frame:")
        self.timeline_label.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self.prev_frames_spinbox = QtWidgets.QSpinBox()
        self.prev_frames_spinbox.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self.prev_frames_spinbox.setMaximum(30)
        self.prev_frames_label = QtWidgets.QLabel()
        self.prev_frames_label.setText("Show Previous Frames:")
        self.prev_frames_label.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self.future_frames_spinbox = QtWidgets.QSpinBox()
        self.future_frames_spinbox.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self.future_frames_spinbox.setMaximum(30)
        self.future_frames_spinbox.setValue(4)
        self.future_frames_label = QtWidgets.QLabel()
        self.future_frames_label.setText("Show Future Frames:")
        self.future_frames_label.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))

        self.timeline_grid_layout.addWidget(self.timeline_slider, 0, 0, 1, 8)
        self.timeline_grid_layout.addWidget(self.timeline_label, 1, 1)
        self.timeline_grid_layout.addWidget(self.timeline_spinbox, 1, 2)
        self.timeline_grid_layout.addWidget(self.prev_frames_label, 1, 3)
        self.timeline_grid_layout.addWidget(self.prev_frames_spinbox, 1, 4)
        self.timeline_grid_layout.addWidget(self.future_frames_label, 1, 5)
        self.timeline_grid_layout.addWidget(self.future_frames_spinbox, 1, 6)

        self.main_grid_layout.addLayout(self.timeline_grid_layout, 1, 0)

        # Menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")
        self.view_menu = self.menu.addMenu("View")
        self.view_menu.addAction(self.options_dock.toggleViewAction())
        self.view_menu.addAction(self.info_dock.toggleViewAction())
        self.view_menu.addAction(self.camera_dock.toggleViewAction())

        ## Exit QAction
        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.setShortcut(QtGui.QKeySequence.Quit)
        exit_action.triggered.connect(self.close)

        # Open Sequence Action
        self.open_action = QtWidgets.QAction("Open Sequence", self)
        self.open_action.setShortcut(QtGui.QKeySequence.Open)
        self.open_action.triggered.connect(self.open_sequence)

        # Open Predictions Action
        self.open_pred_action = QtWidgets.QAction("Open Predictions", self)
        self.open_pred_action.triggered.connect(self.open_predictions)

        self.file_menu.addAction(self.open_action)
        self.file_menu.addAction(self.open_pred_action)
        self.file_menu.addAction(exit_action)

        # Status Bar
        self.status = self.statusBar()
        self.status_label = QtWidgets.QLabel()
        self.status_label.setText(
            "Frame {}/{}.\t\t Current Timestamp: {}.\t\t Time Window Size: {}s".format(0, 0, 0, 0.0))
        self.status.addPermanentWidget(self.status_label)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Right:
            if self.timeline_slider.value() < self.timeline_slider.maximum():
                self.timeline_slider.setValue(self.timeline_slider.value() + 1)
        elif event.key() == QtCore.Qt.Key_Left:
            if self.timeline_slider.value() > self.timeline_slider.minimum():
                self.timeline_slider.setValue(self.timeline_slider.value() - 1)

    def _reset_predictions(self):
        self.predictions = {}
        self.prediction_mapping = {}
        self.prediction_mapping_names = {}
        self.predictions_scheme = None
        self.plot_widget.prediction_mapping = self.prediction_mapping
        self.plot_widget.prediction_mapping_names = self.prediction_mapping_names
        self.plot_widget.prediction_colors = {}
        self.plot_widget.pred_track_id_to_color = {}
        self._toggle_prediction_color_options(enabled=False)

    def _toggle_prediction_color_options(self, enabled: bool):
        for idx in range(self.color_by_list.count()):
            item = self.color_by_list.model().item(idx)
            text = item.text()
            if text == ColorOpts.PREDLABEL.value or text == ColorOpts.PREDTRACK.value or \
                    text == ColorOpts.TRUEFALSE.value:
                item.setEnabled(enabled)

    def on_doppler_cb_clicked(self, state):
        """
        Callback function which is called when the checkbox for displaying the Doppler arrows is clicked.
        Shows and hides the extra slider for scaling the Doppler arrows.
        Re-plots the scene.
        :param state: state of the checkbox
        :return: None
        """
        if self.doppler_arrows_cb.isChecked():
            self.doppler_h_layout.setContentsMargins(20, 0, 0, 10)
            self.doppler_scale_slider.setVisible(True)
            self.doppler_scale_label.setVisible(True)
        else:

            self.doppler_h_layout.setContentsMargins(20, 0, 0, 0)
            self.doppler_scale_slider.setVisible(False)
            self.doppler_scale_label.setVisible(False)
        self.plot_frames()

    def on_slider_value_changed(self, value: int):
        """
        Callback function which is called when the slider is moved.
        The value of the timeline spinbox is updated and the scene is plotted.
        :param value: Current value of the slider.
        :return:
        """
        self.timeline_spinbox.blockSignals(True)  # this is needed to avoid an infinite loop: setValue of the spin
        # box emits a signal that points back to this function. So we have to block signals temporarily for the spinbox
        self.timeline_spinbox.setValue(value)
        self.timeline_spinbox.blockSignals(False)  # unblock signals again

        self.detection_info_label.setText("No detection selected.")

        self.plot_frames()

    def open_predictions(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Predictions',
                                                         os.getcwd(), "Prediction files (*.json)",
                                                         options=QtWidgets.QFileDialog.DontUseNativeDialog)
        filename = filename[0]
        if filename != "" and filename is not None:
            self.load_predictions(filename)

    def load_predictions(self, filename: str):
        """
        Load a predictions *.json file
        :param filename: path to the json file
        :return:
        """

        def str_key_to_int(dict):
            int_key_dict = {}
            for key, val in dict.items():
                int_key_dict[int(key)] = val
            return int_key_dict

        if not filename.endswith(".json") or not os.path.exists(filename):
            return

        with open(filename, "r") as f:
            data = json.load(f)

        fail = False
        if "schema" not in data or "predictions" not in data:
            fail = True
            msg_box = QtWidgets.QMessageBox()
            msg_box.setIcon(QtWidgets.QMessageBox.Critical)
            msg_box.setText("Unable to open prediction file. Invalid format.")
            msg_box.setWindowTitle("File Error")
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg_box.buttonClicked.connect(lambda: msg_box.close)
            msg_box.exec()

        if not fail:
            self.predictions_scheme = PredictionFileSchemas(data["schema"])
            self.predictions = data["predictions"]
            self.prediction_mapping = data.get("label_mapping", {i: i for i in range(len(Label))})
            self.prediction_mapping = str_key_to_int(self.prediction_mapping)
            self.prediction_mapping_names = data.get("new_label_names", {i: Label(i).name for i in range(len(Label))})
            self.prediction_mapping_names = str_key_to_int(self.prediction_mapping_names)
            self.plot_widget.prediction_mapping = self.prediction_mapping
            self.plot_widget.prediction_mapping_names = self.prediction_mapping_names
            self.plot_widget.create_prediction_colors()
            self._toggle_prediction_color_options(enabled=True)
            self.generate_colors_pred_tracks()
        self.plot_frames()

    def open_sequence(self):
        """
        Dialog for opening a measurement sequence.
        Only json files can be loaded. Actual loading is done by the load_sequence function.
        :return: None
        """
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Sequence',
                                                         os.getcwd(), "Radar data files (*.json)",
                                                         options=QtWidgets.QFileDialog.DontUseNativeDialog)
        filename = filename[0]
        if filename != "" and filename is not None:
            self.load_sequence(filename)

    def on_sequence_loading_finished(self, sequence, timestamps):
        QtGui.QApplication.restoreOverrideCursor()
        self.sequence = sequence
        self.timestamps = timestamps
        self.color_by_list.setCurrentIndex(6)
        self.timeline_slider.setMaximum(len(self.timestamps) - 1)
        self.timeline_spinbox.setMaximum(len(self.timestamps) - 1)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setValue(0)
        self.generate_colors_true_tracks()
        self.setWindowTitle("Radar Data Viewer - {}".format(self.sequence.sequence_name))
        self.plot_frames()

    def on_sequence_loading_failed(self):
        """
        Callback function for the case that loading of a sequence failed.
        Resets the cursor and prints and error box.
        :return:
        """
        QtGui.QApplication.restoreOverrideCursor()
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Critical)
        msg_box.setText("Unable to open file.")
        msg_box.setWindowTitle("File Error")
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.buttonClicked.connect(lambda: msg_box.close)
        msg_box.exec()

    def load_sequence(self, path: str):
        """
        Loads the contents of a json file which describes a measurement sequence.
        A timeline is created so that all scenes are in the correct order.
        The slider is initialized to the correct values and the first frame is plotted.
        :param path: full path to the json file.
        :return: None
        """
        if not path.endswith(".json") or not os.path.exists(path):
            return

        self._reset_predictions()

        self.loader_worker = LoadSequenceWorker(path)
        self.thread = QtCore.QThread()
        self.loader_worker.loading_done.connect(self.on_sequence_loading_finished)
        self.loader_worker.loading_failed.connect(self.on_sequence_loading_failed)
        self.loader_worker.moveToThread(self.thread)
        self.loader_worker.finished.connect(self.thread.quit)
        self.thread.started.connect(self.loader_worker.load)
        self.thread.start()
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

    def generate_colors_true_tracks(self):
        """
        Generates a dictionary for the colors of each object. This has to be done once after loading a sequence to
        ensure that the color of an object remains the same in the whole sequence.
        :return: None
        """
        track_ids = set(self.sequence.radar_data["track_id"])
        self.plot_widget.true_track_id_to_color = {}
        for tr_id in track_ids:
            if len(tr_id) == 0:
                continue
            self.plot_widget.true_track_id_to_color[tr_id] = np.random.choice(Colors.object_colors)

    def generate_colors_pred_tracks(self):
        if self.predictions == {} or self.predictions_scheme == PredictionFileSchemas.SemSeg:
            return
        track_ids = {x[1] for x in self.predictions.values()}  # set comprehension, x[0]=class label, x[1]=instance
        self.plot_widget.pred_track_id_to_color = {}
        for tr_id in track_ids:
            if tr_id == -1:
                continue
            self.plot_widget.pred_track_id_to_color[tr_id] = np.random.choice(Colors.object_colors)

    def get_current_scenes(self):
        """
        Retrieves the scenes which should be displayed according to the current values of the time slider and the
        spinboxes for the past and future frames.
        Values of the spinboxes are retrieved and from the list of timestamps, the corresponding times are obtained.
        :return: The current frame (type Scene) and a list of other frames (type Scene) which should be displayed.
        """
        cur_idx = self.timeline_slider.value()
        cur_timestamp = self.timestamps[cur_idx]
        n_prev_frames = self.prev_frames_spinbox.value()
        n_next_frames = self.future_frames_spinbox.value()
        current_scene = self.sequence.get_scene(cur_timestamp)
        other_scenes = []
        for i in range(1, n_prev_frames + 1):
            if cur_idx - i < 0:
                break
            t = self.timestamps[cur_idx - i]
            other_scenes.append(self.sequence.get_scene(t))
        for i in range(1, n_next_frames + 1):
            if cur_idx + i >= len(self.timestamps):
                break
            t = self.timestamps[cur_idx + i]
            other_scenes.append(self.sequence.get_scene(t))
        return current_scene, other_scenes

    def update_status_bar(self, frame_idx, frame_timestamp, window_size):
        """
        Updates the text of the status bar
        :param frame_idx: current frame index
        :param frame_timestamp: Current timestamp of the frame
        :param window_size: Current size of the temporal time window. Depending on how many frames are displayed and
        how close these radar scans are in time, the value changes.
        :return: None
        """
        if len(self.timestamps) == 0:
            current_time = 0
        else:
            current_time = (frame_timestamp - self.timestamps[0]) / 10 ** 6

        if self.predictions != {}:
            pred_str = "Loaded {} predictions.     ".format(len(self.predictions))
        else:
            pred_str = "No predictions loaded.     "
        self.status_label.setText(pred_str +
                                  "Frame {}/{}     Current Timestamp: {}     Time Window Size: {:.1f}ms     Time: {:.2f}s".format(
                                      frame_idx, len(self.timestamps) - 1, frame_timestamp, window_size, current_time))

    def display_camera_image(self, cam_filename: str):
        """
        Given a filename, the image in the camera_widget is updated.
        :param cam_filename: Path to the image file.
        :return: None
        """
        if not os.path.exists(cam_filename):
            raise FileNotFoundError("Cannot find image {}".format(cam_filename))
        if cam_filename != self.current_cam_filename:
            # update the image only, when a new file would be loaded.
            image = plt.imread(cam_filename)
            self.camera_widget.setImage(image)
            self.current_cam_filename = cam_filename

    def trafo_radar_data_world_to_car(self, scene, other_scenes) -> np.ndarray:
        """
        Transforms the radar data listed in other_scenes into the same car coordinate system that is used in 'scene'.
        :param scene: Scene. Containing radar data and odometry information of one scene. The odometry information from
        this scene is used to transform the detections from the other timestamps into this scene.
        :param other_scenes: List of Scene items. All detections in these other scenes are transformed
        :return: A numpy array with all radar data from all scenes. The fields "x_cc" and "y_cc" are now relative to the
        current scene.
        """
        if len(other_scenes) == 0:
            return scene.radar_data
        other_radar_data = np.hstack([x.radar_data for x in other_scenes])
        x_cc, y_cc = transform_detections_sequence_to_car(other_radar_data["x_seq"], other_radar_data["y_seq"],
                                                          scene.odometry_data)
        other_radar_data["x_cc"] = x_cc
        other_radar_data["y_cc"] = y_cc
        return np.hstack([scene.radar_data, other_radar_data])

    def _append_predictions(self, radar_data):
        if self.predictions != {}:
            if self.predictions_scheme == PredictionFileSchemas.SemSeg:
                pred_label_ids = [self.predictions.get(row["uuid"].decode(), -1) for row in radar_data]
                pred_instance_ids = [-1] * len(radar_data)
            elif self.predictions_scheme == PredictionFileSchemas.InstSeg:
                pred_label_ids = [self.predictions.get(row["uuid"].decode(), [-1])[0] for row in radar_data]
                pred_instance_ids = [self.predictions.get(row["uuid"].decode(), [-1, -1])[1] for row in radar_data]
            else:
                raise NotImplementedError("Unknown scheme", self.predictions_scheme)

            radar_data = rfn.append_fields(radar_data, names=["pred_label_id", "pred_instance_id"],
                                           data=[pred_label_ids, pred_instance_ids], usemask=False)
        return radar_data

    def plot_frames(self):
        """
        Plot the current frames.
        This includes:
            - Detections as scatter points
            - Camera image
            - Text labels of all objects containing the class names
            - Doppler velocity arrows
            - Convex Hulls around ground truth objects
        :return: None
        """

        cur_idx = self.timeline_slider.value()
        if len(self.timestamps) == 0 or cur_idx >= len(self.timestamps):
            return
        cur_timestamp = self.timestamps[cur_idx]
        current_scene, other_scenes = self.get_current_scenes()
        radar_data = self.trafo_radar_data_world_to_car(current_scene, other_scenes)

        self.update_status_bar(cur_idx, cur_timestamp,
                               (np.max(radar_data["timestamp"] - np.min(radar_data["timestamp"])) / 10 ** 3))

        radar_data = self._append_predictions(radar_data)

        # CAMERA
        self.display_camera_image(current_scene.camera_image_name)

        # DETECTIONS
        self.plot_widget.plot_detections(radar_data, color_by=self.color_by_list.currentText())

        # NAMES OF TRUE LABELS
        if self.label_text_cb.isChecked():
            self.plot_widget.plot_label_names(radar_data)
        else:
            self.plot_widget.clear_text_items()

        # DOPPLER ARROWS
        if self.doppler_arrows_cb.isChecked():
            scale = self.doppler_scale_slider.value() / 10
            self.plot_widget.plot_doppler_arrows(radar_data, scale=scale)
        else:
            self.plot_widget.clear_doppler_arrows()

        # CONVEX HULLS
        if self.convex_hulls_cb.isChecked():
            self.plot_widget.plot_convex_hulls(radar_data)
        else:
            self.plot_widget.clear_hulls()


class CarImage(QtWidgets.QGraphicsPixmapItem):

    def __init__(self, x, y, phi_deg, *args, **kwargs):
        filepath = resource_filename('radar_scenes.res', 'car.png')
        filepath = os.path.abspath(filepath)
        if not os.path.exists(filepath):
            return
        pxmp = QtGui.QPixmap(filepath)
        super().__init__(pxmp, *args, **kwargs)
        self.setScale(0.0035)
        self.height = pxmp.height()
        self.width = pxmp.width()
        self.setOffset(-pxmp.width() / 2., -pxmp.height() * 3. / 4)
        self.setOpacity(0.9)
        self.setPos(x, y)
        self.setRotation((phi_deg + 90) % 360)


class VelocityArrow(QtWidgets.QGraphicsLineItem):
    def __init__(self, start_point, end_point, brush=None):
        super().__init__(start_point[0], start_point[1], end_point[0], end_point[1])
        self.start_point = start_point
        self.end_point = end_point
        if brush is None:
            brush = QtCore.Qt.gray
        pen = QtGui.QPen()
        pen.setWidthF(1.2)
        pen.setBrush(brush)
        pen.setCosmetic(True)
        self.setPen(pen)

    def update_position(self, start_point, end_point):
        self.setLine(start_point[0], start_point[1], end_point[0], end_point[1])


class PlotWidget:

    def __init__(self, plot_item: pg.PlotItem):
        self.plot_item = plot_item  # type: pg.PlotItem
        self.ego_vehicle = CarImage(0, 0, 90)
        self._doppler_arrows = []  # keep track of plotted doppler arrows
        self._text_items = []
        self._hulls = []
        self.detections_plot_items = None
        self.true_track_id_to_color = {}
        self.pred_track_id_to_color = {}
        self._init_plot_window()
        self.label_id_to_display_name = {
            0: "Car",
            1: "Large Veh.",
            2: "Truck",
            3: "Bus",
            4: "Train",
            5: "Bicycle",
            6: "Motorbike",
            7: "Pedestrian",
            8: "Ped. Group",
            9: "Animal",
            10: "Other",
            11: "Static"
        }
        self.info_label = None
        self.prediction_mapping = {}
        self.prediction_mapping_names = {}
        self.prediction_colors = {}

    def _init_plot_window(self):
        """
        Initialize the basic plot window:
        -add ego vehicle to plot
        -activate grid
        -set aspect ratio
        -invert x axis (this is needed since y_cc is positive to the left)
        :return:
        """
        self.plot_item.showGrid(True, True)
        self.plot_item.invertX(True)
        self.plot_item.addItem(self.ego_vehicle)
        self.plot_item.vb.setXRange(-40, 40, padding=0)
        self.plot_item.vb.setYRange(-15, 100, padding=0)
        self.plot_item.vb.setAspectLocked(True)
        self.plot_item.setLabel("left", "x<sub>cc</sub> (m)")
        self.plot_item.setLabel("bottom", "y<sub>cc</sub> (m) <br>")

    def add_doppler_arrow(self, doppler_arrow: VelocityArrow):
        doppler_arrow.setZValue(-1)
        self._doppler_arrows.append(doppler_arrow)
        self.plot_item.addItem(doppler_arrow)

    def add_text_item(self, text_item: pg.TextItem):
        self._text_items.append(text_item)
        self.plot_item.addItem(text_item)

    def add_hull(self, hull: pg.PlotCurveItem):
        self._hulls.append(hull)
        self.plot_item.addItem(hull)

    def clear_doppler_arrows(self, start=0):
        for arrow in self._doppler_arrows[start:]:
            self.plot_item.removeItem(arrow)
        self._doppler_arrows = self._doppler_arrows[:start]  # if start is zero, then my_list[:0] returns an empty list

    def clear_text_items(self, start=0):
        for text_item in self._text_items[start:]:
            self.plot_item.removeItem(text_item)
        self._text_items = self._text_items[:start]

    def clear_hulls(self, start=0):
        for hull in self._hulls[start:]:
            self.plot_item.removeItem(hull)
        self._hulls = self._hulls[:start]

    def create_prediction_colors(self):

        for label_id, pred_label_id in self.prediction_mapping.items():
            # label_id is the usual label id and pred_label_id is one of the new ids
            new_name = self.prediction_mapping_names.get(pred_label_id, None)
            label_name = Label(label_id).name
            if new_name == label_name:
                # if the new name coincides with an old name, the same color is used as for the original label
                self.prediction_colors[pred_label_id] = Colors.label_id_to_color[label_id]

        # now all predicted labels have a color which is the same as the color for the "un"mapped label
        # e.g. if a new label is called STATIC, the same color is used as for the original STATIC label

        all_pred_labels = set(self.prediction_mapping.values())  # set of all predicted labels
        for l in all_pred_labels:
            if l in self.prediction_colors:
                continue
            if l is None:
                self.prediction_colors[None] = Colors.gray
                continue
            # select a new color for this label
            selected_color = Colors.red
            for color in Colors.sensor_id_to_color.values():
                if color in set(self.prediction_colors.values()):
                    # this color was already used by a different predicted label -> dont use it again
                    continue
                else:
                    selected_color = color
                    break
            self.prediction_colors[l] = selected_color

    def on_plot_clicked(self, ev) -> None:
        """
        Callback function for a mouse click on the plot.
        Resets the pen of the scatter plot. "Deselects" all detections.
        :param ev: The QEvent
        :return: None
        """
        if not ev.isAccepted():
            # event is accepted when a point is clicked. Then this method has no effect.
            scatter_data = self.detections_plot_items.scatter.data
            empty_pen = QtGui.QPen()
            empty_pen.setStyle(QtCore.Qt.NoPen)
            scatter_data["pen"] = np.array([empty_pen] * len(scatter_data), dtype=object)
            self.detections_plot_items.opts["symbolPen"] = scatter_data["pen"]
            self.detections_plot_items.updateItems()
            self.info_label.setText("No detection selected.")

    def on_point_clicked(self, plot_item: pg.PlotDataItem, points: list) -> None:
        """
        Callback function for the click event on a detection. The detection is highlighted with a white circle around it
        and the detection's properties are written on the info label.
        :param plot_item: the PlotDataItem which holds the points
        :param points: List of points which where clicked on.
        :return: None
        """
        scatter_data = plot_item.scatter.data

        if len(points) > 0:
            empty_pen = QtGui.QPen()
            empty_pen.setStyle(QtCore.Qt.NoPen)
            scatter_data["pen"] = np.array([empty_pen] * len(scatter_data), dtype=object)
            uuids = np.array([x[11] for x in scatter_data["data"]])
            detection = points[0].data()
            det_uuid = detection["uuid"]
            idx = np.where(uuids == det_uuid)[0]
            if len(idx) > 0:
                idx = idx[0]
                scatter_data["pen"][idx] = pg.mkPen(color="w", width=3)
                plot_item.opts["symbolPen"] = scatter_data["pen"]
                plot_item.updateItems()
                tr_id = detection["track_id"].decode()
                if len(tr_id) == 0:
                    tr_id = "<None>"
                info_text = (
                    "UUID = {}\nSensor ID = {}\nTimestamp = {}\nRange = {:.3f} m\nAzimuth = {:.3f}°\nRCS = {:.3f} dBsm\n"
                    "x (cc) = {:.3f} m\ny (cc) = {:.3f} m\nRadial Velocity = {:.3f} m/s\n"
                    "Compensated Rad. Vel. = {:.3f} m/s\n"
                    "Label ID = {} ({})\nTrack ID = {}").format(
                    detection["uuid"].decode(), detection["sensor_id"], detection["timestamp"], detection["range_sc"],
                    np.rad2deg(detection["azimuth_sc"]), detection["rcs"], detection["x_cc"], detection["y_cc"],
                    detection["vr"], detection["vr_compensated"], detection["label_id"],
                    self.label_id_to_display_name[detection["label_id"]], tr_id
                )
                if "pred_label_id" in detection.dtype.names:
                    pred_label = detection["pred_label_id"]
                    pred_instance = detection["pred_instance_id"]
                    info_text += "\nPredicted Label = {} ({})".format(pred_label,
                                                                      self.prediction_mapping_names.get(pred_label,
                                                                                                        "Unknown"))
                    mapped_true_label = self.prediction_mapping.get(detection["label_id"], "Unknown")
                    info_text += "\nMapped True Label = {} ({})".format(mapped_true_label,
                                                                        self.prediction_mapping_names.get(
                                                                            mapped_true_label, "Unknown"))
                    if pred_instance is None:
                        info_text += "\nPredicted Instance = <None>"
                    else:
                        info_text += "\nPredicted Instance = {}".format(pred_instance)

                self.info_label.setText(info_text)

    def plot_doppler_arrows(self, radar_data: np.ndarray, scale=0.7, min_velo_threshold=0.02) -> None:
        """
        Plot a Doppler velocity arrow for each detection which has a Doppler velocity greater than min_velo_threshold.
        :param radar_data: numpy array containing the detections. Shape (n_detections,)
        Each row is a named numpy array.
        :param scale: length of the arrows.
        :param min_velo_threshold: Threshold for the arrows. If Doppler value is below this threshold, no arrow is drawn
        for performance optimization.
        :return: None
        """
        sensor_yaw = np.array([get_mounting(s_id)["yaw"] for s_id in radar_data["sensor_id"]])
        angles = radar_data["azimuth_sc"] + sensor_yaw
        vx = radar_data["vr_compensated"] * np.cos(angles)
        vy = radar_data["vr_compensated"] * np.sin(angles)

        n_arrows_drawn = 0
        for idx, detection in enumerate(radar_data):
            if np.abs(detection["vr_compensated"]) < min_velo_threshold:
                # runtime optimization: no arrows for static points
                continue
            start_point = np.array([detection["y_cc"], detection["x_cc"]])
            end_point = start_point + scale * np.array([vy[idx], vx[idx]])
            if n_arrows_drawn < len(self._doppler_arrows):
                # reuse an existing arrow, just change its position
                doppler_arrow = self._doppler_arrows[n_arrows_drawn]  # type: VelocityArrow
                doppler_arrow.update_position(start_point, end_point)
                self._doppler_arrows[n_arrows_drawn] = doppler_arrow
            else:
                # create a new arrow
                self.add_doppler_arrow(VelocityArrow(start_point, end_point))
            n_arrows_drawn += 1
        if n_arrows_drawn < len(self._doppler_arrows):
            # remove all arrows that are not needed anymore
            self.clear_doppler_arrows(n_arrows_drawn)

    def plot_label_names(self, radar_data: np.ndarray) -> None:
        """
        Create text labels for each ground truth object. The label is placed at the center of the object and contains
        the name of the semantic class the object belongs to.
        :param radar_data:  numpy array containing the detections. Shape (n_detections,)
        Each row is a named numpy array.
        :return: None
        """
        track_ids = set(radar_data["track_id"])
        n_texts_drawn = 0
        for tr_id in track_ids:
            if len(tr_id) == 0:
                continue
            idx = np.where(radar_data["track_id"] == tr_id)[0]
            x_mean = np.mean(radar_data[idx]["x_cc"])
            y_mean = np.mean(radar_data[idx]["y_cc"])
            label_id = radar_data["label_id"][idx][0]

            if n_texts_drawn < len(self._text_items):
                text_item = self._text_items[n_texts_drawn]
                text_item.setText(self.label_id_to_display_name.get(label_id, ""))
                text_item.setPos(y_mean, x_mean)
                self._text_items[n_texts_drawn] = text_item
            else:
                text_item = pg.TextItem(self.label_id_to_display_name.get(label_id, ""), color=(230, 230, 230),
                                        anchor=(0.5, 0.5), border=(200, 200, 200, 50), fill=(44, 50, 150, 150))
                text_item.setPos(y_mean, x_mean)
                self.add_text_item(text_item)
            n_texts_drawn += 1
        if n_texts_drawn < len(self._text_items):
            self.clear_text_items(n_texts_drawn)

    def plot_detections(self, radar_data: np.ndarray, color_by: str) -> None:
        """
        Create a scatter plot for each detection in radar_data.
        Depending on the color_by option, a different color map is chosen for the detections.
        In case a scatter plot item already exists, it is reused for plotting.
        :param radar_data: numpy array containing the detections. Shape (n_detections,)
        Each row is a named numpy array.
        :param color_by: This string indicates how the individual points shall be colored. Valid options are within the
        ColorOpts enum.
        :return: None
        """
        if color_by == ColorOpts.SENSORID.value:
            symbol_brush = [brush_for_color(Colors.sensor_id_to_color[x["sensor_id"]]) for x in radar_data]
        elif color_by == ColorOpts.RCS.value:
            cmap = plt.get_cmap("Greens")
            rcs_vals = radar_data["rcs"]
            rcs_vals = np.clip(rcs_vals, -20, 20)
            rcs_vals = 1 / 40.0 * (rcs_vals + 20)
            colors = np.apply_along_axis(lambda x: cmap(x, bytes=True), 0, rcs_vals)
            qtcolors = np.apply_along_axis(lambda x: QtGui.QColor(x[0], x[1], x[2], x[3]), 1, colors)
            symbol_brush = [QtGui.QBrush(x) for x in qtcolors]
        elif color_by == ColorOpts.DOPPLER.value:
            cmap = plt.get_cmap("PiYG")
            doppler_vals = radar_data["vr_compensated"]
            doppler_vals = np.clip(doppler_vals, -10, 10)
            doppler_vals = 1 / 20.0 * (doppler_vals + 10)
            colors = np.apply_along_axis(lambda x: cmap(x, bytes=True), 0, doppler_vals)
            qtcolors = np.apply_along_axis(lambda x: QtGui.QColor(x[0], x[1], x[2], x[3]), 1, colors)
            symbol_brush = [QtGui.QBrush(x) for x in qtcolors]
        elif color_by == ColorOpts.TRUELABEL.value:
            symbol_brush = [brush_for_color(
                Colors.label_id_to_color.get(x["label_id"], Colors.gray)) for x in radar_data]
        elif color_by == ColorOpts.PREDLABEL.value:
            symbol_brush = [brush_for_color(self.prediction_colors.get(x, Colors.gray)) for x in
                            radar_data["pred_label_id"]]
        elif color_by == ColorOpts.TRUETRACK.value:
            symbol_brush = [brush_for_color(self.true_track_id_to_color.get(x["track_id"], Colors.gray)) for x
                            in radar_data]
        elif color_by == ColorOpts.PREDTRACK.value:
            symbol_brush = [brush_for_color(self.pred_track_id_to_color.get(x["pred_instance_id"], Colors.gray)) for x
                            in radar_data]
        elif color_by == ColorOpts.TRUEFALSE.value:
            symbol_brush = []
            for detection in radar_data:
                pred_label = detection["pred_label_id"]
                if pred_label == -1:
                    symbol_brush.append(brush_for_color(Colors.gray))
                    continue
                true_label = detection["label_id"]
                mapped_true_label = self.prediction_mapping.get(true_label, -1)
                if mapped_true_label == -1:
                    symbol_brush.append(brush_for_color(Colors.gray))
                    continue
                if mapped_true_label == pred_label:
                    symbol_brush.append(brush_for_color(Colors.green))
                else:
                    symbol_brush.append(brush_for_color(Colors.red))
        else:
            symbol_brush = "#888888"

        if self.detections_plot_items is None:
            self.detections_plot_items = self.plot_item.plot(radar_data["y_cc"], radar_data["x_cc"], pen=None,
                                                             symbol='h', symbolBrush=symbol_brush, symbolPen=None,
                                                             symbolSize=7, data=radar_data)
            self.detections_plot_items.sigPointsClicked.connect(self.on_point_clicked)
            self.plot_item.scene().sigMouseClicked.connect(self.on_plot_clicked)
        else:
            self.detections_plot_items.setData(radar_data["y_cc"], radar_data["x_cc"], pen=None, symbol='h',
                                               symbolBrush=symbol_brush, symbolPen=None, symbolSize=7, data=radar_data)

    def plot_convex_hulls(self, radar_data: np.ndarray) -> None:
        """
        Computes convex hulls around objects which have the same "track_id".
        :param radar_data: numpy array containing the radar data for which the hulls shall be plotted.
        :return: None
        """
        track_ids = set(radar_data["track_id"])
        n_hulls_drawn = 0
        for tr_id in track_ids:
            if len(tr_id) == 0:
                continue
            idx = np.where(radar_data["track_id"] == tr_id)[0]
            if len(idx) < 2:
                continue
            points = np.zeros((len(idx), 2))
            points[:, 0] = radar_data[idx]["x_cc"]
            points[:, 1] = radar_data[idx]["y_cc"]
            if len(idx) > 2:
                try:
                    hull = ConvexHull(points)
                    vertices = points[hull.vertices]
                    x = np.append(vertices[:, 0], vertices[0, 0])
                    y = np.append(vertices[:, 1], vertices[0, 1])
                except (QhullError, ValueError):
                    continue
            else:
                # only two points
                x = np.array([points[0, 0], points[1, 0]])
                y = np.array([points[0, 1], points[1, 1]])

            if n_hulls_drawn < len(self._hulls):
                hull_visu = self._hulls[n_hulls_drawn]
                hull_visu.setData(y, x)
                hull_visu.setZValue(-1)
                self._hulls[n_hulls_drawn] = hull_visu
            else:
                pen = pg.mkPen("w")
                pen.setWidth(2)
                hull_visu = pg.PlotCurveItem(pen=pen)
                hull_visu.setData(y, x)
                hull_visu.setZValue(-1)
                self.add_hull(hull_visu)
            n_hulls_drawn += 1
        if n_hulls_drawn < len(self._hulls):
            self.clear_hulls(n_hulls_drawn)


def main():
    # pass arguments from command line
    parser = argparse.ArgumentParser(description='Radar Data Viewer.\nCopyright 2021 Ole Schumann')
    parser.add_argument("filename", nargs="?", default="", type=str,
                        help="Path to a *.json or *.h5 file of the radar data set.")

    args = parser.parse_args()
    app = QtWidgets.QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside2'))
    app.setWindowIcon(QtGui.QIcon("res/icon.png"))
    
    window = MainWindow()
    window.show()

    window.load_sequence(args.filename)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
