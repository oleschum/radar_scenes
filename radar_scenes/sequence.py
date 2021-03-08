import os
import h5py
import json
from typing import Union, List
from radar_scenes.scene import Scene


class Sequence:
    """
    Wrapper class for a measurement sequence.
    Usually, this class should be constructed from a *.json file from the RadarScenes dataset.
    """

    def __init__(self):
        self.sequence_name = None
        self.radar_data = None
        self.odometry_data = None
        self.first_timestamp = None
        self.last_timestamp = None
        self._scenes = {}
        self._data_folder = ""
        self._current_timestamp = None

    def __len__(self):
        """
        Returns the length of a sequence in terms of number of scenes.
        :return: Number of scenes within this sequence
        """
        return len(self._scenes)

    @classmethod
    def from_json(cls, filename: str) -> "Sequence":
        """
        Create a Sequence object from a *.json file.
        Usually, this should be a scenes.json file from one sequence of the RadarScenes dataset.
        :param filename: full path to a *.json file
        :return: Sequence object
        """
        with open(filename, "r") as f:
            json_data = json.load(f)
        sequence = cls()
        sequence.sequence_name = json_data["sequence_name"]
        sequence._scenes = json_data["scenes"]
        sequence.first_timestamp = json_data["first_timestamp"]
        sequence.last_timestamp = json_data["last_timestamp"]
        sequence._data_folder = os.path.dirname(filename)
        h5_filename = os.path.join(sequence._data_folder, "radar_data.h5")
        sequence.load_sequence_data(h5_filename)
        return sequence

    @property
    def timestamps(self) -> List[int]:
        return list(map(int, self._scenes.keys()))

    def scenes(self, sensor_id=None):
        same_sensor = sensor_id is not None
        if same_sensor and sensor_id not in {1, 2, 3, 4}:
            raise ValueError("Unknown sensor id. Valid values are 1, 2, 3, 4.")
        self._current_timestamp = self.first_timestamp
        if same_sensor:
            t_init = self.first_timestamp
            while True:
                current_scene = self.get_scene(t_init)
                if current_scene.sensor_id == sensor_id:
                    self._current_timestamp = t_init
                    break
                t_init = self.next_timestamp_after(t_init)
        while self._current_timestamp is not None:
            scene = self.get_scene(self._current_timestamp)
            self._current_timestamp = self.next_timestamp_after(self._current_timestamp, same_sensor=same_sensor)
            yield scene

    def load_sequence_data(self, filename: str) -> None:
        """
        Load contents of a *.h5 sequence file.
        Data is stored in numpy arrays.
        :param filename: Full path to a *.h5 file.
        :return: None
        """
        with h5py.File(filename, "r") as f:
            self.radar_data = f["radar_data"][:]
            self.odometry_data = f["odometry"][:]

    def next_timestamp_after(self, timestamp: Union[str, int], same_sensor=False) -> Union[int, None]:
        """
        Looks for the subsequent timestamp after a given one.
        :param timestamp: timestamp (int or str) for which the next timestamp is sought
        :param same_sensor: If True, the timestamp of the next measurement from the same sensor is returned
        :return: None, if the provided timestamp is the last timestamp in the sequence or if the timestamp does not
        exist at all. Otherwise, the next timestamp of a radar measurement is returned as an int.
        """
        timestamp = str(timestamp)
        if timestamp not in self._scenes:
            return None
        if same_sensor:
            return self._scenes[timestamp]["next_timestamp_same_sensor"]
        else:
            return self._scenes[timestamp]["next_timestamp"]

    def next_scene_after(self, timestamp: Union[str, int], same_sensor=False) -> Union[Scene, None]:
        """
        Creates the next scene following a given timestamp.
        :param timestamp: current timestamp
        :param same_sensor: If true, only the same sensor as the current one is considered.
        :return: None, if timestamp is the last timestamp in the sequence. Otherwise, a Scene object holding information
        about the next scene.
        """
        next_timestamp = self.next_timestamp_after(timestamp, same_sensor)
        return self.get_scene(next_timestamp)

    def prev_timestamp_before(self, timestamp: Union[str, int], same_sensor=False) -> Union[int, None]:
        """
        Looks for the preceding timestamp before a given one.
        :param timestamp: timestamp (int or str) for which the previous timestamp is sought
        :param same_sensor: If true, only measurements from the same sensor as the current one are considered.
        :return: None, if the provided timestamp is the first timestamp in the sequence or if the timestamp does not
        exist at all. Otherwise, the previous timestamp of a radar measurement is returned as an int.
        """
        timestamp = str(timestamp)
        if timestamp not in self._scenes:
            return None
        if same_sensor:
            return self._scenes[timestamp]["prev_timestamp_same_sensor"]
        else:
            return self._scenes[timestamp]["prev_timestamp"]

    def prev_scene_before(self, timestamp: Union[str, int], same_sensor=False) -> Union[Scene, None]:
        """
        Creates the previous scene prior to a given timestamp.
        :param timestamp: current timestamp
        :param same_sensor: If true, only measurements from the same sensor as the current one are considered.
        :return: None, if timestamp is the first timestamp in the sequence. Otherwise, a Scene object holding information
        about the previous scene.
        """
        prev_timestamp = self.prev_timestamp_before(timestamp, same_sensor)
        return self.get_scene(prev_timestamp)

    def get_scene(self, timestamp: Union[str, int]) -> Union[Scene, None]:
        """
        Constructs a Scene object for measurements of a given timestamp.
        The scene holds radar data, odometry data as well as the name of the camera image belonging to this scene.
        If the timestamp is invalid, None is returned.
        :param timestamp: The timestamp for which a scene is desired.
        :return: The Scene object or None, if the timestamp is invalid.
        """
        if timestamp is None or self.radar_data is None or self.odometry_data is None:
            return None
        timestamp = str(timestamp)
        if timestamp not in self._scenes:
            return None

        scene_dict = self._scenes[timestamp]
        scene = Scene()
        scene.timestamp = int(timestamp)
        scene.radar_data = self.radar_data[scene_dict["radar_indices"][0]: scene_dict["radar_indices"][1]]
        scene.odometry_data = self.odometry_data[scene_dict["odometry_index"]]
        scene.odometry_timestamp = scene_dict["odometry_timestamp"]
        scene.sensor_id = scene_dict["sensor_id"]
        scene.camera_image_name = os.path.join(self._data_folder, "camera", scene_dict["image_name"])

        return scene


def read_sequences_json(sequences_filename: str) -> dict:
    """
    Simple helper method to load the contents of the sequences.json file
    :param sequences_filename: full path to the sequences.json file
    :return: The contents of the as a python dictionary
    """
    with open(sequences_filename, "r") as f:
        data = json.load(f)
    return data


def get_training_sequences(sequences_filename: str) -> List[str]:
    """
    Given the path to a sequences.json file, all sequences from the category "training" are retrieved.
    :param sequences_filename: path to the sequences.json file from the RadarScenes dataset.
    :return: list of training sequence names
    """
    training_sequ = []
    data = read_sequences_json(sequences_filename)
    for sequence_name, sequence_data in data["sequences"].items():
        if sequence_data["category"] == "train":
            training_sequ.append(sequence_name)
    return training_sequ


def get_validation_sequences(sequences_filename: str) -> List[str]:
    """
    Given the path to a sequences.json file, all sequences from the category "validation" are retrieved.
    :param sequences_filename: path to the sequences.json file from the RadarScenes dataset.
    :return: list of validation sequence names
    """
    validation_sequ = []
    data = read_sequences_json(sequences_filename)
    for sequence_name, sequence_data in data["sequences"].items():
        if sequence_data["category"] == "validation":
            validation_sequ.append(sequence_name)
    return validation_sequ
