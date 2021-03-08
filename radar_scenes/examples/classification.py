import os
import numpy as np
from random import choices
from radar_scenes.sequence import get_training_sequences, get_validation_sequences, Sequence
from radar_scenes.labels import ClassificationLabel
from radar_scenes.evaluation import per_point_predictions_to_json, PredictionFileSchemas


class SemSegNetwork:
    """
    This is a dummy class for a semantic segmentation neural network.
    For training, it takes as input a point cloud X and per-point labels y.
    The network then learns to predict a class label for each input point p in X.
    However, an instance label (track id) is NOT predicted.
    """

    def __init__(self):
        self._y_true_test = None

    def train(self, X, y):
        """
        Dummy method for training the neural network.
        :param X: training data. Shape (N_points, N_feat)
        :param y: semantic class label for each point. Shape (N_batch, N_points)
        :return: None
        """
        pass

    def predict(self, X):
        """
        Predicts a class label for each point in X.
        This is a mock method which simply uses the true class labels from self._y_true_test to generate
        a prediction which is likely correct
        :param X: validation data. Shape (N_points, N_feat)
        :return: an array of shape (N_points, ) containing the predicted class labels
        """
        y_pred = []
        for y in self._y_true_test:
            if ClassificationLabel(y) == ClassificationLabel.CAR:
                proba_vector = [0.9, 0.01, 0.01, 0.03, 0.04, 0.01]
            elif ClassificationLabel(y) == ClassificationLabel.PEDESTRIAN:
                proba_vector = [0.01, 0.90, 0.06, 0.02, 0.00, 0.01]
            elif ClassificationLabel(y) == ClassificationLabel.PEDESTRIAN_GROUP:
                proba_vector = [0.015, 0.04, 0.88, 0.045, 0.01, 0.01]
            elif ClassificationLabel(y) == ClassificationLabel.TWO_WHEELER:
                proba_vector = [0.06, 0.04, 0.03, 0.84, 0.01, 0.02]
            elif ClassificationLabel(y) == ClassificationLabel.LARGE_VEHICLE:
                proba_vector = [0.05, 0.01, 0.02, 0.03, 0.88, 0.01]
            else:
                proba_vector = [0.02, 0.005, 0.005, 0.005, 0.005, 0.96]
            yy = choices([0, 1, 2, 3, 4, 5], weights=proba_vector)[0]
            y_pred.append(yy)

        return y_pred


class InstSegNetwork(SemSegNetwork):
    def __init__(self):
        super().__init__()
        self._y_inst_true = None
        self.last_instance_id = 1
        self.translation_dict = {}

    def train(self, X, y, y_inst):
        pass

    def predict(self, X: np.ndarray):
        """
        Prediction method of the instance segmentation mock.
        A class label and an instance label is predicted for each detection in X.
        :param X: Array holding the individual detections
        :return: predicted class labels and predicted instance labels.
        """
        y_pred = super().predict(X)

        # the viewer treats instance ID = -1 as "no instance". Therefore, this is used as default value for the instance
        # labels.
        y_inst_pred = np.zeros(len(X), dtype=np.int32) - 1

        # translate string uuids to integers
        for tr_uuid in set(self._y_inst_true):
            if tr_uuid not in self.translation_dict:
                self.translation_dict[tr_uuid] = self.last_instance_id
                self.last_instance_id += 1

        # iterate over true instance labels and assign new labels as prediction
        for idx, true_instance_id in enumerate(self._y_inst_true):
            if (true_instance_id == b"" or true_instance_id == "") and y_pred[idx] != ClassificationLabel.STATIC.value:
                # a detection without a true instance id but with a predicted class label of a dynamic object gets a
                # new instance label
                y_inst_pred[idx] = self.last_instance_id
                self.last_instance_id += 1
            else:
                if np.random.random() < 0.1:
                    # with a 10% chance, a point gets a different track id than all other points of this object
                    y_inst_pred[idx] = self.last_instance_id
                    self.last_instance_id += 1
                else:
                    # assign the same integer instance label to all other points of an instance
                    y_inst_pred[idx] = self.translation_dict[true_instance_id]

        # set default instance id for all points with label "STATIC"
        idx = np.where(np.array(y_pred) == ClassificationLabel.STATIC.value)[0]
        y_inst_pred[idx] = -1

        return y_pred, y_inst_pred


def features_from_radar_data(radar_data):
    """
    Generate a feature vector for each detection in radar_data.
    The spatial coordinates as well as the ego-motion compensated Doppler velocity and the RCS value are used.
    :param radar_data: Input data
    :return: numpy array with shape (len(radar_data), 4), contains the feature vector for each point
    """
    X = np.zeros((len(radar_data), 4))  # construct feature vector
    X[:, 0] = radar_data["x_cc"]
    X[:, 1] = radar_data["y_cc"]
    X[:, 2] = radar_data["vr_compensated"]
    X[:, 3] = radar_data["rcs"]
    return X


def train_data_generator(training_sequences: list, path_to_dataset: str, return_track_ids=False):
    """
    Given a list of training sequence names and the path to the data set,
    the sequences are loaded and from each sequence 5 scenes are randomly chosen and returned as training data
    This is only a mock training data generator. A true generator would require some more work.
    :param training_sequences: list of sequence names
    :param path_to_dataset: path to the dataset on the hard drive
    :param return_track_ids: If true, in addition to the feature vectors and class labels, also the track ids are
    returned.
    :return: feature vectors and true labels.
    """
    for sequence_name in training_sequences:
        try:
            sequence = Sequence.from_json(os.path.join(path_to_dataset, "data", sequence_name, "scenes.json"))
        except FileNotFoundError:
            continue
        timestamps = sequence.timestamps  # obtain all time stamps available in the sequence
        chosen_times = np.random.choice(timestamps, 5)  # choose five of them randomly
        for t in chosen_times:  # iterate over the selected timestamps
            scene = sequence.get_scene(t)  # collect the data which belong to the current timestamp
            radar_data = scene.radar_data  # retrieve the radar data which belong to this scene
            y_true = np.array([ClassificationLabel.label_to_clabel(x) for x in radar_data["label_id"]])  # map labels
            valid_points = y_true != None  # filter invalid points
            y_true = y_true[valid_points]  # keep only valid points
            y_true = [x.value for x in y_true]  # get value of enum type to work with integers
            track_ids = radar_data["track_id"]
            X = features_from_radar_data(radar_data[valid_points])  # construct feature vector
            if return_track_ids:
                yield X, y_true, track_ids
            else:
                yield X, y_true


def validation_data_generator(validation_sequences: list, path_to_dataset: str, return_track_ids=False):
    """
    Similar to the mock training data generator, this generator method returns validation data.
    :param validation_sequences: List of sequence names which should be used for validation of a classifier
    :param path_to_dataset: path to the data set on the hard drive
    :param return_track_ids:  If true, in addition to the feature vectors and class labels, also the track ids are
    returned.
    :return: Feature vectors X, true labels y_true, detection uuids, the sequence name, and optionally the track_ids
    """
    for sequence_name in validation_sequences:
        try:
            sequence = Sequence.from_json(os.path.join(path_to_dataset, "data", sequence_name, "scenes.json"))
        except FileNotFoundError:
            continue

        for scene in sequence.scenes():  # iterate over all scenes in the sequence
            radar_data = scene.radar_data  # retrieve the radar data which belong to this scene
            y_true = np.array([ClassificationLabel.label_to_clabel(x) for x in radar_data["label_id"]])  # map labels
            valid_points = y_true != None  # filter invalid points
            y_true = y_true[valid_points]  # keep only valid points
            y_true = [x.value for x in y_true]  # get value of enum type to work with integers
            X = features_from_radar_data(radar_data[valid_points])  # construct feature vector
            uuids = radar_data["uuid"][valid_points]
            track_ids = radar_data["track_id"][valid_points]
            if return_track_ids:
                yield X, y_true, uuids, sequence_name, track_ids
            else:
                yield X, y_true, uuids, sequence_name


def main():
    # MODIFY THIS LINE AND INSERT PATH WHERE YOU STORED THE RADARSCENES DATASET
    path_to_dataset = "/home/USERNAME/datasets/RadarScenes"
    sequence_file = os.path.join(path_to_dataset, "data", "sequences.json")

    if not os.path.exists(sequence_file):
        print("Please modify this example so that it contains the correct path to the dataset on your machine.")
        return

    # load sequences.json file and obtain list of sequences for training.
    training_sequences = get_training_sequences(sequence_file)
    # load sequences.json file and obtain list of sequences for validation.
    validation_sequences = get_validation_sequences(sequence_file)
    print("Found {} sequences for training and {} sequences for validation.".format(len(training_sequences),
                                                                                    len(validation_sequences)))

    print("-" * 120)
    print("Mocking a semantic segmentation network...")

    classifier = SemSegNetwork()

    # For this example, only a subset of the training/validation files is used.
    # In a real application of course all files would be used
    training_sequences = training_sequences[112:115]
    validation_sequences = validation_sequences[23:24]

    # training loop for the classifier
    print("Training of mock-classifier...", end=" ", flush=True)
    for X, y_true in train_data_generator(training_sequences, path_to_dataset):
        classifier.train(X, y_true)
    print("Done!")

    # Validation loop
    print("Evaluating trained classifier on validation data...", end=" ", flush=True)
    predictions = {}
    for X, y_true, uuids, sequence_name in validation_data_generator(validation_sequences, path_to_dataset):
        if sequence_name not in predictions:
            predictions[sequence_name] = {}
        classifier._y_true_test = y_true  # this is only used to set the internal data of our fake-classifier
        y_pred = classifier.predict(X)  # predict for each point in X a class label
        for y, uid in zip(y_pred, uuids):  # store predictions in a dictionary along with the uuid of the points
            predictions[sequence_name][uid] = y
    print("Done!")

    current_dir = os.getcwd()
    for sequence_name in predictions:  # iterate over all unique sequences
        name = os.path.splitext(sequence_name)[0]
        output_name = os.path.join(current_dir, name + "_predictions.json")  # create output name for this sequence
        print("Writing predictions for sequence {} to file {}.".format(sequence_name, output_name))
        # write predictions to json file. This file can be loaded with the GUI tool to visualize the predictions
        per_point_predictions_to_json(predictions[sequence_name], output_name, ClassificationLabel.translation_dict(),
                                      schema=PredictionFileSchemas.SemSeg)

    print("Done with semantic segmentation!")
    print("-" * 120)
    print("\n")
    print("Mocking an instance segmentation network...")

    classifier = InstSegNetwork()
    print("Training of mock-classifier...", end=" ", flush=True)
    for X, y_true, y_inst in train_data_generator(training_sequences, path_to_dataset, return_track_ids=True):
        classifier.train(X, y_true, y_inst)
    print("Done!")

    # Validation loop instance segmentation
    print("Evaluating trained instance segmentation network on validation data...", end=" ", flush=True)
    predictions = {}
    for X, y_true, uuids, sequence_name, y_inst in validation_data_generator(validation_sequences, path_to_dataset,
                                                                             return_track_ids=True):
        if sequence_name not in predictions:
            predictions[sequence_name] = {}
        classifier._y_true_test = y_true  # this is only used to set the internal data of our fake-classifier
        classifier._y_inst_true = y_inst
        y_pred_labelid, y_pred_instid = classifier.predict(X)  # predict for each point in X a class label
        for y_lid, y_tid, uid in zip(y_pred_labelid, y_pred_instid, uuids):
            # store predictions in a dictionary along with the uuid of the points
            predictions[sequence_name][uid] = [int(y_lid), int(y_tid)] # casting to int for JSON serialization
    print("Done!")

    # write instance segmentation results back to a file
    current_dir = os.getcwd()
    for sequence_name in predictions:  # iterate over all unique sequences
        name = os.path.splitext(sequence_name)[0]
        output_name = os.path.join(current_dir,
                                   name + "_inst_seg_predictions.json")  # create output name for this sequence
        print("Writing predictions for sequence {} to file {}.".format(sequence_name, output_name))
        # write predictions to json file. This file can be loaded with the GUI tool to visualize the predictions
        per_point_predictions_to_json(predictions[sequence_name], output_name, ClassificationLabel.translation_dict(),
                                      schema=PredictionFileSchemas.InstSeg)

    print("Done with instance segmentation!")


if __name__ == '__main__':
    main()
