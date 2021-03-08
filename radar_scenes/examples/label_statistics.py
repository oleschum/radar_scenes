import os
import numpy as np
from radar_scenes.sequence import Sequence
from radar_scenes.labels import Label, ClassificationLabel
from collections import Counter


def count_label_ids(sequence: Sequence):
    """
    Iterate over all scenes in a sequence and collect for each scene, how many detections belong to each class.
    :param sequence: a measurement sequence
    :return: A list of dictionaries. Each dict contains as key the label ids and as value the number of times this
    label_id occured.
    """
    # iterate over all scenes in the sequence and collect the number of labeled detections per class
    labels_per_scene = []
    for scene in sequence.scenes():
        label_ids = scene.radar_data["label_id"]
        c = Counter(label_ids)
        labels_per_scene.append(dict(c.items()))
    return labels_per_scene


def count_unique_objects(sequence: Sequence):
    """
    For each scene in the sequence, count how many different objects exist.
    Objects labeled as "clutter" are excluded from the counting, as well as the static detections.
    :param sequence: A measurement sequence
    :return: a list holding the number of unique objects for each scene
    """
    objects_per_scene = []
    for scene in sequence.scenes():
        track_ids = scene.radar_data["track_id"]
        label_ids = scene.radar_data["label_id"]

        valid_idx = np.where(label_ids != Label.STATIC.value)[0]
        unique_tracks = set(track_ids[valid_idx])
        objects_per_scene.append(len(unique_tracks))
    return objects_per_scene


def main():
    # MODIFY THIS LINE AND INSERT PATH WHERE YOU STORED THE RADARSCENES DATASET
    path_to_dataset = "/home/USERNAME/datasets/RadarScenes"

    # Define the *.json file from which data should be loaded
    filename = os.path.join(path_to_dataset, "data", "sequence_137", "scenes.json")

    if not os.path.exists(filename):
        print("Please modify this example so that it contains the correct path to the dataset on your machine.")
        return

    # create sequence object from json file
    sequence = Sequence.from_json(filename)
    labels_per_scene = count_label_ids(sequence)

    # obtain labels for one specific scene:
    scene_id = 1234
    labels = labels_per_scene[scene_id]
    for label_id, n in labels.items():
        print("In scene {}, class {} occurred {} times".format(scene_id, Label.label_id_to_name(label_id), n))

    print("\n")
    # count all labels at once
    all_labels = sequence.radar_data["label_id"]
    c = Counter(all_labels)
    for label_id, n in c.items():
        print("In the whole sequence, class {} occurred {} times".format(Label.label_id_to_name(label_id), n))

    # mapping to a reduced set of labels
    c_label_ids = []
    n_ignored = 0
    for l in all_labels:
        mapped_label = ClassificationLabel.label_to_clabel(Label(l))
        if mapped_label is None:
            n_ignored += 1
            continue
        c_label_ids.append(mapped_label.value)

    c = Counter(c_label_ids)
    print("\nMapping to a reduced label set results in the following distribution:")
    print("{} detections were ignored (mapped to None)".format(n_ignored))
    for label_id, n in c.items():
        print("In the whole sequence, class {} occurred {} times".format(ClassificationLabel.label_id_to_name(label_id),
                                                                         n))

    print("\nCounting the number of unique dynamic objects in each scene:")
    object_counts = count_unique_objects(sequence)
    print("The most unique objects appear in scene {} in which {} different objects were labeled.".format(
        np.argmax(object_counts), object_counts[np.argmax(object_counts)]))


if __name__ == '__main__':
    main()
