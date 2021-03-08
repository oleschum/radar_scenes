import os
from radar_scenes.sequence import Sequence
import numpy as np


def main():
    # MODIFY THIS LINE AND INSERT PATH WHERE YOU STORED THE RADARSCENES DATASET
    path_to_dataset = "/home/USERNAME/datasets/RadarScenes"


    # Define the *.json file from which data should be loaded
    # some random sequence is chosen here.
    filename = os.path.join(path_to_dataset, "data", "sequence_137", "scenes.json")

    if not os.path.exists(filename):
        print("Please modify this example so that it contains the correct path to the dataset on your machine.")
        return

    # create a Sequence object by passing the filename to the class method `from_json`
    sequence = Sequence.from_json(filename)

    print("The sequence contains {} different scenes".format(len(sequence)))

    # iterate over the individual radar measurements in this sequence and print the number of detections with RCS > 0
    start_time = sequence.first_timestamp

    # get the second scene in the sequence:
    second_scene = sequence.next_scene_after(start_time)
    print("The second scene was measured by the radar sensor with the id {}".format(second_scene.sensor_id))

    # get the second measurement of the same sensor which also measured the first scene:
    second_measurement = sequence.next_scene_after(start_time, same_sensor=True)
    assert second_measurement.sensor_id == sequence.get_scene(start_time).sensor_id

    for idx, scene in enumerate(sequence.scenes()):
        if idx == 0:
            # check that start_time of the sequence is in fact identical to the timestamp of the first returned scene
            assert start_time == scene.timestamp
        radar_data = scene.radar_data
        indices = np.where(radar_data["rcs"] > 0)[0]

        print("Scene number {} at timestamp {} contains {} detections with RCS > 0".format(idx, scene.timestamp,
                                                                                           len(indices)))

    # iterate only over measurement from radar 1:
    for scene in sequence.scenes(sensor_id=1):
        assert scene.sensor_id == 1


if __name__ == '__main__':
    main()
