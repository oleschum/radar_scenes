import json

RADAR_DEFAULT_MOUNTING = {
    1: {"x": 3.663, "y": -0.873, "yaw": -1.48418552},
    2: {"x": 3.86, "y": -0.70, "yaw": -0.436185662},
    3: {"x": 3.86, "y": 0.70, "yaw": 0.436},
    4: {"x": 3.663, "y": 0.873, "yaw": 1.484},
}


def get_mounting(sensor_id: int, json_path=None) -> dict:
    """
    Returns the sensor mounting positions of a single sensor with id sensor_id.
    The positions and the azimuth angle are given relative to the car coordinate system.
    :param sensor_id: Integer sensor id.
    :param json_path: str, path to the sensor.json file. If not defined, the default mounting positions are used.
    :return: dictionary containing the x and y position of the sensor in car coordinates as well as the yaw angle:
            structure: {"x": x_val, "y": y_val, "yaw": yaw_val}
    """
    if json_path is None:
        return RADAR_DEFAULT_MOUNTING[sensor_id]
    else:
        with open(json_path, "r") as f:
            data = json.load(f)
        radar_name = "radar_{}".format(sensor_id)
        if radar_name in data:
            return data[radar_name]
        else:
            raise KeyError("Radar {} does not exist in the json file {}.".format(radar_name, json_path))
