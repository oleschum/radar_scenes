class Scene:
    """
    A small wrapper class for a single Scene.
    """
    def __init__(self):
        self.timestamp = None
        self.odometry_timestamp = None
        self.radar_data = None
        self.odometry_data = None
        self.camera_image_name = None
        self.sensor_id = None
