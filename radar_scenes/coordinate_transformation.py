import numpy as np


def batch_transform_3d_vector(trafo_matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Applies a 3x3 transformation matrix to every (1,3) vector contained in vec.
    Vec has shape (n_vec, 3)
    :param trafo_matrix: numpy array with shape (3,3)
    :param vec: numpy array with shape (n_vec, 3)
    :return: Transformed vector. Numpy array of shape (n_vec, 3)
    """
    return np.einsum('ij,kj->ki', trafo_matrix, vec)


def trafo_matrix_seq_to_car(odometry: np.ndarray) -> np.ndarray:
    """
    Computes the transformation matrix from sequence coordinates to car coordiantes, given an odometry entry.
    :param odometry: Numpy array containing at least the names fields "x_seq", "y_seq" and "yaw_seq" which give the
    position and orientation of the sensor vehicle.
    :return: Numpy array with shape (3,3), the transformation matrix. Last column is the translation vector.
    """
    x_car = odometry["x_seq"]
    y_car = odometry["y_seq"]
    yaw_car = odometry["yaw_seq"]
    c = np.cos(yaw_car)
    s = np.sin(yaw_car)
    return np.array([[c, s, -x_car * c - y_car * s],
                     [-s, c, x_car * s - y_car * c],
                     [0, 0, 1]])


def transform_detections_sequence_to_car(x_seq: np.ndarray, y_seq: np.ndarray, odometry: np.ndarray):
    """
    Computes the transformation matrix from sequence coordinates (global coordinate system) to car coordinates.
    The position of the car is extracted from the odometry array.
    :param x_seq: Shape (n_detections,). Contains the x-coordinate of the detections in the sequence coord. system.
    :param y_seq: Shape (n_detections,). Contains the y-coordinate of the detections in the sequence coord. system.
    :param odometry: Numpy array containing at least the names fields "x_seq", "y_seq" and "yaw_seq" which give the
    position and orientation of the sensor vehicle.
    :return: Two 1D numpy arrays, both of shape (n_detections,). The first array contains the x-coordinate and the
    second array contains the y-coordinate of the detections in car coordinates.
    """
    trafo_matrix = trafo_matrix_seq_to_car(odometry)
    v = np.ones((len(x_seq), 3))
    v[:, 0] = x_seq
    v[:, 1] = y_seq
    res = batch_transform_3d_vector(trafo_matrix, v)
    return res[:, 0], res[:, 1]
