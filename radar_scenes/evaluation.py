import json
from enum import Enum


class PredictionFileSchemas(Enum):
    SemSeg = 1  # semantic segmentation. only per-point prediction for class label
    InstSeg = 2  # instance segmentation. per point prediction for class label and instance label


def per_point_predictions_to_json(predictions: dict, filename: str, label_translation: dict,
                                  schema: PredictionFileSchemas) -> dict:
    """
    Writes predictions to a json file. The created output file can be read by the viewer so that predicted class
    label and predicted instance label can be visualized.
    :param predictions: Dictionary containing the predictions. Key is the uuid of a detection.
    Value can be either a single integer (a single class label, e.g. for semantic segmentation tasks)
    or a list with two entries: a class label and an instance label. The first entry of the list should be the class
    label and the second entry an integer label for the instance the respective detection is predicted to belong to.
    :param filename: Full path to the file to which the results shall be written (output filename)
    :param label_translation: A dictionary containing the label mapping. Such a mapping may have been used if not
    all semantic classes were used for training/evaluation.
    :param schema: An enum entry from PredictionFileSchemas, indicating what kind of predictions can be expected
    :return: The contents of the json file in the form of a dictionary.
    """
    result_dict = {}
    mapping_int = {}  # key is original label as int, value is new integer
    mapping_name = {}  # key is original label as int, value is string representation of new label
    for label, other_label in label_translation.items():
        if isinstance(label, Enum):
            label_int = label.value
        else:
            label_int = label
        if isinstance(other_label, Enum):
            other_label_int = other_label.value
            other_label_str = other_label.name
        else:
            other_label_int = other_label
            other_label_str = str(other_label)
        mapping_int[label_int] = other_label_int
        if other_label_int is not None:
            mapping_name[other_label_int] = other_label_str
    result_dict["schema"] = schema.value
    result_dict["label_mapping"] = mapping_int
    result_dict["new_label_names"] = mapping_name
    result_dict["predictions"] = {}
    for detection_uuid, pred_label in predictions.items():
        # if schema == PredictionFileSchemas.SemSeg, then pred_label is a single integer representing the class
        # if schema == PredictionFileSchemas.InstSeg, then pred_label is a list: [class_label, inst_label]
        if isinstance(detection_uuid, bytes):
            detection_uuid = detection_uuid.decode()
        result_dict["predictions"][detection_uuid] = pred_label

    with open(filename, "w") as f:
        json.dump(result_dict, f, ensure_ascii=True, indent=2)

    return result_dict
