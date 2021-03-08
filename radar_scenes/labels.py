from enum import Enum
from typing import Union


class Label(Enum):
    """
    The Labels enum contains all semantic labels available in the data set.
    """
    CAR = 0
    LARGE_VEHICLE = 1
    TRUCK = 2
    BUS = 3
    TRAIN = 4
    BICYCLE = 5
    MOTORIZED_TWO_WHEELER = 6
    PEDESTRIAN = 7
    PEDESTRIAN_GROUP = 8
    ANIMAL = 9
    OTHER = 10
    STATIC = 11

    @staticmethod
    def label_id_to_name(label_id: int) -> str:
        """
        Convert an integer label_id to the string representation of the corresponding enum
        :param label_id: Label ID of a class for which a string is desired.
        :return: The class name as a string
        """
        return Label(label_id).name


class ClassificationLabel(Enum):
    """
    This Enum contains a subset of the complete label hierarchy. All classes from the "Labels" enum are mapped
    to a set of 6 different classes.
    This set of labels may be used for machine learning based classification tasks to reduce complexity .
    Mappings to "None" indicate that this class should be omitted in training/evaluation.
    """
    CAR = 0
    PEDESTRIAN = 1
    PEDESTRIAN_GROUP = 2
    TWO_WHEELER = 3
    LARGE_VEHICLE = 4
    STATIC = 5

    @staticmethod
    def label_to_clabel(label: Union[Label, int]) -> "ClassificationLabel":
        """
        Convert a member of the Labels enum to a ClassificationLabel
        :param label: Label that shall be translated. Can be either a "Label" instance or an integer representation
        of a Label
        :return: a ClassificationLabel
        """
        dic = ClassificationLabel.translation_dict()
        if not isinstance(label, Label):
            label = Label(label)
        return dic[label]

    @staticmethod
    def translation_dict() -> dict:
        """
        Provides the translation from Label to ClassificationLabel
        :return: a dictionary with Label instances as key and a ClassificationLabel as value.
        """
        return {
            Label.CAR: ClassificationLabel.CAR,
            Label.LARGE_VEHICLE: ClassificationLabel.LARGE_VEHICLE,
            Label.TRUCK: ClassificationLabel.LARGE_VEHICLE,
            Label.BUS: ClassificationLabel.LARGE_VEHICLE,
            Label.TRAIN: ClassificationLabel.LARGE_VEHICLE,
            Label.BICYCLE: ClassificationLabel.TWO_WHEELER,
            Label.MOTORIZED_TWO_WHEELER: ClassificationLabel.TWO_WHEELER,
            Label.PEDESTRIAN: ClassificationLabel.PEDESTRIAN,
            Label.PEDESTRIAN_GROUP: ClassificationLabel.PEDESTRIAN_GROUP,
            Label.ANIMAL: None,
            Label.OTHER: None,
            Label.STATIC: ClassificationLabel.STATIC
        }

    @staticmethod
    def label_id_to_name(cl_label_id: int) -> str:
        """
        Convert an integer classification_label_id to the string representation of the corresponding enum
        :param cl_label_id: Label ID of a class for which a string is desired.
        :return: The class name as a string
        """
        return ClassificationLabel(cl_label_id).name
