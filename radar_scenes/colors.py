import pyqtgraph as pg
from functools import lru_cache


class Colors:
    """
    Helper class for colors. Provides mappings from sensor ids to colors as well as colors for the individual classes.
    """
    red = "#f02b2b"
    blue = "#4763ff"
    green = "#47ff69"
    light_green = "#73ff98"
    orange = "#ff962e"
    violet = "#c561d4"
    indigo = "#8695e3"
    grey = "#7f8c8d"
    yellow = "#ffff33"
    lime = "#c6ff00"
    amber = "#ffd54f"
    teal = "#19ffd2"
    pink = "#ff6eba"
    brown = "#c97240"
    black = "#1e272e"
    midnight_blue = "#34495e"
    deep_orange = "#e64a19"
    light_blue = "#91cded"
    light_gray = "#dedede"
    gray = "#888888"

    sensor_id_to_color = {
        1: red,
        2: blue,
        3: green,
        4: pink
    }

    label_id_to_color = {
        0: violet,
        1: orange,
        2: green,
        3: light_green,
        4: brown,
        5: teal,
        6: light_blue,
        7: yellow,
        8: pink,
        9: blue,
        10: indigo,
        11: gray
    }

    object_colors = [red, blue, green, light_green, orange, violet, yellow, teal, pink, brown,
                     light_blue, lime, deep_orange, amber, indigo]


@lru_cache(maxsize=50)
def brush_for_color(color):
    """
    Simple wrapper with lru cache for brushes
    :param color: hex color string like "#ff00ff"
    :return: a QBrush object
    """
    return pg.mkBrush(color)
