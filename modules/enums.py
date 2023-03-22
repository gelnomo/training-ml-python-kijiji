from enum import Enum


class OutputTypeEnum(Enum):
    FILE = 1
    CONSOLE = 2


class OptionEnum(Enum):
    YES = 1
    NO = 0


class OutlierTypeEnum(Enum):
    Z_SCORE = 1
    Z_R_SCORE = 2
    IQR = 3
    WINSORIZATION = 4
    DBSCAN = 5
    ISOLATION_FOREST = 6


class PrintColorEnum:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
