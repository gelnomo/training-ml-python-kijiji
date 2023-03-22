import argparse
import sys
import warnings
from modules.enums import OutputTypeEnum, OptionEnum, PrintColorEnum
from modules.analyze import AnalyzeData
from modules.training import TrainingData
from typing import List


def read_arguments() -> List:
    """
    Read all arguments from console
    :return: List of arguments
    """
    parser = argparse.ArgumentParser(description="Predict Kijiji Car Price")
    parser.add_argument(
        "--analyze",
        dest="analyze",
        type=str,
        help="Generating results (File/Console)",
        choices=["file", "f", "console", "c"],
    )
    parser.add_argument(
        "--training",
        dest="training",
        type=str,
        choices=["file", "f", "console", "c"],
        help="Generating results (File/Console)",
    )
    parser.add_argument(
        "--normalization",
        dest="normalization",
        type=str,
        choices=["yes", "true", "t", "y", "no", "false", "f", "n"],
        help="Normalization (yes/no)",
    )
    parser.add_argument(
        "--storage_model",
        dest="storage_model",
        type=str,
        choices=["yes", "true", "t", "y", "no", "false", "f", "n"],
        help="Normalization (yes/no)",
    )
    args = parser.parse_args()

    if args.analyze and args.analyze.lower() in ["file", "f"]:
        args.analyze = OutputTypeEnum.FILE
    elif args.analyze and args.analyze.lower() in ["console", "c"]:
        args.analyze = OutputTypeEnum.CONSOLE

    if args.training and args.training.lower() in ["file", "f"]:
        args.training = OutputTypeEnum.FILE
    elif args.training and args.training.lower() in ["console", "c"]:
        args.training = OutputTypeEnum.CONSOLE

    if args.normalization and args.normalization.lower() in [
        "yes",
        "true",
        "t",
        "y",
    ]:
        args.normalization = OptionEnum.YES
    else:
        args.normalization = OptionEnum.NO

    if args.storage_model and args.storage_model.lower() in [
        "yes",
        "true",
        "t",
        "y",
    ]:
        args.storage_model = OptionEnum.YES
    else:
        args.storage_model = OptionEnum.NO

    return args


if __name__ == "__main__":
    try:
        # Ignore warning messages
        warnings.filterwarnings("ignore")
        df, unique_cities, standard_normalization = None, None, None

        args = read_arguments()
        if args.analyze:
            print(PrintColorEnum.PURPLE, "-" * 50)
            print(
                PrintColorEnum.BOLD,
                PrintColorEnum.PURPLE,
                "ANALYZE DATA:",
                PrintColorEnum.END,
            )

            analyze_data = AnalyzeData(args.analyze, args.normalization)
            df, unique_cities, standard_normalization = analyze_data.process()
            del analyze_data

            print(PrintColorEnum.PURPLE, "-" * 50)

        if args.analyze and args.training:
            print(PrintColorEnum.CYAN, "-" * 50)
            print(
                PrintColorEnum.BOLD,
                PrintColorEnum.CYAN,
                "TRAINING DATA:",
                PrintColorEnum.END,
            )

            training_data = TrainingData(
                df, unique_cities, args.training, args.storage_model
            )
            training_data.process(standard_normalization)
            del training_data

            print(PrintColorEnum.CYAN, "-" * 50)

        del df
        del unique_cities
        del standard_normalization

        print(
            PrintColorEnum.GREEN,
            PrintColorEnum.BOLD,
            "All done!",
            PrintColorEnum.END,
            "\U0001F64C",
        )
    except Exception as ex:
        print(
            PrintColorEnum.RED,
            PrintColorEnum.BOLD,
            "Error!!",
            "\U0001F525",
            PrintColorEnum.END,
        )

        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno

        print("\tDescription: ", PrintColorEnum.RED, ex, PrintColorEnum.END)
        print(
            "\tException type: ", PrintColorEnum.RED, exception_type, PrintColorEnum.END
        )
        print("\tFile name: ", PrintColorEnum.RED, filename, PrintColorEnum.END)
        print("\tLine number: ", PrintColorEnum.RED, line_number, PrintColorEnum.END)
