import argparse
import sys
import warnings
from features.utils import PrintColor
from features.enums import OutputTypeEnum, OptionEnum
from features.analyze import AnalyzeData
from features.training import TrainingData


def read_arguments():
    parser = argparse.ArgumentParser(description="Predict Kijiji Car Price")
    parser.add_argument(
        "--analyze", dest="analyze", type=str, help="Generating results (File/Console)"
    )
    parser.add_argument(
        "--training",
        dest="training",
        type=str,
        help="Generating results (File/Console)",
    )
    parser.add_argument(
        "--normalization",
        dest="normalization",
        type=str,
        help="Normalization (yes/no)",
    )

    args = parser.parse_args()

    if args.analyze and args.analyze.lower() in ["file", "f", "1"]:
        args.analyze = OutputTypeEnum.FILE
    elif args.analyze and args.analyze.lower() in ["console", "c", "2"]:
        args.analyze = OutputTypeEnum.CONSOLE

    if args.training and args.training.lower() in ["file", "f", "1"]:
        args.training = OutputTypeEnum.FILE
    elif args.training and args.training.lower() in ["console", "c", "2"]:
        args.training = OutputTypeEnum.CONSOLE

    if args.normalization and args.normalization.lower() in [
        "yes",
        "true",
        "t",
        "y",
        "1",
    ]:
        args.normalization = OptionEnum.YES
    else:
        args.normalization = OptionEnum.NO

    return args


if __name__ == "__main__":
    try:
        # Ignore warning messages
        warnings.filterwarnings("ignore")
        df, unique_make, standard_normalization = None, None, None

        args = read_arguments()
        if args.analyze:
            print(PrintColor.PURPLE, "-" * 50)
            print(PrintColor.BOLD, PrintColor.PURPLE, "ANALYZE DATA:", PrintColor.END)

            analyze_data = AnalyzeData(args.analyze, args.normalization)
            df, unique_make, standard_normalization = analyze_data.process()
            del analyze_data

            print(PrintColor.PURPLE, "-" * 50)

        if args.analyze and args.training:
            print(PrintColor.CYAN, "-" * 50)
            print(PrintColor.BOLD, PrintColor.CYAN, "TRAINING DATA:", PrintColor.END)

            training_data = TrainingData(df, unique_make, args.training)
            training_data.process(standard_normalization)
            del training_data

            print(PrintColor.CYAN, "-" * 50)

        print(
            PrintColor.GREEN, PrintColor.BOLD, "All done!", PrintColor.END, "\U0001F64C"
        )
    except Exception as ex:
        print(PrintColor.RED, PrintColor.BOLD, "Error!!", "\U0001F525", PrintColor.END)

        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno

        print("\tDescription: ", PrintColor.RED, ex, PrintColor.END)
        print("\tException type: ", PrintColor.RED, exception_type, PrintColor.END)
        print("\tFile name: ", PrintColor.RED, filename, PrintColor.END)
        print("\tLine number: ", PrintColor.RED, line_number, PrintColor.END)
