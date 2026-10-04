import argparse
import sys
import traceback
import warnings
from typing import List, Optional

from modules.analyze import AnalyzeData
from modules.cleansing import clean_data
from modules.enums import OutputTypeEnum, PrintColorEnum
from modules.extraction import read_visits
from modules.training import TrainingData

OUTPUT_CHOICES = {"file": OutputTypeEnum.FILE, "f": OutputTypeEnum.FILE,
                  "console": OutputTypeEnum.CONSOLE, "c": OutputTypeEnum.CONSOLE}


def read_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Read arguments from the console
    :param argv: argument list (defaults to sys.argv)
    :return: parsed arguments, with output options converted to OutputTypeEnum
    """
    parser = argparse.ArgumentParser(description="Predict the number of visits of Kijiji car ads")
    parser.add_argument("--analyze", choices=OUTPUT_CHOICES, help="Run the EDA, report to file or console")
    parser.add_argument("--training", choices=OUTPUT_CHOICES, help="Train and evaluate models, report to file/console")
    parser.add_argument("--save-model", action="store_true", help="Save the best pipeline to reports/models")
    parser.add_argument("--no-plots", action="store_true", help="Skip chart generation")
    parser.add_argument("--fast", action="store_true", help="Smaller models and fewer CV folds (quick check)")
    parser.add_argument("--sample", type=int, help="Use a random sample of N ads (quick experiments)")
    args = parser.parse_args(argv)

    if not (args.analyze or args.training):
        parser.error("choose at least one of --analyze or --training")

    args.analyze = OUTPUT_CHOICES.get(args.analyze)
    args.training = OUTPUT_CHOICES.get(args.training)
    return args


def section(title: str, color: str) -> None:
    print(color, "-" * 50)
    print(PrintColorEnum.BOLD, color, title, PrintColorEnum.END)


def main(argv: Optional[List[str]] = None) -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = read_arguments(argv)
    plots = not args.no_plots

    raw = read_visits()
    if args.sample:
        raw = raw.sample(n=min(args.sample, len(raw)), random_state=0)

    if args.analyze:
        section("ANALYZE DATA:", PrintColorEnum.PURPLE)
        df, _ = AnalyzeData(args.analyze, plots=plots).process(raw)
    else:
        df, _ = clean_data(raw)

    if args.training:
        section("TRAINING DATA:", PrintColorEnum.CYAN)
        result = TrainingData(df, args.training, save_model=args.save_model, plots=plots, fast=args.fast).process()
        print(PrintColorEnum.CYAN, f"Best model: {result['best_name']}", PrintColorEnum.END)
        print(result["test_metrics"].round(3))

    print(PrintColorEnum.GREEN, PrintColorEnum.BOLD, "All done!", PrintColorEnum.END, "\U0001F64C")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        print(PrintColorEnum.RED, PrintColorEnum.BOLD, "Error!!", "\U0001F525", PrintColorEnum.END)
        print("\tDescription: ", PrintColorEnum.RED, ex, PrintColorEnum.END)
        traceback.print_exc()
        sys.exit(1)
