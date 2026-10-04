import datetime
from pathlib import Path
from typing import Any

from modules.config import OUTPUT_DIR
from modules.enums import OutputTypeEnum, PrintColorEnum


class OutputResult:
    """Write report sections either to the console or to a timestamped text file (one per run)."""

    def __init__(self, output_type: OutputTypeEnum, output_dir: Path = OUTPUT_DIR):
        self.output_type = output_type
        self.title = None
        format_date = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.file = Path(output_dir) / f"{format_date}_output.txt"

    def process(self, content: Any) -> None:
        """
        Output content under the current title
        :param content: anything printable (text, dataframe, series...)
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.output_type == OutputTypeEnum.CONSOLE:
            if self.title:
                print(PrintColorEnum.BOLD, self.title, PrintColorEnum.END)
            print(f"[{now}]\n{content}\n")
        else:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file, "a", encoding="utf-8") as f:
                if self.title:
                    f.write(f"## {self.title}\n")
                f.write(f"[{now}]\n{content}\n\n")
