from features.utils import PrintColor
from features.enums import OutputTypeEnum
import datetime


class OutputResult:
    def __init__(self, output_type: OutputTypeEnum):
        # Initialize the output result with the output type and default file path.
        self._title = None
        self.output_type = output_type
        self.file = "data/output/result.txt"

    @property
    def title(self):
        # Get the current title of the output result.
        return self._title

    @title.setter
    def title(self, new_title):
        # Set the title of the output result.
        self._title = new_title

    def process(self, content):
        # Process the given content with the current output type and title, if any.
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")

        if self.output_type == OutputTypeEnum.CONSOLE:
            if self._title:
                print(PrintColor.BOLD, self._title.title(), PrintColor.END)
            print(f"[{now}]\n{content}")
        else:
            f = open(self.file, "a")
            if self._title:
                f.write(f"{self._title.title()}\n")
            f.write(f"[{now}]\n{content}\n")
            f.close()
