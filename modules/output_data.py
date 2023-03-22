from modules.enums import OutputTypeEnum, PrintColorEnum
import datetime


class OutputResult:
    def __init__(self, output_type: OutputTypeEnum):
        """
        Initialize the output result with the output type and default file path.
        :param output_type: output type, console o file
        """
        self._title = None
        self.output_type = output_type
        format_date = datetime.datetime.now().strftime("%Y_%m_%d")
        self.file = f"reports/output/{format_date}_output.txt"

    @property
    def title(self) -> str:
        """
        Get the current title of the output result.
        :return: return the current value
        """
        #
        return self._title

    @title.setter
    def title(self, new_title: str):
        """
        Set the title of the output result.
        :param new_title: new value
        """
        self._title = new_title

    def process(self, content: str) -> None:
        """
        Output content
        :param content: content to process
        """
        # Process the given content with the current output type and title, if any.
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")

        if self.output_type == OutputTypeEnum.CONSOLE:
            if self._title:
                print(PrintColorEnum.BOLD, self._title.title(), PrintColorEnum.END)
            print(f"[{now}]\n{content}")
        else:
            f = open(self.file, "a")
            if self._title:
                f.write(f"{self._title.title()}\n")
            f.write(f"[{now}]\n{content}\n")
            f.close()
