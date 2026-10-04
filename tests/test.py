"""CLI tests."""

import pytest

from main import read_arguments
from modules.enums import OutputTypeEnum


def test_read_arguments_maps_output_types():
    args = read_arguments(["--analyze", "c", "--training", "file", "--save-model"])
    assert args.analyze == OutputTypeEnum.CONSOLE
    assert args.training == OutputTypeEnum.FILE
    assert args.save_model


def test_read_arguments_requires_an_action():
    with pytest.raises(SystemExit):
        read_arguments([])


def test_read_arguments_rejects_unknown_output():
    with pytest.raises(SystemExit):
        read_arguments(["--analyze", "pdf"])
