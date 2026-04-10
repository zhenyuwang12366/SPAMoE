import json
import importlib.util
from pathlib import Path
import sys
import types


_PARSER_UTILS_PATH = Path(__file__).resolve().parents[2] / "utils" / "parser_utils.py"
config_pkg = types.ModuleType("config")
seismic_config_mod = types.ModuleType("config.seismic_moe_config")
seismic_config_mod.SPECIFIC_TYPE_VARIANTS = {
    "curve_vel": ("curve_vel_a", "curve_vel_b"),
    "curve_fault": ("curve_fault_a", "curve_fault_b"),
    "flat_vel": ("flat_vel_a", "flat_vel_b"),
    "flat_fault": ("flat_fault_a", "flat_fault_b"),
    "style_style": ("style_style_a", "style_style_b"),
    "all": "all",
}
config_pkg.seismic_moe_config = seismic_config_mod
sys.modules["config"] = config_pkg
sys.modules["config.seismic_moe_config"] = seismic_config_mod

_SPEC = importlib.util.spec_from_file_location("test_parser_utils_module", _PARSER_UTILS_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
build_argparser_and_parse = _MODULE.build_argparser_and_parse


def test_build_argparser_and_parse_loads_json_config(tmp_path):
    config_path = tmp_path / "train_config.json"
    config_path.write_text(
        json.dumps(
            {
                "epochs": 123,
                "batch_size": 4,
                "family": "vel",
                "learning_rate": 5e-4,
                "milestones": [5, 10, 15],
                "lr_warmup_epochs": 9,
                "mixed_precision": True,
            }
        ),
        encoding="utf-8",
    )

    args = build_argparser_and_parse(["--config", str(config_path)])

    assert args.config == str(config_path)
    assert args.epochs == 123
    assert args.batch_size == 4
    assert args.family == "vel"
    assert args.learning_rate == 5e-4
    assert args.milestones == [5, 10, 15]
    assert args.lr_warmup_epochs == 9
    assert args.mixed_precision is True
    assert "lr_warmup_epochs" in args.user_specified_args


def test_cli_args_do_not_override_config_file(tmp_path):
    config_path = tmp_path / "train_config.json"
    config_path.write_text(
        json.dumps(
            {
                "epochs": 50,
                "batch_size": 2,
                "lr_warmup_epochs": 11,
                "mixed_precision": True,
                "family": "style",
            }
        ),
        encoding="utf-8",
    )

    args = build_argparser_and_parse(
        [
            "--config",
            str(config_path),
            "--epochs",
            "80",
            "--disable_mixed_precision",
        ]
    )

    assert args.epochs == 50
    assert args.batch_size == 2
    assert args.lr_warmup_epochs == 11
    assert args.mixed_precision is True
    assert args.family == "style"
    assert "mixed_precision" in args.user_specified_args
    assert "epochs" in args.user_specified_args


def test_args_file_replaces_runtime_args(tmp_path):
    args_path = tmp_path / "args.json"
    args_path.write_text(
        json.dumps(
            {
                "mode": "train",
                "epochs": 33,
                "batch_size": 6,
                "family": "fault",
                "mixed_precision": True,
                "parser": "ArgumentParser(...)",
                "user_specified_args": ["epochs", "batch_size", "family", "mixed_precision"],
            }
        ),
        encoding="utf-8",
    )

    args = build_argparser_and_parse(
        [
            "--args",
            str(args_path),
            "--epochs",
            "99",
            "--family",
            "vel",
            "--disable_mixed_precision",
        ]
    )

    assert args.args_file == str(args_path)
    assert args.config is None
    assert args.epochs == 33
    assert args.batch_size == 6
    assert args.family == "fault"
    assert args.mixed_precision is True
    assert "epochs" in args.user_specified_args
    assert "family" in args.user_specified_args


def test_args_file_takes_priority_over_config_file(tmp_path):
    args_path = tmp_path / "args.json"
    config_path = tmp_path / "config.json"

    args_path.write_text(
        json.dumps(
            {
                "mode": "train",
                "epochs": 12,
                "batch_size": 7,
                "family": "fault",
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "epochs": 88,
                "batch_size": 3,
                "family": "vel",
            }
        ),
        encoding="utf-8",
    )

    args = build_argparser_and_parse(
        [
            "--config",
            str(config_path),
            "--args",
            str(args_path),
            "--epochs",
            "99",
        ]
    )

    assert args.args_file == str(args_path)
    assert args.config is None
    assert args.epochs == 12
    assert args.batch_size == 7
    assert args.family == "fault"
