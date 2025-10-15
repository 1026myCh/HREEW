#!/usr/bin/env python3
"""Run EEW test harness against local testdata samples."""

from __future__ import annotations

import argparse
import json
import inspect
from pathlib import Path
import sys
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
ALG_DIR = REPO_ROOT / 'alg'
if str(ALG_DIR) not in sys.path:
    sys.path.insert(0, str(ALG_DIR))

try:
    from StaticVar import StaticVar as persistent  # type: ignore
    from StaticVar import Static_EEW_Params as EEWParams  # type: ignore
    from EEW_Test_Massive import EEW_Test_Massvie  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    missing = getattr(exc, 'name', None) or str(exc)
    raise SystemExit(
        f"缺少依赖模块：{missing}。请先运行 'pip install -r requirements.txt' 安装所需第三方库。"
    ) from exc

HEADERS = [
    'time_start', 'epi_No.', 'EQ_Info', 'STA_Name', 'Epi_time', 'epiLon_real', 'epiLat_real',
    'Magnitude_real', 'Distance_Real', 'sprate', 'STA_Long', 'STA_Lat', 'PGA_Real', 'Azimuth_real',
    'StartT', 'P_time', 'S_time', 'Magnitude', 'Azimuth', 'Epi_time', 'Distance', 'Epi_Long', 'Epi_Lat',
    'PGA_Pred', 'S_time_cal', 'PGA_Curr', 'PGV_Curr', 'PGD_Curr', 'DurCurr', 'S_time2', 'AlarmLevel',
    'delt_Epi', 'POS', 'Is_EQK', 'Newinfo', 'PGAPre'
]


def _collect_defaults(cls) -> Dict[str, object]:
    defaults: Dict[str, object] = {}
    for key, value in cls.__dict__.items():
        if key.startswith('__'):
            continue
        if inspect.isroutine(value):
            continue
        defaults[key] = value
    return defaults


PERSISTENT_DEFAULTS = _collect_defaults(persistent)
EEW_PARAMS_DEFAULTS = _collect_defaults(EEWParams)


def reset_static_state() -> None:
    for key, value in PERSISTENT_DEFAULTS.items():
        setattr(persistent, key, value)
    for key, value in EEW_PARAMS_DEFAULTS.items():
        setattr(EEWParams, key, value)


def rows_to_dicts(rows: Iterable[Iterable[object]]) -> List[Dict[str, object]]:
    return [dict(zip(HEADERS, row)) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=REPO_ROOT / 'testdata',
                        help='Directory that contains sub-folders with waveform samples (default: %(default)s)')
    parser.add_argument('--excel-dir', type=Path,
                        help='Optional directory to store Excel outputs alongside printed summaries')
    parser.add_argument('--nature-mode', type=int, choices=[0, 1],
                        help='Override NatureMode configuration before running tests')
    args = parser.parse_args()

    data_root = args.data_root
    if not data_root.exists():
        raise SystemExit(f'Data directory {data_root} does not exist')

    excel_dir: Path | None = args.excel_dir
    if excel_dir:
        excel_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not dataset_dirs:
        raise SystemExit(f'No dataset folders found under {data_root}')

    for idx, dataset_dir in enumerate(dataset_dirs):
        reset_static_state()
        excel_path = None
        if excel_dir:
            excel_path = excel_dir / f'{dataset_dir.name}.xlsx'

        if args.nature_mode is not None:
            EEWParams.NatureMode = str(args.nature_mode)

        print(f'=== Dataset {dataset_dir.name} (index {idx}) ===')
        rows = EEW_Test_Massvie(str(dataset_dir), str(excel_path) if excel_path else None, idx)
        result_dicts = rows_to_dicts(rows)
        if result_dicts:
            print(json.dumps(result_dicts, ensure_ascii=False, indent=2))
        else:
            print('No results produced for this dataset.')
        print()


if __name__ == '__main__':
    main()
