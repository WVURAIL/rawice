import datetime
import sys
import types

import numpy as np
from click.testing import CliRunner

from test_rawice import write_acquisition


if "ephem" not in sys.modules:
    ephem = types.ModuleType("ephem")
    ephem.Observer = lambda: types.SimpleNamespace(lat=None, long=None)
    sys.modules["ephem"] = ephem

if "chime_frb_constants" not in sys.modules:
    constants = types.ModuleType("chime_frb_constants")
    constants.CHIME_LATITUDE_DEG = 49.32
    constants.CHIME_LONGITUDE_DEG = -119.62
    sys.modules["chime_frb_constants"] = constants

from raw_acq_diagnostics import raw_acq_diagnostics as diagnostics
from raw_acq_diagnostics import raw_acq_cli


def test_direct_filename_loading_uses_all_frames_and_utc(tmp_path):
    filename = write_acquisition(tmp_path / "diagnostics.h5")
    acquisition = diagnostics.RawAcq(
        filenames=np.array([str(filename)]),
        plot_dir=tmp_path,
    )

    assert len(acquisition.timestream) == 4
    assert acquisition.num_frames == 2
    assert acquisition.start_time.tzinfo is not None
    assert acquisition.start_time.utcoffset() == datetime.timedelta(0)
    assert acquisition.end_time.utcoffset() == datetime.timedelta(0)


def test_frame_count_survives_counter_reset_between_files(tmp_path):
    first = write_acquisition(tmp_path / "first.h5")
    second = write_acquisition(tmp_path / "second.h5", ctime_offset=60)
    acquisition = diagnostics.RawAcq(
        filenames=np.array([str(first), str(second)]),
        plot_dir=tmp_path,
    )

    assert acquisition.num_frames == 4
    assert acquisition.ctime_frames.tolist() == [
        1_700_000_000.0,
        1_700_000_030.0,
        1_700_000_060.0,
        1_700_000_090.0,
    ]


def test_cli_rejects_incomplete_or_invalid_date_ranges(tmp_path):
    runner = CliRunner()
    incomplete = runner.invoke(
        raw_acq_cli.cli,
        ["plot-summed-spectrum", "--raw-acq-dir", str(tmp_path), "--start-time", "2024-01-01 00:00:00"],
    )
    invalid = runner.invoke(
        raw_acq_cli.cli,
        [
            "plot-summed-spectrum",
            "--raw-acq-dir",
            str(tmp_path),
            "--start-time",
            "not-a-date",
            "--end-time",
            "also-not-a-date",
        ],
    )

    assert incomplete.exit_code != 0
    assert invalid.exit_code != 0
    assert "provide both" in incomplete.output
    assert "times must use" in invalid.output


def test_summed_spectrum_honors_bad_inputs(monkeypatch):
    acquisition = object.__new__(diagnostics.RawAcq)
    acquisition.start_time = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    acquisition.end_time = acquisition.start_time + datetime.timedelta(seconds=30)
    acquisition.plot_dir = "."
    calls = []

    def calc_fft(crate, slot, input_number):
        calls.append((crate, slot, input_number))
        spectrum = np.ones((2, 1024))
        return spectrum, np.ones(1024)

    acquisition.calc_fft = calc_fft
    acquisition.get_timestream_for_input = lambda *coordinates: (
        np.ones((2, 2048)),
        np.array([1_700_000_000.0, 1_700_000_030.0]),
        np.array([100, 200]),
    )
    monkeypatch.setattr(diagnostics.pytz, "timezone", lambda zone: diagnostics.pytz.utc)
    monkeypatch.setattr(diagnostics.plt, "close", lambda *args, **kwargs: None)

    acquisition.plot_total_dynamic_spectrum(
        mask_rfi=False,
        mask_sun=False,
        ds_time_factor=1,
        ds_freq_factor=1,
        save_plot=False,
        bad_inputs=[[0, 0, 0]],
    )

    assert (0, 0, 0) not in calls
    assert (0, 0, 1) in calls
    assert len(calls) == 255


def test_custom_plot_filenames_are_accepted(tmp_path):
    acquisition = object.__new__(diagnostics.RawAcq)
    acquisition.plot_dir = str(tmp_path)
    acquisition.start_time = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
    acquisition.end_time = acquisition.start_time
    acquisition.num_inputs = 0

    acquisition.plot_input_summary_diagnostic(
        inputs=np.empty((0, 3), dtype=int),
        plot_types=[],
        plot_filename="summary",
    )
    acquisition.plot_slot_dynamic_spectrum_summary(
        0,
        0,
        plot_filename="slot-summary",
    )
