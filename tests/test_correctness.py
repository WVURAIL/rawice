"""Offline regression tests for acquisition selection and time-range correctness."""
import ast
import contextlib
import datetime
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import dateutil.parser
import h5py
import numpy as np
import pytz

sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(os.environ.get("RAWICE_TEST_SOURCE_DIR", Path(__file__).resolve().parents[1]))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


raw = load_module("rawice_under_test", "rawice.py")
# These optional dependencies are used only by unrelated sky/plot routines.
# The acquisition classes themselves are imported and executed without changes.
with mock.patch.dict(sys.modules, {"ephem": types.ModuleType("ephem"),
                                   "chime_frb_constants": types.ModuleType("chime_frb_constants")}):
    diagnostics = load_module("diagnostics_under_test", "raw_acq_diagnostics/raw_acq_diagnostics.py")
with mock.patch.dict(sys.modules, {"raw_acq_diagnostics": diagnostics}):
    cli = load_module("cli_under_test", "raw_acq_diagnostics/raw_acq_cli.py")

BASE_TIME = 1664640000.0


def make_acquisition(path, values, coordinates=None, start=BASE_TIME):
    n = len(values)
    coords = np.array(coordinates if coordinates is not None else [[0, 0, 0]] * n,
                      dtype=np.uint8)
    timestamp = np.zeros((n, 1), dtype=[("ctime", "<f8"), ("fpga_count", "<u8")])
    timestamp["ctime"][:, 0] = start + np.arange(n)
    timestamp["fpga_count"][:, 0] = int(start) + np.arange(n)
    with h5py.File(path, "w") as f:
        f.create_group("index_map").create_dataset("timestream", data=np.arange(2048))
        for i, name in enumerate(("crate", "slot", "adc_input")):
            f.create_dataset(name, data=coords[:, i, None])
        f.create_dataset("timestamp", data=timestamp)
        f.create_dataset("timestream", data=np.repeat(np.array(values, dtype=np.int8)[:, None], 2048, axis=1))
    os.utime(path, (start, start))


class FixtureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rawice-test-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.addCleanup(raw.plt.close, "all")

    def load_raw(self, name, values, coordinates=None):
        path = self.directory / name
        make_acquisition(path, values, coordinates)
        with contextlib.redirect_stdout(io.StringIO()):
            acquisition = raw.raw_acq(str(path))
        self.addCleanup(acquisition.hdf5.close)
        return acquisition

    def input(self, acquisition, coordinates):
        with contextlib.redirect_stdout(io.StringIO()):
            return acquisition.check_input(coordinates)


class InputSelectionTests(FixtureTest):
    def test_raw_timestamp_metadata_is_utc(self):
        acquisition = self.load_raw("times.h5", [10, 10])
        self.assertEqual(acquisition.start_time, "2022-10-01T16:00:00+00:00")
        self.assertEqual(acquisition.end_time, "2022-10-01T16:00:01+00:00")

    def test_all_three_input_coordinates_are_applied(self):
        acquisition = self.load_raw("select.h5", [10, 20, 30, 40],
                                    [[0, 0, 2], [0, 1, 2], [1, 0, 2], [0, 0, 3]])
        for coordinates, expected in [([0, 0, 2], 10), ([0, 1, 2], 20),
                                      ([1, 0, 2], 30), ([0, 0, 3], 40)]:
            with self.subTest(coordinates=coordinates):
                result = self.input(acquisition, coordinates)
                np.testing.assert_array_equal(result.time_streams[:, 0], [expected])
                np.testing.assert_array_equal(result.rms, [expected])

    def test_new_load_does_not_replace_existing_acquisition_or_helper(self):
        first = self.load_raw("first.h5", [10, 10])
        existing_helper = self.input(first, [0, 0, 0])
        second = self.load_raw("second.h5", [50, 50])
        np.testing.assert_array_equal(first.timestream[:, 0], [10, 10])
        np.testing.assert_array_equal(self.input(first, [0, 0, 0]).time_streams[:, 0], [10, 10])
        existing_helper.get_timestream_for_input()
        np.testing.assert_array_equal(existing_helper.time_streams[:, 0], [10, 10])
        np.testing.assert_array_equal(self.input(second, [0, 0, 0]).time_streams[:, 0], [50, 50])

    def test_legacy_class_calls_and_array_access_use_latest_acquisition(self):
        self.load_raw("first.h5", [10, 10])
        self.load_raw("latest.h5", [50, 50])
        np.testing.assert_array_equal(raw.raw_acq.timestream[:, 0], [50, 50])
        np.testing.assert_array_equal(self.input(raw.raw_acq, [0, 0, 0]).time_streams[:, 0], [50, 50])

    def test_iceboard_uses_owner_crate_slot_and_input(self):
        coordinates = [[crate, slot, inp] for crate, slot in [(0, 0), (0, 1), (1, 0)] for inp in range(16)]
        first = self.load_raw("boards.h5", [10]*16 + [20]*16 + [30]*16, coordinates)
        self.load_raw("other.h5", [90]*16, [[0, 1, inp] for inp in range(16)])
        original_histogram = np.histogram
        observed = []

        def histogram(values, *args, **kwargs):
            observed.append(np.unique(values).tolist())
            return original_histogram(values, *args, **kwargs)

        with contextlib.ExitStack() as stack:
            for name in ("figure", "suptitle", "subplot", "plot", "title", "tight_layout", "show"):
                stack.enter_context(mock.patch.object(raw.plt, name))
            stack.enter_context(mock.patch.object(raw.np, "histogram", side_effect=histogram))
            first.check_iceboard(0, 1)
            self.assertEqual(observed, [[20]] * 16)
            observed.clear()
            raw.raw_acq.check_iceboard(0, 1)
            self.assertEqual(observed, [[90]] * 16)


class DiagnosticLoadingTests(FixtureTest):
    def load_files(self, names):
        return diagnostics.RawAcq(filenames=np.array([str(p) for p in names]),
                                  plot_dir=str(self.directory), raw_acq_dir=str(self.directory))

    def run_directory(self):
        folder = self.directory / "2022-10-01T00:00:00Z_gbo_rawadc"
        folder.mkdir()
        return folder

    def load_dates(self, start, end):
        dates = np.array([datetime.datetime.fromtimestamp(start, tz=pytz.utc),
                          datetime.datetime.fromtimestamp(end, tz=pytz.utc)])
        return diagnostics.RawAcq(dates=dates, plot_dir=str(self.directory),
                                  raw_acq_dir=str(self.directory))

    def test_explicit_filenames_load_all_frames_without_date_bounds(self):
        paths = [self.directory / "first.h5", self.directory / "second.h5"]
        make_acquisition(paths[0], [10, 11])
        make_acquisition(paths[1], [20, 21], start=BASE_TIME + 3600)
        acquisition = self.load_files(paths)
        np.testing.assert_array_equal(acquisition.timestream[:, 0], [10, 11, 20, 21])
        self.assertEqual(acquisition.start_time, datetime.datetime.fromtimestamp(BASE_TIME, tz=pytz.utc))
        self.assertEqual(acquisition.end_time, datetime.datetime.fromtimestamp(BASE_TIME + 3601, tz=pytz.utc))

    def test_first_middle_and_last_queries_use_only_valid_neighbors(self):
        folder = self.run_directory()
        paths = [folder / f"{i}.h5" for i in range(3)]
        for i, path in enumerate(paths):
            make_acquisition(path, [10+i, 10+i], start=BASE_TIME + 3600*i)
        original_open = h5py.File
        for index, expected_opened in [(0, [0, 1]), (1, [0, 1, 2]), (2, [1, 2])]:
            with self.subTest(index=index):
                opened = []

                def open_file(path, *args, **kwargs):
                    opened.append(Path(path).name)
                    return original_open(path, *args, **kwargs)

                with mock.patch.object(diagnostics.h5py, "File", side_effect=open_file):
                    acquisition = self.load_dates(BASE_TIME + 3600*index - 1, BASE_TIME + 3600*index + 2)
                self.assertEqual(opened, [f"{i}.h5" for i in expected_opened])
                np.testing.assert_array_equal(acquisition.timestream[:, 0], [10+index, 10+index])

    def test_single_file_date_query(self):
        folder = self.run_directory()
        make_acquisition(folder / "only.h5", [10, 11])
        acquisition = self.load_dates(BASE_TIME - 1, BASE_TIME + 2)
        np.testing.assert_array_equal(acquisition.timestream[:, 0], [10, 11])

    def test_empty_explicit_file_list_has_domain_error(self):
        with self.assertRaisesRegex(diagnostics.RawAcqException, "No acquisition files"):
            self.load_files([])

    def test_empty_run_search_has_domain_error(self):
        with self.assertRaisesRegex(diagnostics.RawAcqException, "No acquisition runs"):
            self.load_dates(BASE_TIME - 1, BASE_TIME + 2)

    def test_run_without_files_has_domain_error(self):
        self.run_directory()
        with self.assertRaisesRegex(diagnostics.RawAcqException, "No acquisition files"):
            self.load_dates(BASE_TIME - 1, BASE_TIME + 2)

    def test_no_file_matches_even_with_existing_one_hour_margin(self):
        folder = self.run_directory()
        make_acquisition(folder / "old.h5", [10, 11])
        with self.assertRaisesRegex(diagnostics.RawAcqException, "No acquisition files overlap"):
            self.load_dates(BASE_TIME + 86400, BASE_TIME + 86402)

    def test_no_matching_frames_has_domain_error(self):
        folder = self.run_directory()
        make_acquisition(folder / "oldframes.h5", [10, 11], start=BASE_TIME - 86400)
        os.utime(folder / "oldframes.h5", (BASE_TIME, BASE_TIME))
        with self.assertRaisesRegex(diagnostics.RawAcqException, "No acquisition frames overlap"):
            self.load_dates(BASE_TIME - 1, BASE_TIME + 2)

    def test_frame_filter_and_metadata_are_utc_in_any_local_timezone(self):
        folder = self.run_directory()
        make_acquisition(folder / "utc.h5", [10, 11])
        acquisition = self.load_dates(BASE_TIME - 1, BASE_TIME + 2)
        self.assertEqual(acquisition.start_time.isoformat(), "2022-10-01T16:00:00+00:00")
        self.assertEqual(acquisition.end_time.isoformat(), "2022-10-01T16:00:01+00:00")
        np.testing.assert_array_equal(acquisition.ctimes, [BASE_TIME, BASE_TIME+1])

    def test_plot_timestamp_expressions_are_also_utc(self):
        tree = ast.parse((ROOT / "raw_acq_diagnostics/raw_acq_diagnostics.py").read_text())
        targets = {"ctimes_all_averaged", "ctimes_dates"}
        for target in targets:
            with self.subTest(target=target):
                assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                                  and any(isinstance(t, ast.Name) and t.id == target for t in node.targets))
                namespace = dict(np=np, datetime=datetime, pytz=pytz,
                                 ctimes_all=np.array([[BASE_TIME]]), ctimes=np.array([BASE_TIME]))
                exec(compile(ast.Module(body=[assignment], type_ignores=[]), "plot timestamp expression", "exec"), namespace)
                self.assertEqual(namespace[target][0].isoformat(), "2022-10-01T16:00:00+00:00")

    def test_cli_default_window_is_last_24_hours_in_utc(self):
        class Capture:
            def __init__(self, dates, plot_dir):
                self.start_time, self.end_time = dates
                self.plot_dir = plot_dir
                self.dates = dates

            def plot_total_dynamic_spectrum(self, **kwargs):
                pass

        captured = []

        def capture(**kwargs):
            item = Capture(**kwargs)
            captured.append(item)
            return item

        before = datetime.datetime.now(tz=pytz.utc) - datetime.timedelta(minutes=5)
        # The empty-date branch never uses the Eastern timezone object.
        # Avoid depending on optional system timezone aliases for this test.
        with mock.patch.object(cli.pytz, "timezone", return_value=pytz.utc), \
                mock.patch.object(cli.rad, "RawAcq", side_effect=capture):
            cli.plot_summed_spectrum.callback("", "", False, False, 3, 1, "gbo", True,
                                              str(self.directory), ())
        after = datetime.datetime.now(tz=pytz.utc) - datetime.timedelta(minutes=5)
        start, end = captured[0].dates
        self.assertLessEqual(before, end)
        self.assertLessEqual(end, after)
        self.assertEqual(end-start, datetime.timedelta(hours=24))


if __name__ == "__main__":
    unittest.main()
