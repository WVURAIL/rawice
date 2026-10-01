"""Offline regressions for remaining acquisition and plotting failures."""
import contextlib
import datetime
import io
import os
import subprocess
import sys
from pathlib import Path
import types
import unittest
from unittest import mock

from test_correctness import FixtureTest, make_acquisition, raw, diagnostics, cli, BASE_TIME, ROOT
import h5py
import numpy as np
import pytz
from click.testing import CliRunner


class RemainingCaptureTests(FixtureTest):
    def capture(self, name, counts, times):
        path = self.directory / name
        make_acquisition(path, [10] * len(counts), coordinates=np.zeros((len(counts), 3), dtype=int))
        with h5py.File(path, 'r+') as f:
            values = f['timestamp'][:]
            values['fpga_count'][:, 0] = counts
            values['ctime'][:, 0] = times
            f['timestamp'][:] = values
        stamp = times[0] if len(times) else BASE_TIME
        os.utime(path, (stamp, stamp))
        return path

    def diagnostic(self, paths, **kwargs):
        return diagnostics.RawAcq(filenames=np.array([str(p) for p in paths]),
                                  plot_dir=str(self.directory),
                                  raw_acq_dir=kwargs.get('raw_acq_dir', str(self.directory)))

    def test_raw_counts_actual_frame_boundaries(self):
        for counts, expected in [([100, 100], 1), ([100, 100, 200, 200], 2)]:
            with self.subTest(counts=counts):
                p = self.capture(f'counts{expected}.h5', counts, BASE_TIME + np.arange(len(counts)))
                acq = raw.raw_acq(str(p))
                self.addCleanup(acq.hdf5.close)
                self.assertEqual(acq.num_timestamps, expected)

    def test_diagnostic_counts_frames_in_read_order_after_counter_reset(self):
        p1 = self.capture('first.h5', [200, 200, 300, 300], [BASE_TIME]*2 + [BASE_TIME+30]*2)
        p2 = self.capture('second.h5', [100, 100, 200, 200], [BASE_TIME+60]*2 + [BASE_TIME+90]*2)
        os.utime(p2, (BASE_TIME+60, BASE_TIME+60))
        acq = self.diagnostic([p1, p2])
        self.assertEqual(acq.num_frames, 4)
        np.testing.assert_array_equal(acq.ctime_frames, BASE_TIME + np.array([0, 30, 60, 90]))

    def test_diagnostic_loads_valid_file_after_unreadable_first_file(self):
        bad = self.directory / 'first.h5'; bad.write_text('not HDF5')
        os.utime(bad, (BASE_TIME-1, BASE_TIME-1))
        good = self.capture('second.h5', [100], [BASE_TIME])
        acq = self.diagnostic([bad, good])
        np.testing.assert_array_equal(acq.timestream[:, 0], [10])

    def test_diagnostic_all_unreadable_files_raise_domain_error(self):
        paths = [self.directory / name for name in ('bad1.h5', 'bad2.h5')]
        for path in paths: path.write_text('not HDF5')
        with self.assertRaises(diagnostics.RawAcqException):
            self.diagnostic(paths)

    def test_explicit_paths_do_not_require_unrelated_acquisition_directory(self):
        p = self.capture('explicit.h5', [100], [BASE_TIME])
        acq = self.diagnostic([p], raw_acq_dir=str(self.directory / 'unrelated-missing'))
        self.assertEqual(len(acq.timestream), 1)

    def test_empty_raw_capture_rejects_and_closes_file(self):
        p = self.capture('empty.h5', [], [])
        opened = h5py.File(p, 'r')
        with mock.patch.object(raw.h5py, 'File', return_value=opened):
            with self.assertRaises(ValueError):
                raw.raw_acq(str(p))
        self.assertFalse(opened.id.valid)

    def test_failed_maser_file_does_not_duplicate_previous_capture(self):
        good = self.capture('1.h5', [100], [BASE_TIME])
        bad = self.directory / '2.h5'; bad.write_text('not HDF5')
        helper_class = raw.raw_acq._CheckInput
        def inspect(helper):
            helper.tau = np.array([2.0]); helper.angles = np.array([0.1])
        with mock.patch.object(helper_class, 'inspect_maser', inspect):
            result = raw.analyse_maser(str(self.directory) + '/', [0, 0, 0])
        self.assertEqual(len(result.fpgatime), 1)
        raw.raw_acq._latest_instance.hdf5.close()

    def test_all_failed_maser_files_raise_without_reusing_previous_load(self):
        self.load_raw('previous.h5', [50])
        folder = self.directory / 'bad'; folder.mkdir()
        (folder/'1.h5').write_text('not HDF5')
        helper_class = raw.raw_acq._CheckInput
        def inspect(helper):
            helper.tau = np.array([2.0]); helper.angles = np.array([0.1])
        with mock.patch.object(helper_class, 'inspect_maser', inspect):
            with self.assertRaises(OSError):
                raw.analyse_maser(str(folder) + '/', [0, 0, 0])


class RemainingCurveTests(FixtureTest):
    def test_curve_fit_uses_frame_and_sample_dimensions(self):
        selected = self.input(self.load_raw('curve.h5', [10]), [0, 0, 0])
        for frames, samples in [(1, 2048), (3, 8), (2049, 8)]:
            with self.subTest(frames=frames, samples=samples):
                selected.time_streams = np.ones((frames, samples))
                calls = []
                def fit(objective, x, y, sigma, **kwargs):
                    self.assertEqual(len(x), len(y))
                    self.assertEqual(len(sigma), len(y))
                    calls.append(len(y))
                    return np.array([2., 1., 1. + .0001*len(calls), 0.]), np.eye(4)
                with mock.patch.object(raw, 'curve_fit', side_effect=fit):
                    selected.get_curve_fit()
                self.assertEqual(calls, [samples] * frames)
                self.assertEqual(len(selected.tau_shift), frames)

    def test_real_curve_fit_handles_three_short_sine_wave_frames(self):
        selected = self.input(self.load_raw('real.h5', [10]), [0, 0, 0])
        x = np.arange(1, 129)
        phases = np.array([3.9, 4.0, 4.1])
        selected.time_streams = np.array([raw.objective(x, 20., 1., phase, 0.) for phase in phases])
        selected.get_curve_fit()
        np.testing.assert_allclose(selected.phase, phases, atol=1e-6)
        np.testing.assert_allclose(selected.phase_unwrapped, [0, .1, .2], atol=1e-6)

    def test_fit_diagnostic_uses_per_frame_axis(self):
        selected = self.input(self.load_raw('plot.h5', [10]), [0, 0, 0])
        selected.time_streams = np.ones((3, 8))
        selected.amp = [2.] * 3; selected.freq_stability = [1.] * 3
        selected.phase = [1.] * 3; selected.vert = [0.] * 3
        selected.tau_shift = [0., .1, .2]; selected.tau_err = [.1] * 3
        selected.get_single_curve_fit(0)


class RemainingFileTests(FixtureTest):
    def test_directory_and_wildcard_paths_select_regular_unlocked_files(self):
        first = self.directory / 'adc'; first.write_text('first')
        second = self.directory / '2.h5'; second.write_text('second')
        (self.directory / '3.lock').write_text('locked')
        (self.directory / '4.h5').mkdir()
        stamp = {str(first): 1, str(second): 2}
        with mock.patch.object(raw.os.path, 'getmtime', side_effect=lambda p: stamp.get(p, 99)):
            self.assertEqual(raw.get_newest_file(str(self.directory)), str(second))
            self.assertEqual(raw.get_second_newest_file(str(self.directory) + '/'), str(first))
            self.assertEqual(raw.get_newest_file(str(self.directory / '*')), str(second))

    def test_empty_progressbar_and_directory(self):
        self.assertEqual(list(raw.progressbar([], out=io.StringIO())), [])
        with self.assertRaises(FileNotFoundError):
            raw.get_newest_file(str(self.directory))
        with self.assertRaises(FileNotFoundError):
            raw.get_second_newest_file(str(self.directory))
        with self.assertRaises(FileNotFoundError):
            raw.analyse_maser(str(self.directory), [0, 0, 0])


class RemainingDiagnosticTests(FixtureTest):
    def test_bad_input_argument_controls_spectrum_selection(self):
        acq = object.__new__(diagnostics.RawAcq)
        class StopAfterSelection(Exception): pass
        for bad, expected in [([[0, 0, 0]], 255), ([], 256), (diagnostics.BAD_INPUTS, 256-len({tuple(x) for x in diagnostics.BAD_INPUTS}))]:
            with self.subTest(bad=bad):
                calls = []
                def fft(*coords):
                    calls.append(coords); return np.ones((2, 1024)), None
                acq.calc_fft = fft
                acq.get_timestream_for_input = lambda *coords: (None, np.array([BASE_TIME, BASE_TIME+30]), None)
                original_min = np.min
                def stop_after_selection(values, *args, **kwargs):
                    if isinstance(values, list) and values and all(isinstance(v, int) for v in values):
                        raise StopAfterSelection
                    return original_min(values, *args, **kwargs)
                with mock.patch.object(diagnostics.np, 'min', side_effect=stop_after_selection):
                    with self.assertRaises(StopAfterSelection):
                        acq.plot_total_dynamic_spectrum(bad_inputs=bad)
                self.assertEqual(len(calls), expected)
                self.assertTrue(all(tuple(x) not in calls for x in bad))

    def test_all_masked_inputs_raise_domain_error(self):
        acq = object.__new__(diagnostics.RawAcq)
        acq.calc_fft = mock.Mock(side_effect=AssertionError('masked input processed'))
        with self.assertRaises(diagnostics.RawAcqException):
            acq.plot_total_dynamic_spectrum(bad_inputs=[[0, s, i] for s in range(16) for i in range(16)])
        acq.calc_fft.assert_not_called()

    def test_custom_plot_filenames(self):
        acq = object.__new__(diagnostics.RawAcq)
        acq.plot_dir = str(self.directory); acq.num_inputs = 0
        acq.start_time = acq.end_time = datetime.datetime.fromtimestamp(BASE_TIME, tz=pytz.utc)
        with mock.patch.object(diagnostics, 'PdfPages') as pdf:
            acq.plot_input_summary_diagnostic(inputs=np.empty((0, 3), dtype=int), plot_types=[], plot_filename='summary')
            self.assertEqual(pdf.call_args.args[0], str(self.directory/'summary.pdf'))
            acq.plot_slot_dynamic_spectrum_summary(0, 0, plot_filename='slot')
            self.assertEqual(pdf.call_args.args[0], str(self.directory/'slot.pdf'))

    def test_cli_package_import_binds_the_diagnostics_module(self):
        code = """import sys, types
sys.modules['ephem'] = types.ModuleType('ephem')
sys.modules['chime_frb_constants'] = types.ModuleType('chime_frb_constants')
from raw_acq_diagnostics import raw_acq_cli
assert callable(raw_acq_cli.rad.RawAcq)
"""
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_rejects_partial_and_invalid_dates_without_loading(self):
        with mock.patch.object(cli.pytz, 'timezone', return_value=pytz.utc), mock.patch.object(cli.rad, 'RawAcq') as loader:
            for args in [['--start-time', '2024-01-01 00:00:00'], ['--end-time', '2024-01-01 00:00:00'], ['--start-time', 'invalid', '--end-time', 'invalid']]:
                with self.subTest(args=args):
                    result = CliRunner().invoke(cli.cli, ['plot-summed-spectrum', '--raw-acq-dir', str(self.directory), *args])
                    self.assertEqual(result.exit_code, 2, result.output)
            loader.assert_not_called()

    def test_maintenance_custom_filename_and_bad_input_argument(self):
        for bad, expected in [([[0, 0, 0]], 255), ([], 256)]:
            with self.subTest(bad=bad):
                calls = []
                def read(*coords):
                    calls.append(coords)
                    return np.ones((1, 8)), np.array([BASE_TIME]), np.array([100])
                acq = types.SimpleNamespace(get_timestream_for_input=read)
                with mock.patch.object(diagnostics.pytz, 'timezone', return_value=pytz.utc), \
                     mock.patch.object(diagnostics, 'RawAcq', return_value=acq), \
                     mock.patch.object(diagnostics, 'identify_gbo_maintenance_days', return_value=([], [])), \
                     mock.patch.object(diagnostics, 'PdfPages') as pdf:
                    diagnostics.plot_maintenance_vs_nonmaintenance_timeseries(
                        '2024-01-01 00:00:00', '2024-01-02 00:00:00', plot_types=[],
                        plot_filename='maintenance', bad_inputs=bad, plot_dir=str(self.directory))
                self.assertEqual(pdf.call_args.args[0], str(self.directory/'maintenance.pdf'))
                self.assertEqual(len(calls), expected)
                self.assertTrue(all(tuple(x) not in calls for x in bad))

    def test_maintenance_honors_mask_with_default_filename(self):
        calls = []
        def read(*coords):
            calls.append(coords)
            return np.ones((1, 8)), np.array([BASE_TIME]), np.array([100])
        acq = types.SimpleNamespace(get_timestream_for_input=read)
        with mock.patch.object(diagnostics.pytz, 'timezone', return_value=pytz.utc), \
             mock.patch.object(diagnostics, 'RawAcq', return_value=acq), \
             mock.patch.object(diagnostics, 'identify_gbo_maintenance_days', return_value=([], [])), \
             mock.patch.object(diagnostics, 'PdfPages'):
            diagnostics.plot_maintenance_vs_nonmaintenance_timeseries(
                '2024-01-01 00:00:00', '2024-01-02 00:00:00', plot_types=[],
                bad_inputs=[[0, 0, 0]], plot_dir=str(self.directory))
        self.assertEqual(len(calls), 255)
        self.assertNotIn((0, 0, 0), calls)

    def test_main_selects_the_existing_single_crate(self):
        acq = mock.Mock()
        with mock.patch.dict(os.environ, {'RAW_ACQ_DIR': str(self.directory)}), \
             mock.patch.object(diagnostics.pytz, 'timezone', return_value=pytz.utc), \
             mock.patch.object(diagnostics, 'RawAcq', return_value=acq):
            diagnostics.main()
        self.assertEqual([call.args for call in acq.plot_slot_dynamic_spectrum_summary.call_args_list],
                         [(0, slot) for slot in range(16)])

    def test_email_requires_both_credentials(self):
        acq = mock.Mock()
        acq.start_time = acq.end_time = datetime.datetime.fromtimestamp(BASE_TIME, tz=pytz.utc)
        acq.plot_dir = str(self.directory)
        for username, password in [(None, 'example'), ('example', None)]:
            with self.subTest(username_present=username is not None, password_present=password is not None):
                with mock.patch.object(cli.pytz, 'timezone', return_value=pytz.utc), \
                     mock.patch.object(cli.rad, 'RawAcq', return_value=acq), \
                     mock.patch.object(cli, 'username', username), \
                     mock.patch.object(cli, 'app_password', password), \
                     mock.patch.object(cli, 'send_email') as send:
                    cli.plot_summed_spectrum.callback('', '', False, False, 3, 1, 'gbo', True,
                                                      str(self.directory), str(self.directory), ('recipient@example.invalid',))
                send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
