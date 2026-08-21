import os

import h5py
import numpy as np
import pytest

import rawice


def write_acquisition(path, value_offset=0, ctime_offset=0):
    records = [
        (0, 1, 2, 100, 1_700_000_000.0, 11 + value_offset),
        (0, 1, 3, 100, 1_700_000_000.0, 12 + value_offset),
        (1, 0, 2, 100, 1_700_000_000.0, 13 + value_offset),
        (0, 1, 2, 200, 1_700_000_030.0, 21 + value_offset),
    ]
    timestamp_dtype = np.dtype([("fpga_count", "<i8"), ("ctime", "<f8")])
    timestamps = np.zeros((len(records), 1), dtype=timestamp_dtype)
    timestamps[:, 0]["fpga_count"] = [record[3] for record in records]
    timestamps[:, 0]["ctime"] = [record[4] + ctime_offset for record in records]

    with h5py.File(path, "w") as output:
        index_map = output.create_group("index_map")
        index_map.create_dataset("timestream", data=np.arange(len(records)))
        output.create_dataset("crate", data=np.array([[r[0]] for r in records]))
        output.create_dataset("slot", data=np.array([[r[1]] for r in records]))
        output.create_dataset("adc_input", data=np.array([[r[2]] for r in records]))
        output.create_dataset("timestamp", data=timestamps)
        output.create_dataset(
            "timestream",
            data=np.array([
                np.full(2048, record[5], dtype=np.int16)
                for record in records
            ]),
        )
        output.attrs["archive_version"] = np.bytes_("test")
        output.attrs["collection_server"] = np.bytes_("test")
        output.attrs["git_version_tag"] = np.bytes_("test")
        output.attrs["file_name"] = str(path)
        output.attrs["data_type"] = np.bytes_("raw")
        output.attrs["system_user"] = np.bytes_("pytest")
        output.attrs["rawadc_version"] = 1
        output.attrs["timestamping_warning"] = np.bytes_("")
    return path


def test_instance_state_and_legacy_class_access(tmp_path):
    first = rawice.raw_acq(write_acquisition(tmp_path / "first.h5"))
    first_values = first.timestream.copy()
    second = rawice.raw_acq(write_acquisition(tmp_path / "second.h5", 100))

    np.testing.assert_array_equal(first.timestream, first_values)
    np.testing.assert_array_equal(rawice.raw_acq.timestream, second.timestream)

    first_input = first.check_input([0, 1, 2])
    latest_input = rawice.raw_acq.check_input([0, 1, 2])
    np.testing.assert_array_equal(first_input.time_streams[:, 0], [11, 21])
    np.testing.assert_array_equal(latest_input.time_streams[:, 0], [111, 121])

    np.testing.assert_array_equal(first.adc_record_ctime, [1_700_000_000.0, 1_700_000_030.0])
    assert first.num_timestamps == 2
    assert first.start_time.endswith("+00:00")
    assert first.end_time.endswith("+00:00")
    first.hdf5.close()
    second.hdf5.close()


def test_input_coordinates_are_crate_slot_input(tmp_path):
    acquisition = rawice.raw_acq(write_acquisition(tmp_path / "coordinates.h5"))

    crate_zero_slot_one = acquisition.check_input([0, 1, 2])
    crate_one_slot_zero = acquisition.check_input([1, 0, 2])

    np.testing.assert_array_equal(crate_zero_slot_one.time_streams[:, 0], [11, 21])
    np.testing.assert_array_equal(crate_one_slot_zero.time_streams[:, 0], [13])
    assert acquisition._input_mask(0, 1, 2).tolist() == [True, False, False, True]
    acquisition.hdf5.close()


def test_curve_fit_uses_available_frame_count(monkeypatch, tmp_path):
    acquisition = rawice.raw_acq(write_acquisition(tmp_path / "curve.h5"))
    selected = acquisition.check_input([0, 1, 2])
    selected.time_streams = np.ones((3, 8))
    calls = []

    def fake_curve_fit(*args, **kwargs):
        phase = 1.0 + 0.1 * len(calls)
        calls.append(phase)
        return np.array([2.0, 1.0, phase, 0.0]), np.eye(4)

    monkeypatch.setattr(rawice, "curve_fit", fake_curve_fit)
    selected.get_curve_fit()

    assert len(calls) == 3
    assert len(selected.phase) == 3
    np.testing.assert_allclose(selected.phase_unwrapped, [0.0, 0.1, 0.2])
    acquisition.hdf5.close()


def test_file_helpers_ignore_lock_files_and_handle_empty_directories(tmp_path):
    lock_file = tmp_path / "capture.lock"
    lock_file.write_text("locked")

    with pytest.raises(FileNotFoundError):
        rawice.get_newest_file(tmp_path)
    with pytest.raises(FileNotFoundError):
        rawice.analyse_maser(tmp_path, [0, 0, 0])

    first = tmp_path / "000001"
    second = tmp_path / "000002"
    first.write_text("first")
    second.write_text("second")
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))

    assert rawice.get_newest_file(tmp_path) == str(second)
    assert rawice.get_second_newest_file(tmp_path) == str(first)

    nested = tmp_path / "run" / "raw_acq"
    nested.mkdir(parents=True)
    wildcard_file = nested / "000003"
    wildcard_file.write_text("third")
    pattern = str(tmp_path / "*" / "raw_acq" / "*")
    assert rawice.get_newest_file(pattern) == str(wildcard_file)


def test_empty_acquisition_has_a_clear_error(tmp_path):
    filename = tmp_path / "empty.h5"
    timestamp_dtype = np.dtype([("fpga_count", "<i8"), ("ctime", "<f8")])
    with h5py.File(filename, "w") as output:
        index_map = output.create_group("index_map")
        index_map.create_dataset("timestream", data=np.empty(0, dtype=int))
        output.create_dataset("crate", data=np.empty((0, 1), dtype=int))
        output.create_dataset("slot", data=np.empty((0, 1), dtype=int))
        output.create_dataset("adc_input", data=np.empty((0, 1), dtype=int))
        output.create_dataset("timestamp", data=np.empty((0, 1), dtype=timestamp_dtype))
        output.create_dataset("timestream", data=np.empty((0, 2048), dtype=np.int16))

    with pytest.raises(ValueError, match="contains no frames"):
        rawice.raw_acq(filename)
