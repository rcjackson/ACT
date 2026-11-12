import numpy as np
import pandas as pd
import xarray as xr

import act


def test_doppler_lidar_winds():
    # Process a single file
    dl_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_DLPPI)
    result = act.retrievals.doppler_lidar.compute_winds_from_ppi(dl_ds, intensity_name='intensity')
    assert np.round(np.nansum(result['wind_speed'].values)).astype(int) == 1570
    assert np.round(np.nansum(result['wind_direction'].values)).astype(int) == 32635
    assert result['wind_speed'].attrs['units'] == 'm/s'
    assert result['wind_direction'].attrs['units'] == 'degree'
    assert result['height'].attrs['units'] == 'm'
    dl_ds.close()

    # Process multiple files
    dl_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_DLPPI_MULTI)
    del dl_ds['range'].attrs['units']
    result = act.retrievals.doppler_lidar.compute_winds_from_ppi(dl_ds)
    assert np.round(np.nansum(result['wind_speed'].values)).astype(int) == 2854
    assert np.round(np.nansum(result['wind_direction'].values)).astype(int) == 64986
    dl_ds.close()


def test_doppler_lidar_tucker_retrieval():
    dl_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_DLFPT)
    dl_ds = act.retrievals.calculate_tucker_method_pbl(dl_ds, interval="10min")
    assert np.allclose(dl_ds['pbl_tucker'].values, [675.0, 795.0, 735.0, 825.0, 795.0, 735.0])
    assert np.allclose(dl_ds['tucker_atmospheric_variance'].values.max(), 19.012731495487955)
    assert np.allclose(dl_ds['tucker_noise_variance'].values.max(), 145.83166344062627)
    dl_ds.close()


def test_doppler_lidar_tucker_uses_above_threshold_gate(monkeypatch):
    """The retrieval should select a turbulent gate, not the first quiet gate."""
    ds = xr.Dataset(
        {
            "radial_velocity": (
                ("time", "range"),
                [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, -1.0, -1.0], [0.0, 1.0, 1.0]],
            )
        },
        coords={
            "time": pd.date_range("2026-01-01", periods=4, freq="min"),
            "range": [100.0, 200.0, 300.0],
        },
    )

    # Make the atmospheric/noise split deterministic: the variable gates have
    # atmospheric variance above threshold, while the quiet surface gate does
    # not.  The fixed-threshold implementation must therefore return 200 m.
    monkeypatch.setattr(
        act.retrievals.pbl_lidar,
        "acf",
        lambda series, nlags, fft: np.array([1.0, 0.9, 0.0]),
    )
    result = act.retrievals.calculate_tucker_method_pbl(
        ds, interval="10min", threshold=0.5, min_gate_height=50.0
    )

    np.testing.assert_allclose(result["pbl_tucker"].values, [200.0])
