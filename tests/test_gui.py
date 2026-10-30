"""Smoke tests of the Qt GUI (offscreen); dialogs are replaced by stubs."""

import numpy as np
import pytest

pytest.importorskip('PyQt5')

from PyQt5 import QtWidgets  # noqa: E402

from spectral_cube_viewer import gui  # noqa: E402
from spectral_cube_viewer.fitting import MODEL_HFS_FIXED, MODEL_LABELS  # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def viewer(app, monkeypatch):
    messages = []
    for name in ('information', 'warning', 'critical'):
        monkeypatch.setattr(gui.QMessageBox, name,
                            staticmethod(lambda parent, title, text, _n=name: messages.append((_n, title, text))))
    win = gui.SpectralCubeViewer()
    win.messages = messages
    yield win
    win.close()


@pytest.mark.parametrize('in_ram', [True, False])
def test_load_and_moment0(viewer, cube_file, monkeypatch, in_ram):
    path, data, vel = cube_file
    monkeypatch.setattr(gui, 'ram_budget', lambda: 1e12 if in_ram else 0)
    viewer.load_cube(str(path))
    assert viewer.data is not None and viewer.messages == []
    assert isinstance(viewer.data, np.ndarray) == in_ram
    viewer.vmin_edit.setText('0')
    viewer.vmax_edit.setText('10')
    viewer.apply_spectral_slab()
    assert viewer.mom0.shape == data.shape[1:]
    assert np.isnan(viewer.mom0[:, 0]).all() and np.nanmax(viewer.mom0) > 5
    assert viewer.hfs_combo.currentText() == 'HCN 1-0'      # picked from RESTFRQ


def test_spectrum_baseline_and_fit(viewer, cube_file):
    path, _, _ = cube_file
    viewer.load_cube(str(path))
    viewer.set_spectrum(('pixel', 9, 6))
    assert viewer.spec['y'].shape == viewer.vel.shape
    viewer.win_edit.setText('-19:-5, 12:19')
    viewer.fit_spectrum_baseline()
    assert viewer.baseline is not None and viewer.baseline_rms < 0.1
    viewer.fit_lo_edit.setText('-3')
    viewer.fit_hi_edit.setText('13')
    viewer.ncomp_spin.setValue(1)
    viewer.auto_guess_clicked()
    viewer.fit_clicked()
    assert viewer.fit is not None, viewer.result_text.toPlainText()
    t, v, fwhm = viewer.fit.popt
    assert abs(v - 5.0) < 0.1 and abs(fwhm - 2.0) < 0.2 and abs(t - 3.0) < 0.2
    viewer.model_combo.setCurrentIndex(list(MODEL_LABELS).index(MODEL_HFS_FIXED))   # model change clears fit
    assert viewer.fit is None


def test_polygons_roundtrip(viewer, cube_file, tmp_path, monkeypatch):
    path, _, _ = cube_file
    viewer.load_cube(str(path))
    for verts in ([[1, 1], [5, 1], [5, 5], [1, 5]], [[6, 6], [9, 6], [8, 10]]):
        viewer.start_polygon()
        viewer._finish_polygon(np.array(verts, float))
    assert len(viewer.polygons) == 2
    viewer.plot_selected_average()
    assert 'Combined average of 2' in viewer.spec['title']
    poly = tmp_path / 'regions.poly'
    monkeypatch.setattr(gui.QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(poly), '')))
    monkeypatch.setattr(gui.QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(poly), '')))
    viewer.save_polygons()
    viewer.reset_polygons()
    viewer.load_polygons()
    assert len(viewer.polygons) == 2


def test_cube_baseline_and_save(viewer, cube_file, tmp_path, monkeypatch):
    from astropy.io import fits
    path, _, _ = cube_file
    viewer.load_cube(str(path))
    viewer.win_edit.setText('-19:-5, 12:19')
    viewer.apply_cube_baseline()
    assert viewer.cube_baseline is not None and len(viewer.vel) < len(viewer.vel_orig)
    out = tmp_path / 'blcorr.fits'
    monkeypatch.setattr(gui.QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), '')))
    viewer.save_cube()
    with fits.open(out) as h:
        assert h[0].data.shape[0] == len(viewer.vel) and 'BSCALE' not in h[0].header
    viewer.revert_cube()
    assert viewer.data is viewer.data_orig


def test_offline_line_markers(viewer, cube_file):
    path, _, _ = cube_file
    viewer.load_cube(str(path))
    viewer.set_spectrum(('pixel', 9, 6))
    viewer.vsys_edit.setText('5')
    viewer.marks_combo.setCurrentIndex(1)                   # offline catalog
    labels = [t.get_text() for t in viewer.ax_spec.texts]
    assert any(label.startswith('HCN') for label in labels)
    hcn = next(t for t in viewer.ax_spec.texts if t.get_text().startswith('HCN'))
    assert abs(hcn.get_position()[0] - 5.0) < 1.0           # unsplit J=1-0 is 0.83 km/s from F=2-1


def test_scaled_integer_cube(viewer, scaled_cube_file):
    path, expected, _ = scaled_cube_file
    viewer.load_cube(str(path))
    assert viewer.messages == []
    np.testing.assert_allclose(np.asarray(viewer.data[:]), expected, atol=1e-4, equal_nan=True)
