"""Qt GUI of Spectral Cube Viewer."""

import argparse
import os
import re
import sys
import traceback
import warnings
from contextlib import contextmanager

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from astropy import units as u
from astropy.io import fits

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from matplotlib.widgets import PolygonSelector, SpanSelector  # noqa: E402
from scipy import interpolate  # noqa: E402

from . import __version__  # noqa: E402
from .core import (Cancelled, CHUNK_BYTES, baseline_cube, chunk_ranges, fit_baseline, format_windows,  # noqa: E402
                   merge_windows, moment0, open_cube, parse_windows, ram_budget, region_mean_spectrum,
                   windows_mask)
from .fitting import (MODEL_GAUSS, MODEL_LABELS, LineModel, auto_guess, fit_lines)  # noqa: E402
from .lines import (C_KMS, CATALOG_ERROR, CATALOG_PATH, HFS_PRESETS, catalog_lines, find_preset,  # noqa: E402
                    preset_hfs, preset_reference, splatalogue_lines)


class _Worker(QtCore.QObject):
    """Runs fn(*args) in a QThread; emits finished(result, error message or None)."""
    finished = QtCore.pyqtSignal(object, object)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    def run(self):
        try:
            result, error = self.fn(*self.args), None
        except Exception as e:
            result, error = None, f'{type(e).__name__}: {e}'
        self.finished.emit(result, error)


def _toolbar_active(toolbar):
    """True while the zoom or pan tool of a navigation toolbar is selected."""
    mode = getattr(toolbar, 'mode', '')
    return bool(getattr(mode, 'value', mode))


class SpectralCubeViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spectral Cube Viewer')
        self.resize(1500, 900)

        # Cube state
        self.path = None
        self._hdul = None           # kept open while the data are memory-mapped
        self.header = None          # 3-D header (spectral axis last in FITS order)
        self.unit = u.K
        self.data_orig = None       # (nchan, ny, nx), in RAM or memory-mapped
        self.vel_orig = None        # km/s
        self.data = None            # working cube: original or baseline-corrected sub-cube
        self.vel = None
        self.dv = None              # channel widths [km/s]
        self.ch_offset = 0          # index of self.data[0] in the original cube
        self.cube_baseline = None   # dict(windows, order, rms) once a cube baseline is applied
        self.wcs2d = None
        self.l0 = self.b0 = None
        self.rest_freq_mhz = None
        self.mom0 = None
        self.mom0_range = None

        # Line markers
        self.line_marks = []        # dicts with species, transition, freq_mhz
        self._marks_thread = None
        self._marks_worker = None

        # Regions / spectrum state
        self.polygons = []          # list of (N, 2) vertex arrays in pixel coordinates
        self.poly_selector = None
        self.span_selector = None
        self.spec = None            # dict(source, y, title)
        self.baseline = None
        self.baseline_rms = None
        self.fit = None
        self._pix_marker = None
        self._spec_drawn = False

        self._build_ui()

    # ------------------------------------------------------------------ UI construction
    @staticmethod
    def _button(text, slot, layout, checkable=False):
        btn = QtWidgets.QPushButton(text)
        btn.setCheckable(checkable)
        btn.clicked.connect(slot)
        layout.addWidget(btn)
        return btn

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Top controls
        ctrl_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl_layout)
        self.open_btn = self._button('Open FITS...', self.open_file, ctrl_layout)
        ctrl_layout.addWidget(QtWidgets.QLabel('v min (km/s)'))
        self.vmin_edit = QtWidgets.QLineEdit('0.0')
        ctrl_layout.addWidget(self.vmin_edit)
        ctrl_layout.addWidget(QtWidgets.QLabel('v max (km/s)'))
        self.vmax_edit = QtWidgets.QLineEdit('10.0')
        ctrl_layout.addWidget(self.vmax_edit)
        self.apply_btn = self._button('Apply slab && show Moment0', self.apply_spectral_slab, ctrl_layout)
        self.avg_whole_btn = self._button('Average spectrum (whole cube)', self.plot_whole_average, ctrl_layout)
        self.save_mom0_btn = self._button('Save moment-0 FITS...', self.save_moment0, ctrl_layout)
        ctrl_layout.addStretch()

        self.status = self.statusBar()

        # Two-plot layout
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, stretch=1)

        map_widget = QtWidgets.QWidget()
        map_layout = QtWidgets.QVBoxLayout(map_widget)
        map_layout.setContentsMargins(0, 0, 0, 0)
        self.fig_map = Figure(figsize=(6, 5), layout='constrained')
        self.canvas_map = FigureCanvas(self.fig_map)
        self.map_toolbar = NavigationToolbar(self.canvas_map, self)
        map_layout.addWidget(self.map_toolbar)
        map_layout.addWidget(self.canvas_map, stretch=1)
        splitter.addWidget(map_widget)

        spec_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        spec_widget = QtWidgets.QWidget()
        spec_layout = QtWidgets.QVBoxLayout(spec_widget)
        spec_layout.setContentsMargins(0, 0, 0, 0)
        self.fig_spec = Figure(figsize=(6, 5), layout='constrained')
        self.canvas_spec = FigureCanvas(self.fig_spec)
        self.spec_toolbar = NavigationToolbar(self.canvas_spec, self)
        spec_layout.addWidget(self.spec_toolbar)
        spec_layout.addWidget(self.canvas_spec, stretch=1)
        spec_split.addWidget(spec_widget)
        spec_split.addWidget(self._build_analysis_tabs())
        spec_split.setStretchFactor(0, 3)
        spec_split.setStretchFactor(1, 2)
        splitter.addWidget(spec_split)

        # Polygon / spectrum controls
        poly_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(poly_layout)
        self.poly_btn = self._button('Draw polygon', self.start_polygon, poly_layout, checkable=True)
        self.reset_poly_btn = self._button('Reset polygons', self.reset_polygons, poly_layout)
        self.plot_selected_btn = self._button('Plot selected-region average', self.plot_selected_average, poly_layout)
        self.save_poly_btn = self._button('Save polygons', self.save_polygons, poly_layout)
        self.load_poly_btn = self._button('Load polygons', self.load_polygons, poly_layout)
        self.smooth_chk = QtWidgets.QCheckBox('Smooth outline')
        self.smooth_chk.setToolTip('Use a periodic spline through the vertices (for both display and pixel selection)')
        self.smooth_chk.toggled.connect(self._smooth_toggled)
        poly_layout.addWidget(self.smooth_chk)
        poly_layout.addStretch()
        poly_layout.addWidget(QtWidgets.QLabel('Hide edge channels:'))
        self.trim_spin = QtWidgets.QSpinBox()
        self.trim_spin.setRange(0, 100000)
        self.trim_spin.setValue(100)
        self.trim_spin.setToolTip('Number of channels hidden at each end of the spectrum')
        self.trim_spin.valueChanged.connect(lambda _: self.redraw_spectrum(keep_view=False))
        poly_layout.addWidget(self.trim_spin)
        self.save_spec_btn = self._button('Save last spectrum...', self.save_last_spectrum, poly_layout)
        self.export_spec_btn = self._button('Export spectrum (txt)...', self.export_spectrum, poly_layout)

        self.canvas_map.mpl_connect('button_press_event', self.on_map_click)
        self.canvas_spec.mpl_connect('button_press_event', self.on_spec_click)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self, activated=self.cancel_modes)

        self.ax_map = self._placeholder(self.fig_map, self.canvas_map, 'Open a FITS cube')
        self.ax_spec = self._placeholder(self.fig_spec, self.canvas_spec, 'Click on the map to show a spectrum')
        self.ax_resid = None

    def _build_analysis_tabs(self):
        tabs = QtWidgets.QTabWidget()

        # Baseline tab
        bl = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(bl)
        grid.addWidget(QtWidgets.QLabel('Line-free windows (km/s):'), 0, 0)
        self.win_edit = QtWidgets.QLineEdit()
        self.win_edit.setPlaceholderText('e.g.  -90:-40, 0:40')
        self.win_edit.editingFinished.connect(self._windows_changed)
        grid.addWidget(self.win_edit, 0, 1, 1, 3)
        row = QtWidgets.QHBoxLayout()
        self.win_pick_btn = self._button('Select on plot', self.toggle_window_select, row, checkable=True)
        self.win_pick_btn.setToolTip('Drag on the spectrum to add line-free windows (Esc to stop)')
        self._button('Clear windows', self.clear_windows, row)
        grid.addLayout(row, 0, 4)

        grid.addWidget(QtWidgets.QLabel('Polynomial order:'), 1, 0)
        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(0, 15)
        self.order_spin.setValue(1)
        self.order_spin.valueChanged.connect(lambda _: self._windows_changed())
        grid.addWidget(self.order_spin, 1, 1)
        row = QtWidgets.QHBoxLayout()
        self._button('Fit baseline', self.fit_spectrum_baseline, row)
        self.subtract_chk = QtWidgets.QCheckBox('Subtract')
        self.subtract_chk.setChecked(True)
        self.subtract_chk.toggled.connect(self._subtract_toggled)
        row.addWidget(self.subtract_chk)
        self.bl_auto_chk = QtWidgets.QCheckBox('Auto-fit new spectra')
        row.addWidget(self.bl_auto_chk)
        row.addStretch()
        grid.addLayout(row, 1, 2, 1, 3)

        row = QtWidgets.QHBoxLayout()
        self._button('Apply baseline to cube', self.apply_cube_baseline, row)
        self._button('Revert to original cube', self.revert_cube, row)
        self._button('Save cube FITS...', self.save_cube, row)
        row.addStretch()
        self.rms_label = QtWidgets.QLabel('rms: -')
        row.addWidget(self.rms_label)
        grid.addLayout(row, 2, 0, 1, 5)
        grid.setRowStretch(3, 1)
        tabs.addTab(bl, 'Baseline')

        # Line-fit tab
        gt = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(gt)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Model:'))
        self.model_combo = QtWidgets.QComboBox()
        for kind, label in MODEL_LABELS.items():
            self.model_combo.addItem(label, kind)
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        row.addWidget(self.model_combo)
        row.addWidget(QtWidgets.QLabel('HFS line:'))
        self.hfs_combo = QtWidgets.QComboBox()
        for name, preset in HFS_PRESETS.items():
            self.hfs_combo.addItem(name)
            self.hfs_combo.setItemData(self.hfs_combo.count() - 1,
                                       f'{len(preset["components"])} components, source: {preset.get("source", "?")}',
                                       QtCore.Qt.ToolTipRole)
        self.hfs_combo.addItem('Custom')
        self.hfs_combo.setMaxVisibleItems(25)
        self.hfs_combo.currentIndexChanged.connect(self._hfs_preset_changed)
        row.addWidget(self.hfs_combo)
        row.addWidget(QtWidgets.QLabel('offsets (km/s):'))
        self.hfs_off_edit = QtWidgets.QLineEdit()
        row.addWidget(self.hfs_off_edit, stretch=1)
        row.addWidget(QtWidgets.QLabel('rel. strengths:'))
        self.hfs_str_edit = QtWidgets.QLineEdit()
        row.addWidget(self.hfs_str_edit, stretch=1)
        vbox.addLayout(row)
        self.hfs_ref_label = QtWidgets.QLabel()
        self.hfs_ref_label.setStyleSheet('color: grey')
        if CATALOG_ERROR:
            self.hfs_ref_label.setToolTip(CATALOG_ERROR)
        vbox.addWidget(self.hfs_ref_label)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Components:'))
        self.ncomp_spin = QtWidgets.QSpinBox()
        self.ncomp_spin.setRange(1, 30)
        self.ncomp_spin.valueChanged.connect(lambda n: self.guess_table.setRowCount(n))
        row.addWidget(self.ncomp_spin)
        row.addWidget(QtWidgets.QLabel('Fit range (km/s):'))
        self.fit_lo_edit = QtWidgets.QLineEdit()
        self.fit_hi_edit = QtWidgets.QLineEdit()
        for edit in (self.fit_lo_edit, self.fit_hi_edit):
            edit.setPlaceholderText('view')
            edit.setMaximumWidth(70)
            row.addWidget(edit)
        self._button('From view', self.fit_range_from_view, row)
        self._button('Auto guess', self.auto_guess_clicked, row)
        self.pick_btn = self._button('Pick on plot', self.toggle_pick, row, checkable=True)
        self.pick_btn.setToolTip('Click on line peaks in the spectrum to add initial guesses (Esc to stop)')
        self._button('Fit', self.fit_clicked, row)
        self._button('Clear', self.clear_fit, row)
        self.fit_auto_chk = QtWidgets.QCheckBox('Auto-fit new spectra')
        row.addWidget(self.fit_auto_chk)
        row.addStretch()
        vbox.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Line markers:'))
        self.marks_combo = QtWidgets.QComboBox()
        self.marks_combo.addItems(['Off', 'Offline catalog', 'Splatalogue (online)'])
        self.marks_combo.currentIndexChanged.connect(lambda _: self.update_line_marks())
        row.addWidget(self.marks_combo)
        row.addWidget(QtWidgets.QLabel('Vsys (km/s):'))
        self.vsys_edit = QtWidgets.QLineEdit()
        self.vsys_edit.setPlaceholderText('auto')
        self.vsys_edit.setMaximumWidth(70)
        self.vsys_edit.setToolTip('Source velocity used to place the markers. Empty: the fitted V, '
                                  'or the velocity of the brightest channel in view')
        self.vsys_edit.editingFinished.connect(self._marks_vsys_changed)
        row.addWidget(self.vsys_edit)
        row.addWidget(QtWidgets.QLabel('Eu max (K):'))
        self.eu_spin = QtWidgets.QDoubleSpinBox()
        self.eu_spin.setRange(1, 5000)
        self.eu_spin.setValue(100)
        row.addWidget(self.eu_spin)
        self.known_chk = QtWidgets.QCheckBox('Known ISM species only')
        self.known_chk.setChecked(True)
        self.known_chk.setToolTip('Splatalogue: only species already detected in space')
        row.addWidget(self.known_chk)
        self.find_lines_btn = self._button('Find lines', self.update_line_marks, row)
        online_note = QtWidgets.QLabel('(Splatalogue needs an internet connection)')
        online_note.setStyleSheet('color: grey')
        row.addWidget(online_note)
        row.addStretch()
        vbox.addLayout(row)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.guess_table = QtWidgets.QTableWidget(1, 3)
        self.guess_table.setHorizontalHeaderLabels(['T peak [K]', 'V [km/s]', 'FWHM [km/s]'])
        self.guess_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.guess_table.itemChanged.connect(lambda _: self._guesses_edited())
        split.addWidget(self.guess_table)
        self.result_text = QtWidgets.QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        split.addWidget(self.result_text)
        split.setStretchFactor(1, 2)
        vbox.addWidget(split, stretch=1)
        tabs.addTab(gt, 'Line fit')

        self._hfs_preset_changed()
        self._model_changed()
        return tabs

    @staticmethod
    def _placeholder(fig, canvas, text):
        fig.clf()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes, color='grey')
        ax.set_xticks([])
        ax.set_yticks([])
        canvas.draw_idle()
        return ax

    # ------------------------------------------------------------------ helpers
    @contextmanager
    def _progress(self, label):
        """Modal progress dialog (shown only for slow operations); yields a callback(fraction)."""
        dlg = QtWidgets.QProgressDialog(label, 'Cancel', 0, 1000, self)
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumDuration(400)
        dlg.setValue(0)
        self.status.showMessage(label)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        def callback(frac):
            dlg.setValue(int(1000 * min(max(frac, 0.0), 1.0)))
            QtWidgets.QApplication.processEvents()
            if dlg.wasCanceled():
                raise Cancelled()
        try:
            yield callback
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            dlg.reset()
            dlg.deleteLater()

    def _start_dir(self):
        return os.path.dirname(self.path) if self.path else os.getcwd()

    def _need_cube(self):
        if self.data is None:
            QMessageBox.information(self, 'No cube', 'Please open a FITS cube first.')
            return False
        return True

    def _need_spectrum(self):
        if self.spec is None:
            QMessageBox.information(self, 'No spectrum', 'Show a spectrum first (click on the map or use a region).')
            return False
        return True

    def _is_kelvin(self):
        try:
            return self.unit.is_equivalent(u.K)
        except Exception:
            return False

    def _mom0_unit(self):
        return 'K km/s' if self._is_kelvin() else f'{self.unit} km/s'

    def _ylabel(self):
        return r'$\mathregular{T_{mb}}$ [K]' if self._is_kelvin() else f'Intensity [{self.unit}]'

    def _default_trim(self):
        return 100 if self.vel is not None and len(self.vel) > 400 else 0

    def _display_slice(self):
        n = len(self.vel)
        trim = self.trim_spin.value()
        if 2 * trim >= n - 1:
            trim = 0
        return slice(trim, n - trim)

    def _spec_y(self):
        y = self.spec['y']
        if self.baseline is not None and self.subtract_chk.isChecked():
            return y - self.baseline
        return y

    def _windows(self):
        return parse_windows(self.win_edit.text())

    def _noise_rms(self):
        """Noise estimate of the displayed spectrum from the baseline windows (None if unknown)."""
        if self.baseline_rms is not None and self.subtract_chk.isChecked():
            return float(self.baseline_rms)
        try:
            windows = self._windows()
        except ValueError:
            return None
        if not windows:
            return None
        y = self._spec_y()[windows_mask(self.vel, windows)]
        y = y[np.isfinite(y)]
        return float(np.std(y, ddof=1)) if y.size >= 3 else None

    # ------------------------------------------------------------------ loading / moment 0
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open FITS cube', self._start_dir(),
                                              'FITS files (*.fits *.fit *.fts *.fz *.fits.gz);;All files (*)')
        if path:
            self.load_cube(path)

    def load_cube(self, path):
        self.cancel_modes()
        hdul = None
        try:
            with self._progress(f'Loading {os.path.basename(path)} ...') as progress:
                hdul, cube, raw = open_cube(path)      # nothing read from disk yet
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    vel = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio').spectral_axis
                vel = np.asarray(vel.value, dtype=float)
                if raw.size * 4 <= ram_budget():
                    data = np.empty(raw.shape, dtype=np.float32)  # native-endian copy -> fast numpy
                    for c0, c1 in chunk_ranges(0, raw.shape[0], CHUNK_BYTES // (4 * raw.shape[1] * raw.shape[2])):
                        data[c0:c1] = raw[c0:c1]
                        progress(0.8 * c1 / raw.shape[0])
                    in_ram = True
                else:
                    data = raw
                    in_ram = False
                dv = np.abs(np.gradient(vel))
                mom0 = moment0(data, 0, len(vel), dv, lambda f: progress(0.8 + 0.2 * f))
                header, unit, wcs2d = cube.header, cube.unit, cube.wcs.celestial
                rest_freq = cube.wcs.wcs.restfrq or header.get('RESTFRQ') or header.get('RESTFREQ')
                del cube, raw
        except Cancelled:
            if hdul is not None:
                hdul.close()
            self.status.showMessage('Loading cancelled')
            return
        except Exception as e:
            if hdul is not None:
                hdul.close()
            QMessageBox.critical(self, 'Error', f'Failed to load FITS: {e}')
            return

        if in_ram:
            hdul.close()
            hdul = None
        old_hdul, self._hdul = self._hdul, hdul
        self.path = path
        self.header, self.unit = header, unit
        self.data_orig = self.data = data
        self.vel_orig = self.vel = vel
        self.dv = dv
        self.ch_offset = 0
        self.cube_baseline = None
        self.wcs2d = wcs2d if wcs2d.naxis == 2 else None
        self.l0, self.b0 = self.wcs2d.wcs.crval if self.wcs2d is not None else (None, None)
        self.mom0 = mom0
        self.mom0_range = (vel.min(), vel.max())
        self.polygons = []
        self.spec = None
        self.baseline = self.baseline_rms = None
        self.fit = None
        self._spec_drawn = False
        if old_hdul is not None:
            old_hdul.close()
        self.rest_freq_mhz = float(rest_freq) / 1e6 if rest_freq else None
        self._select_hfs_preset_for(self.rest_freq_mhz)

        self.trim_spin.blockSignals(True)
        self.trim_spin.setValue(self._default_trim())
        self.trim_spin.blockSignals(False)
        self.show_moment0()
        self.ax_spec = self._placeholder(self.fig_spec, self.canvas_spec, 'Click on the map to show a spectrum')
        self.ax_resid = None
        self.result_text.clear()
        self.rms_label.setText('rms: -')
        nchan, ny, nx = data.shape
        self.setWindowTitle(f'Spectral Cube Viewer - {os.path.basename(path)}')
        self.status.showMessage(f'Loaded {os.path.basename(path)}: {nx} x {ny} px, {nchan} channels '
                                f'({vel.min():.1f} .. {vel.max():.1f} km/s), '
                                f'{"in memory" if in_ram else "memory-mapped (large cube)"}')

    def _compute_mom0(self, vrange=None):
        """Moment 0 of the working cube over vrange (km/s) or over all channels. Raises Cancelled."""
        if vrange is not None:
            idx = np.nonzero((self.vel >= vrange[0]) & (self.vel <= vrange[1]))[0]
            if idx.size == 0:
                raise ValueError(f'No channels between {vrange[0]:g} and {vrange[1]:g} km/s '
                                 f'(cube covers {self.vel.min():.2f} .. {self.vel.max():.2f} km/s).')
            i0, i1 = int(idx[0]), int(idx[-1]) + 1
        else:
            i0, i1 = 0, len(self.vel)
        with self._progress('Computing moment 0 ...') as progress:
            self.mom0 = moment0(self.data, i0, i1, self.dv, progress)
        self.mom0_range = (min(self.vel[i0], self.vel[i1 - 1]), max(self.vel[i0], self.vel[i1 - 1]))

    def apply_spectral_slab(self):
        if not self._need_cube():
            return
        try:
            vmin = float(self.vmin_edit.text())
            vmax = float(self.vmax_edit.text())
        except ValueError:
            QMessageBox.warning(self, 'Input error', 'Enter valid numeric velocities in km/s')
            return
        if vmax <= vmin:
            QMessageBox.warning(self, 'Input error', 'vmax must be greater than vmin')
            return
        try:
            self._compute_mom0((vmin, vmax))
        except Cancelled:
            self.status.showMessage('Cancelled')
            return
        except ValueError as e:
            QMessageBox.warning(self, 'Input error', str(e))
            return
        self.show_moment0(keep_view=True)
        self.status.showMessage(f'Moment 0 over {self.mom0_range[0]:.2f} .. {self.mom0_range[1]:.2f} km/s')

    def show_moment0(self, keep_view=False):
        self._stop_polygon()
        lims = (self.ax_map.get_xlim(), self.ax_map.get_ylim()) if keep_view and self.ax_map.images else None
        self.fig_map.clf()
        if self.wcs2d is not None:
            ax = self.fig_map.add_subplot(111, projection=self.wcs2d)
        else:
            ax = self.fig_map.add_subplot(111)
        im = ax.imshow(np.ma.masked_invalid(self.mom0), origin='lower', interpolation='nearest')
        self.fig_map.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f'Moment-0 [{self._mom0_unit()}]')
        lo, hi = self.mom0_range
        title = f'Moment-0 map ({lo:.2f} .. {hi:.2f} km/s)'
        if self.cube_baseline is not None:
            title += ', baseline-corrected'
        ax.set_title(title, fontsize=10)
        if self.wcs2d is not None:
            ctype = self.wcs2d.wcs.ctype[0].upper()
            if ctype.startswith('GLON'):
                ax.set_xlabel('Galactic Longitude')
                ax.set_ylabel('Galactic Latitude')
            elif ctype.startswith('RA'):
                ax.set_xlabel('Right Ascension')
                ax.set_ylabel('Declination')
            ax.coords.grid(color='white', ls='dotted')
        else:
            ax.set_xlabel('x [pix]')
            ax.set_ylabel('y [pix]')
        for k, verts in enumerate(self.polygons, 1):
            outline = self._poly_outline(verts)
            ax.plot(outline[:, 0], outline[:, 1], color='white', linewidth=1.5)
            if len(self.polygons) > 1:
                ax.text(*verts.mean(axis=0), str(k), color='white', ha='center', va='center', fontsize=9)
        self._pix_marker, = ax.plot([], [], '+', color='red', ms=12, mew=2)
        if self.spec is not None and self.spec['source'][0] == 'pixel':
            self._pix_marker.set_data([self.spec['source'][1]], [self.spec['source'][2]])
        self.ax_map = ax
        self.map_toolbar.update()
        self.map_toolbar.push_current()      # Home = full map
        if lims is not None:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        self.canvas_map.draw_idle()

    def save_moment0(self):
        if not self._need_cube():
            return
        default = os.path.splitext(self.path)[0] + '_mom0.fits'
        path, _ = QFileDialog.getSaveFileName(self, 'Save moment-0 map', default, 'FITS files (*.fits)')
        if not path:
            return
        hdr = self.wcs2d.to_header() if self.wcs2d is not None else fits.Header()
        hdr['BUNIT'] = self._mom0_unit()
        hdr['VMIN'] = (float(self.mom0_range[0]), '[km/s] lower integration limit')
        hdr['VMAX'] = (float(self.mom0_range[1]), '[km/s] upper integration limit')
        hdr['HISTORY'] = f'Moment 0 of {os.path.basename(self.path)} (Spectral Cube Viewer)'
        if self.cube_baseline is not None:
            hdr['HISTORY'] = self._baseline_history()
        try:
            fits.PrimaryHDU(self.mom0.astype(np.float32), header=hdr).writeto(path, overwrite=True)
            self.status.showMessage(f'Moment-0 map saved to {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Save error', f'Could not save: {e}')

    # ------------------------------------------------------------------ spectra
    def on_map_click(self, event):
        if (self.data is None or event.button != 1 or event.inaxes is not self.ax_map or event.xdata is None
                or self.poly_selector is not None or _toolbar_active(self.map_toolbar)):
            return
        xi, yi = int(round(event.xdata)), int(round(event.ydata))
        ny, nx = self.data.shape[1:]
        if 0 <= xi < nx and 0 <= yi < ny:
            self.set_spectrum(('pixel', xi, yi))

    def _pixel_title(self, x, y):
        if self.wcs2d is None:
            return f'Pixel ({x}, {y})'
        lon, lat = self.wcs2d.wcs_pix2world([[x, y]], 0)[0]
        dl = ((lon - self.l0 + 180) % 360 - 180) * 3600
        db = (lat - self.b0) * 3600
        names = ('l', 'b') if self.wcs2d.wcs.ctype[0].upper().startswith('GLON') else (r'\alpha', r'\delta')
        return rf'$\Delta {names[0]}$: {dl:.0f}",  $\Delta {names[1]}$: {db:.0f}"   (pixel {x}, {y})'

    def _extract(self, source):
        if source[0] == 'pixel':
            _, x, y = source
            return np.array(self.data[:, y, x], dtype=float), self._pixel_title(x, y)
        _, mask, title = source
        with self._progress('Averaging spectra ...') as progress:
            spec = region_mean_spectrum(self.data, mask, progress)
        return spec, f'{title} ({int(mask.sum())} px)'

    def set_spectrum(self, source):
        try:
            y, title = self._extract(source)
        except Cancelled:
            self.status.showMessage('Cancelled')
            return
        except ValueError as e:
            QMessageBox.warning(self, 'Spectrum', str(e))
            return
        self.spec = dict(source=source, y=y, title=title)
        self.baseline = self.baseline_rms = None
        self.fit = None
        messages = []
        if self.bl_auto_chk.isChecked():
            try:
                self._fit_baseline_current()
            except ValueError as e:
                messages.append(f'Baseline not fitted: {e}')
        self._update_rms_label()
        if self.fit_auto_chk.isChecked():
            err = self._do_fit(quiet=True, redraw=False)
            if err:
                messages.append(f'Fit failed: {err}')
        else:
            self.result_text.clear()
        self.redraw_spectrum()
        if self._pix_marker is not None:
            if source[0] == 'pixel':
                self._pix_marker.set_data([source[1]], [source[2]])
            else:
                self._pix_marker.set_data([], [])
            self.canvas_map.draw_idle()
        self.status.showMessage('; '.join(messages) if messages else 'Spectrum: ' + title.replace('$', ''))

    def redraw_spectrum(self, keep_view=True):
        self._stop_span()
        old_xlim = self.ax_spec.get_xlim() if keep_view and self._spec_drawn else None
        # before clf(), which resets the old axes limits used for the automatic Vsys
        marks_vsys = self._marks_vsys() if self.line_marks and self.spec is not None else None
        self.fig_spec.clf()
        if self.spec is None:
            self.ax_spec = self._placeholder(self.fig_spec, self.canvas_spec, 'Click on the map to show a spectrum')
            self.ax_resid = None
            self._spec_drawn = False
            return
        fit = self.fit
        if fit is not None:
            gs = self.fig_spec.add_gridspec(2, 1, height_ratios=[3, 1])
            ax = self.fig_spec.add_subplot(gs[0])
            axr = self.fig_spec.add_subplot(gs[1], sharex=ax)
            ax.tick_params(labelbottom=False)
        else:
            ax = self.fig_spec.add_subplot(111)
            axr = None

        sl = self._display_slice()
        v, y = self.vel[sl], self._spec_y()[sl]
        visible = [(v, y)]
        ax.step(v, y, where='mid', color='k', lw=1)
        ax.axhline(0, ls='--', c='grey', alpha=0.5)
        try:
            windows = self._windows()
        except ValueError:
            windows = []
        for k, (lo, hi) in enumerate(windows):
            ax.axvspan(lo, hi, color='tab:green', alpha=0.12, lw=0, label='baseline windows' if k == 0 else None)
        if self.baseline is not None and not self.subtract_chk.isChecked():
            ax.plot(v, self.baseline[sl], color='tab:blue', ls='--', lw=1.2,
                    label=f'baseline (order {self.order_spin.value()})')
            visible.append((v, self.baseline[sl]))
        self._draw_line_marks(ax, np.nanmin(v), np.nanmax(v), marks_vsys)

        if fit is not None:
            m = fit.model
            lo, hi = fit.vrange
            vv = np.linspace(lo, hi, 3000)
            groups = m.groups(vv, fit.popt)
            if len(groups) > 1 or m.kind != MODEL_GAUSS:
                for g, (prof, p) in enumerate(zip(groups, m.split(fit.popt))):
                    color = f'C{g % 10}'
                    ax.plot(vv, prof, color=color, ls='--', lw=1)
                    if m.kind != MODEL_GAUSS:
                        ax.vlines(m.line_positions(p), 0, m.line_heights(p), color=color, lw=1.5)
            total = np.sum(groups, axis=0)
            ax.plot(vv, total, color='tab:red', lw=1.5, label='fit')
            visible.append((vv, total))
            ax.axvline(lo, color='tab:orange', ls=':', lw=1)
            ax.axvline(hi, color='tab:orange', ls=':', lw=1)
            sel = (self.vel >= lo) & (self.vel <= hi)
            resid = self._spec_y()[sel] - m(self.vel[sel], *fit.popt)
            axr.step(self.vel[sel], resid, where='mid', color='k', lw=0.8)
            axr.axhline(0, ls='--', c='grey', alpha=0.5)
            if fit.rms:
                axr.axhspan(-fit.rms, fit.rms, color='grey', alpha=0.2, lw=0)
            axr.set_ylabel('Resid. [K]')
            axr.set_xlabel('Velocity [km/s]')
        else:
            self._draw_guesses(ax)
            ax.set_xlabel('Velocity [km/s]')

        ax.set_ylabel(self._ylabel())
        ax.set_title(self.spec['title'], fontsize=10)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc='upper right')
        self.ax_spec, self.ax_resid = ax, axr
        self._spec_drawn = True

        self.spec_toolbar.update()
        self.spec_toolbar.push_current()     # Home = full spectrum
        if old_xlim is not None:
            ax.set_xlim(old_xlim)
            self._autoscale_y(ax, old_xlim, visible)
        if self.win_pick_btn.isChecked():
            self._start_span()
        self.canvas_spec.draw_idle()

    @staticmethod
    def _autoscale_y(ax, xlim, curves):
        lo, hi = sorted(xlim)
        vals = np.concatenate([y[(x >= lo) & (x <= hi)] for x, y in curves])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ymin, ymax = vals.min(), vals.max()
            pad = 0.06 * (ymax - ymin) or 1.0
            ax.set_ylim(ymin - pad, ymax + pad)

    def _draw_guesses(self, ax):
        guesses = self._table_guesses(strict=False)
        if not guesses:
            return
        try:
            model = self._model(len(guesses))
        except ValueError:
            return
        for t, v0, _ in guesses:
            ax.plot([v0], [t], 'v', color='tab:orange', ms=8)
        if model.kind != MODEL_GAUSS:
            for p in model.split(model.initial_params(guesses)):
                ax.vlines(model.line_positions(p), 0, model.line_heights(p), color='tab:orange', ls=':', lw=1.2)

    def plot_whole_average(self):
        if not self._need_cube():
            return
        self.set_spectrum(('region', np.ones(self.data.shape[1:], dtype=bool), 'Average spectrum (whole cube)'))

    def save_last_spectrum(self):
        if not self._need_spectrum():
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save spectrum image', self._start_dir(),
                                              'Images (*.png *.jpg);;Vector (*.svg *.pdf)')
        if not path:
            return
        try:
            self.fig_spec.savefig(path, dpi=200)
            self.status.showMessage(f'Spectrum saved to {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Save error', f'Could not save: {e}')

    def export_spectrum(self):
        if not self._need_spectrum():
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export spectrum', self._start_dir(), 'Text files (*.txt *.dat)')
        if not path:
            return
        nan = np.full(len(self.vel), np.nan)
        baseline = self.baseline if self.baseline is not None else nan
        model = self.fit.model(self.vel, *self.fit.popt) if self.fit is not None else nan
        header = [f'Spectrum: {self.spec["title"]}'.replace('$', ''), f'Cube: {self.path}']
        if self.cube_baseline is not None:
            header.append(self._baseline_history())
        if self.fit is not None:
            header += [''] + self.fit.report().splitlines() + ['']
        header.append('velocity[km/s]  raw  baseline  spectrum(used)  model')
        try:
            np.savetxt(path, np.column_stack([self.vel, self.spec['y'], baseline, self._spec_y(), model]),
                       header='\n'.join(header), fmt='%.6g')
            self.status.showMessage(f'Spectrum exported to {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Save error', f'Could not save: {e}')

    # ------------------------------------------------------------------ polygons
    def start_polygon(self):
        if self.poly_selector is not None:   # button pressed again -> cancel
            self.cancel_modes()
            return
        if self.mom0 is None:
            QMessageBox.information(self, 'No map', 'Compute a moment-0 map first')
            self.poly_btn.setChecked(False)
            return
        self.poly_selector = PolygonSelector(self.ax_map, self.on_polygon_complete, useblit=True,
                                             props=dict(color='white', linewidth=1.5))
        self.poly_btn.setChecked(True)
        self.status.showMessage('Click the polygon vertices; click the first vertex to close it (Esc cancels).')

    def _stop_polygon(self):
        if self.poly_selector is not None:
            self.poly_selector.set_active(False)
            self.poly_selector.disconnect_events()
            self.poly_selector = None
        self.poly_btn.setChecked(False)

    def on_polygon_complete(self, verts):
        verts = np.asarray(verts, dtype=float)
        if len(verts) >= 3:
            # defer: the selector is still processing the mouse event that closed the polygon
            QtCore.QTimer.singleShot(0, lambda: self._finish_polygon(verts))

    def _finish_polygon(self, verts):
        if self.poly_selector is None:
            return
        self.polygons.append(self._clean_vertices(verts))
        self.show_moment0(keep_view=True)
        mask = self._polygon_mask(self.polygons[-1:])
        if not mask.any():
            QMessageBox.information(self, 'Empty polygon', 'The polygon contains no pixel centres.')
            return
        self.set_spectrum(('region', mask, f'Polygon #{len(self.polygons)} average'))

    @staticmethod
    def _clean_vertices(verts):
        verts = np.asarray(verts, dtype=float)[:, :2]
        if len(verts) > 3 and np.allclose(verts[0], verts[-1]):
            verts = verts[:-1]
        return verts

    def _poly_outline(self, verts):
        """Closed outline actually used for drawing and for the pixel selection."""
        pts = np.vstack([verts, verts[:1]])
        if not self.smooth_chk.isChecked() or len(verts) < 3:
            return pts
        spline = interpolate.CubicSpline(np.arange(len(pts)), pts, bc_type='periodic')
        return spline(np.linspace(0, len(pts) - 1, 50 * len(verts)))

    def _polygon_mask(self, polygons):
        ny, nx = self.data.shape[1:]
        mask = np.zeros((ny, nx), dtype=bool)
        for verts in polygons:
            outline = self._poly_outline(verts)
            x0, y0 = np.maximum(np.floor(outline.min(axis=0)).astype(int), 0)
            x1, y1 = np.ceil(outline.max(axis=0)).astype(int) + 1
            x1, y1 = min(x1, nx), min(y1, ny)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            inside = Path(outline).contains_points(np.column_stack([xx.ravel(), yy.ravel()]))
            mask[y0:y1, x0:x1] |= inside.reshape(yy.shape)
        return mask

    def _smooth_toggled(self):
        if self.mom0 is not None:
            self.show_moment0(keep_view=True)

    def save_polygons(self):
        if not self.polygons:
            QMessageBox.information(self, 'No polygons', 'There are no polygons to save.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save polygon file', self._start_dir(), 'Polygon files (*.poly)')
        if not path:
            return
        lines = ['# Spectral Cube Viewer polygons',
                 f'# cube: {os.path.basename(self.path)}',
                 '# columns: x y (0-based pixel)' + ('  lon lat (deg, used when loaded on another grid)'
                                                     if self.wcs2d is not None else '')]
        for k, verts in enumerate(self.polygons, 1):
            lines.append(f'# polygon {k}')
            world = self.wcs2d.wcs_pix2world(verts, 0) if self.wcs2d is not None else None
            for i, (x, y) in enumerate(verts):
                row = f'{x:.10e} {y:.10e}'
                if world is not None:
                    row += f' {world[i, 0]:.10f} {world[i, 1]:.10f}'
                lines.append(row)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            self.status.showMessage(f'{len(self.polygons)} polygon(s) saved to {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Save error', f'Could not save: {e}')

    def load_polygons(self):
        if not self._need_cube():
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Load polygon file', self._start_dir(),
                                              'Polygon files (*.poly *.txt *.npy)')
        if not path:
            return
        try:
            if path.lower().endswith('.npy'):
                arr = np.load(path)
                blocks = [arr] if arr.ndim == 2 else list(arr)
            else:
                blocks, current = [], []
                with open(path, encoding='utf-8', errors='replace') as f:
                    for line in f:
                        s = line.strip()
                        if s.lower().startswith('# polygon') and current:
                            blocks.append(current)
                            current = []
                        if s and not s.startswith('#'):
                            current.append([float(t) for t in s.split()])
                if current:
                    blocks.append(current)
            polygons = []
            for block in blocks:
                arr = np.asarray(block, dtype=float)
                if arr.shape[1] >= 4 and self.wcs2d is not None:   # world coordinates -> this cube's pixels
                    arr = self.wcs2d.wcs_world2pix(arr[:, 2:4], 0)
                if len(arr) >= 3:
                    polygons.append(self._clean_vertices(arr))
            if not polygons:
                raise ValueError('no polygon with at least 3 vertices found')
        except Exception as e:
            QMessageBox.critical(self, 'Load error', f'Could not load polygons: {e}')
            return
        self.polygons = polygons
        self.show_moment0(keep_view=True)
        self.status.showMessage(f'Loaded {len(polygons)} polygon(s) from {path}')

    def reset_polygons(self):
        self.polygons = []
        if self.mom0 is not None:
            self.show_moment0(keep_view=True)

    def plot_selected_average(self):
        if not self.polygons:
            QMessageBox.information(self, 'No polygons', 'Draw at least one polygon first')
            return
        mask = self._polygon_mask(self.polygons)
        if not mask.any():
            QMessageBox.information(self, 'Empty region', 'The polygons contain no pixel centres.')
            return
        self.set_spectrum(('region', mask, f'Combined average of {len(self.polygons)} polygon(s)'))

    # ------------------------------------------------------------------ interaction modes
    def cancel_modes(self):
        if self.poly_selector is not None:
            self._stop_polygon()
            self.show_moment0(keep_view=True)
        self.poly_btn.setChecked(False)
        if self.win_pick_btn.isChecked():
            self.win_pick_btn.setChecked(False)
            self._stop_span()
            self.canvas_spec.draw_idle()
        self.pick_btn.setChecked(False)

    def toggle_window_select(self):
        if not self.win_pick_btn.isChecked():
            self._stop_span()
            return
        if not self._need_spectrum():
            self.win_pick_btn.setChecked(False)
            return
        self.pick_btn.setChecked(False)
        self._start_span()
        self.status.showMessage('Drag on the spectrum to add line-free baseline windows (Esc to stop).')

    def _start_span(self):
        self._stop_span()
        self.span_selector = SpanSelector(self.ax_spec, self._on_span, 'horizontal', useblit=True,
                                          props=dict(facecolor='tab:green', alpha=0.3),
                                          minspan=float(np.median(self.dv)))

    def _stop_span(self):
        if self.span_selector is not None:
            self.span_selector.set_active(False)
            self.span_selector.disconnect_events()
            self.span_selector = None

    def _on_span(self, lo, hi):
        if _toolbar_active(self.spec_toolbar):
            return
        try:
            windows = self._windows()
        except ValueError:
            windows = []
        self.win_edit.setText(format_windows(merge_windows(windows + [(lo, hi)])))
        QtCore.QTimer.singleShot(0, self._windows_changed)

    def toggle_pick(self):
        if self.pick_btn.isChecked():
            if not self._need_spectrum():
                self.pick_btn.setChecked(False)
                return
            if self.win_pick_btn.isChecked():
                self.win_pick_btn.setChecked(False)
                self._stop_span()
            self.status.showMessage('Click on line peaks to add initial guesses (Esc to stop).')

    def on_spec_click(self, event):
        if (not self.pick_btn.isChecked() or event.button != 1 or event.inaxes is not self.ax_spec
                or event.xdata is None or _toolbar_active(self.spec_toolbar)):
            return
        guesses = self._table_guesses(strict=False)
        fwhm = guesses[-1][2] if guesses else max(3 * float(np.median(self.dv)), 1.0)
        guesses.append((event.ydata, event.xdata, fwhm))
        self._set_guesses(guesses)
        self.fit = None
        QtCore.QTimer.singleShot(0, self.redraw_spectrum)

    # ------------------------------------------------------------------ baseline
    def _update_rms_label(self):
        rms = self._noise_rms() if self.spec is not None else None
        self.rms_label.setText(f'rms: {rms:.4g} K' if rms else 'rms: -')

    def _fit_baseline_current(self):
        windows = self._windows()
        if not windows:
            raise ValueError('define at least one line-free window, e.g. "-90:-40, 0:40"')
        self.baseline, self.baseline_rms = fit_baseline(self.vel, self.spec['y'], windows_mask(self.vel, windows),
                                                        self.order_spin.value())

    def fit_spectrum_baseline(self):
        if not self._need_spectrum():
            return
        try:
            self._fit_baseline_current()
        except ValueError as e:
            QMessageBox.warning(self, 'Baseline', str(e))
            return
        self.fit = None
        self._update_rms_label()
        self.redraw_spectrum()

    def _windows_changed(self):
        if self.spec is None:
            return
        if self.baseline is not None or self.bl_auto_chk.isChecked():
            try:
                self._fit_baseline_current()
            except ValueError as e:
                self.baseline = self.baseline_rms = None
                self.status.showMessage(f'Baseline not fitted: {e}')
            if self.subtract_chk.isChecked():
                self.fit = None
        self._update_rms_label()
        self.redraw_spectrum()

    def clear_windows(self):
        self.win_edit.clear()
        self.baseline = self.baseline_rms = None
        if self.spec is not None:
            self.fit = None
            self._update_rms_label()
            self.redraw_spectrum()

    def _subtract_toggled(self):
        if self.spec is not None and self.baseline is not None:
            self.fit = None
            self._update_rms_label()
            self.redraw_spectrum()

    def _baseline_history(self):
        cb = self.cube_baseline
        return f'Baseline: polynomial order {cb["order"]}, windows {format_windows(cb["windows"])} km/s'

    def apply_cube_baseline(self):
        if not self._need_cube():
            return
        try:
            windows = self._windows()
            if not windows:
                raise ValueError('Define at least one line-free window first, e.g. "-90:-40, 0:40".')
            wmask = windows_mask(self.vel_orig, windows)
            if not wmask.any():
                raise ValueError('The windows contain no channels of the cube.')
            order = self.order_spin.value()
            with self._progress('Fitting baselines of all spectra ...') as progress:
                out, rms, i0, i1 = baseline_cube(self.data_orig, self.vel_orig, wmask, order, progress)
        except Cancelled:
            self.status.showMessage('Cancelled - cube unchanged')
            return
        except ValueError as e:
            QMessageBox.warning(self, 'Baseline', str(e))
            return
        self.data = out
        self.vel = self.vel_orig[i0:i1]
        self.dv = np.abs(np.gradient(self.vel))
        self.ch_offset = i0
        self.cube_baseline = dict(windows=windows, order=order, rms=rms)
        self.baseline = self.baseline_rms = None
        self.bl_auto_chk.setChecked(False)
        self.trim_spin.blockSignals(True)
        self.trim_spin.setValue(0)
        self.trim_spin.blockSignals(False)
        if not self._after_cube_change():
            return
        self.status.showMessage(f'Baseline (order {order}) subtracted from all spectra; kept channels {i0}-{i1 - 1} '
                                f'({self.vel.min():.2f} .. {self.vel.max():.2f} km/s); '
                                f'median rms = {np.nanmedian(rms):.4g} K')

    def revert_cube(self):
        if self.data_orig is None or self.cube_baseline is None:
            return
        self.data, self.vel = self.data_orig, self.vel_orig
        self.dv = np.abs(np.gradient(self.vel))
        self.ch_offset = 0
        self.cube_baseline = None
        self.trim_spin.blockSignals(True)
        self.trim_spin.setValue(self._default_trim())
        self.trim_spin.blockSignals(False)
        if self._after_cube_change():
            self.status.showMessage('Reverted to the original cube')

    def _after_cube_change(self):
        """Recompute the moment-0 map and the current spectrum from the new working cube."""
        lo, hi = self.mom0_range
        vrange = (lo, hi) if np.any((self.vel >= lo) & (self.vel <= hi)) else None
        done = True
        try:
            self._compute_mom0(vrange)
            self.show_moment0(keep_view=True)
        except Cancelled:
            self.status.showMessage('Moment 0 not recomputed (cancelled)')
            done = False
        if self.spec is not None:   # the old spectrum no longer matches self.vel
            self.set_spectrum(self.spec['source'])
        return done

    def save_cube(self):
        if self.cube_baseline is None:
            QMessageBox.information(self, 'Nothing to save', 'Apply a baseline to the cube first.')
            return
        default = os.path.splitext(self.path)[0] + '_blcorr.fits'
        path, _ = QFileDialog.getSaveFileName(self, 'Save baseline-corrected cube', default, 'FITS files (*.fits)')
        if not path:
            return
        hdr = self.header.copy()
        for key in ('BSCALE', 'BZERO', 'BLANK', 'DATAMIN', 'DATAMAX'):
            hdr.remove(key, ignore_missing=True)
        hdr['CRPIX3'] = hdr['CRPIX3'] - self.ch_offset
        hdr['HISTORY'] = f'{self._baseline_history()} (Spectral Cube Viewer)'
        try:
            with self._progress('Writing cube ...'):
                fits.PrimaryHDU(self.data, header=hdr).writeto(path, overwrite=True)
            self.status.showMessage(f'Cube saved to {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Save error', f'Could not save: {e}')

    # ------------------------------------------------------------------ line fitting
    def _hfs_definition(self):
        try:
            offsets = [float(s) for s in re.split(r'[,\s;]+', self.hfs_off_edit.text().strip()) if s]
            strengths = [float(s) for s in re.split(r'[,\s;]+', self.hfs_str_edit.text().strip()) if s]
        except ValueError:
            raise ValueError('HFS offsets and strengths must be numbers separated by commas.')
        if not offsets or len(offsets) != len(strengths):
            raise ValueError('Give the same number (>0) of HFS velocity offsets and relative strengths.')
        if min(strengths) <= 0:
            raise ValueError('HFS relative strengths must be positive.')
        preset = HFS_PRESETS.get(self.hfs_combo.currentText())
        labels = [c['label'] for c in preset['components']] if preset else None
        if labels is not None and len(labels) != len(offsets):
            labels = None
        return dict(offsets=offsets, strengths=strengths, labels=labels)

    def _model(self, ncomp=None):
        kind = self.model_combo.currentData()
        hfs = self._hfs_definition() if kind != MODEL_GAUSS else None
        return LineModel(kind, ncomp or self.ncomp_spin.value(), hfs)

    def _model_changed(self):
        hfs = self.model_combo.currentData() != MODEL_GAUSS
        for w in (self.hfs_combo, self.hfs_off_edit, self.hfs_str_edit):
            w.setEnabled(hfs)
        self._hfs_preset_changed()
        self.guess_table.setHorizontalHeaderLabels(
            ['T peak (strongest line) [K]' if hfs else 'T peak [K]', 'V [km/s]', 'FWHM [km/s]'])
        self._clear_fit_result()

    def _hfs_preset_changed(self):
        preset = HFS_PRESETS.get(self.hfs_combo.currentText())
        custom = preset is None
        hfs = self.model_combo.currentData() != MODEL_GAUSS
        self.hfs_off_edit.setReadOnly(not custom)
        self.hfs_str_edit.setReadOnly(not custom)
        self.hfs_off_edit.setEnabled(hfs)
        self.hfs_str_edit.setEnabled(hfs)
        if preset is not None:
            ref, from_cube = preset_reference(preset, self.rest_freq_mhz)
            definition = preset_hfs(preset, ref)
            self.hfs_off_edit.setText(', '.join(f'{o:.4f}' for o in definition['offsets']))
            self.hfs_str_edit.setText(', '.join(f'{s:.4g}' for s in definition['strengths']))
            self.hfs_ref_label.setText(
                f'{self.hfs_combo.currentText()}: {len(preset["components"])} components '
                f'({preset.get("source", "?")}); offsets relative to '
                + (f'the cube rest frequency {ref:.4f} MHz, so V is the systemic velocity' if from_cube else
                   f'the strongest component {ref:.4f} MHz (cube rest frequency is not in this multiplet)'))
        else:
            self.hfs_ref_label.setText('Custom: velocity offsets of the hyperfine components relative to the '
                                       'cube rest frequency, and their relative strengths')
        if CATALOG_ERROR:
            self.hfs_ref_label.setText(self.hfs_ref_label.text() + f'   [{CATALOG_ERROR}]')
        self._clear_fit_result()

    def _select_hfs_preset_for(self, rest_freq_mhz):
        name = find_preset(rest_freq_mhz)
        if name is not None and name != self.hfs_combo.currentText():
            self.hfs_combo.setCurrentText(name)     # triggers _hfs_preset_changed
        else:
            self._hfs_preset_changed()              # same preset, new reference frequency

    # ------------------------------------------------------------------ line markers
    def _marks_vsys(self):
        """Velocity used to place line markers: user value, else the fitted V, else the brightest channel in view."""
        try:
            return float(self.vsys_edit.text())
        except ValueError:
            pass
        if self.fit is not None:
            return float(self.fit.popt[self.fit.model.iv])
        sl = self._display_slice()
        v, y = self.vel[sl], self._spec_y()[sl]
        if self._spec_drawn:
            lo, hi = sorted(self.ax_spec.get_xlim())
            inside = (v >= lo) & (v <= hi)
            if inside.sum() >= 3:
                v, y = v[inside], y[inside]
        y = np.convolve(np.nan_to_num(y, nan=-np.inf), np.ones(3) / 3, mode='same')
        return float(v[int(np.argmax(y))])

    def _marks_vsys_changed(self):
        if self.line_marks and self.spec is not None:
            self.redraw_spectrum()

    def update_line_marks(self):
        mode = self.marks_combo.currentIndex()
        if mode == 0:
            self.line_marks = []
            if self.spec is not None:
                self.redraw_spectrum()
            return
        if self.spec is None or not self.rest_freq_mhz:
            if self.spec is not None:
                QMessageBox.information(self, 'Line markers', 'The cube header has no rest frequency (RESTFRQ).')
            return
        if mode == 1:
            self.line_marks = catalog_lines(self.eu_spin.value())
            if not self.line_marks:
                QMessageBox.information(self, 'Line markers', CATALOG_ERROR or 'The offline catalog has no lines.')
            self.redraw_spectrum()
            self.status.showMessage(f'Line markers from {os.path.basename(CATALOG_PATH)} '
                                    f'(Vsys = {self._marks_vsys():.2f} km/s)')
            return
        if self._marks_thread is not None:
            return
        vsys = self._marks_vsys()
        sl = self._display_slice()
        v = self.vel[sl]
        freqs = self.rest_freq_mhz * (1 - (np.array([v.min(), v.max()]) - vsys) / C_KMS)
        self.find_lines_btn.setEnabled(False)
        self.status.showMessage(f'Querying Splatalogue for {freqs.min():.1f} .. {freqs.max():.1f} MHz '
                                f'(requires an internet connection) ...')
        self._marks_worker = _Worker(splatalogue_lines, freqs.min(), freqs.max(), self.eu_spin.value(),
                                     self.known_chk.isChecked())
        self._marks_thread = QtCore.QThread(self)
        self._marks_worker.moveToThread(self._marks_thread)
        self._marks_thread.started.connect(self._marks_worker.run)
        self._marks_worker.finished.connect(self._online_marks_done)
        self._marks_thread.start()

    def _online_marks_done(self, result, error):
        self._marks_thread.quit()
        self._marks_thread.wait()
        self._marks_thread = self._marks_worker = None
        self.find_lines_btn.setEnabled(True)
        if error:
            self.status.showMessage('Splatalogue query failed')
            QMessageBox.warning(self, 'Splatalogue', 'The online line search failed. It needs an internet '
                                                     f'connection and the astroquery package.\n\n{error}')
            return
        self.line_marks = result
        if self.spec is not None:
            self.redraw_spectrum()
        self.status.showMessage(f'Splatalogue: {len(result)} line(s) in the spectrum range '
                                f'(Vsys = {self._marks_vsys():.2f} km/s)')

    def _draw_line_marks(self, ax, vlo, vhi, vsys):
        if not self.line_marks or not self.rest_freq_mhz or vsys is None:
            return
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for mark in self.line_marks:
            vm = vsys + C_KMS * (self.rest_freq_mhz - mark['freq_mhz']) / self.rest_freq_mhz
            if vlo <= vm <= vhi:
                ax.axvline(vm, color='tab:purple', ls=':', lw=0.9, alpha=0.8)
                label = f'{mark["species"]} {mark["transition"]}'
                ax.text(vm, 0.98, label[:32], transform=trans, rotation=90, va='top', ha='right',
                        fontsize=7, color='tab:purple', clip_on=True)

    def _clear_fit_result(self):
        if self.fit is not None:
            self.fit = None
            self.result_text.clear()
            if self.spec is not None:
                self.redraw_spectrum()

    def _table_guesses(self, strict):
        guesses = []
        for r in range(self.guess_table.rowCount()):
            try:
                guesses.append(tuple(float(self.guess_table.item(r, c).text()) for c in range(3)))
            except (AttributeError, ValueError):
                if strict:
                    raise ValueError(f'Initial guess in row {r + 1} is incomplete.')
        return guesses

    def _set_guesses(self, guesses):
        self.guess_table.blockSignals(True)
        self.ncomp_spin.blockSignals(True)
        self.ncomp_spin.setValue(max(len(guesses), 1))
        self.guess_table.setRowCount(max(len(guesses), 1))
        self.guess_table.clearContents()
        for r, row in enumerate(guesses):
            for c, val in enumerate(row):
                self.guess_table.setItem(r, c, QtWidgets.QTableWidgetItem(f'{val:.4g}'))
        self.ncomp_spin.blockSignals(False)
        self.guess_table.blockSignals(False)

    def _guesses_edited(self):
        if self.spec is not None and self.fit is None:
            self.redraw_spectrum()

    def _fit_range(self):
        lo_txt, hi_txt = self.fit_lo_edit.text().strip(), self.fit_hi_edit.text().strip()
        if not lo_txt and not hi_txt:
            return tuple(sorted(self.ax_spec.get_xlim()))
        try:
            lo, hi = float(lo_txt), float(hi_txt)
        except ValueError:
            raise ValueError('Fit range: give both velocities, or leave both empty to use the current view.')
        return min(lo, hi), max(lo, hi)

    def fit_range_from_view(self):
        if not self._need_spectrum():
            return
        lo, hi = sorted(self.ax_spec.get_xlim())
        self.fit_lo_edit.setText(f'{lo:.2f}')
        self.fit_hi_edit.setText(f'{hi:.2f}')

    def _fit_data(self):
        lo, hi = self._fit_range()
        y = self._spec_y()
        sel = (self.vel >= lo) & (self.vel <= hi) & np.isfinite(y)
        return self.vel[sel], y[sel], (lo, hi)

    def auto_guess_clicked(self):
        if not self._need_spectrum():
            return
        try:
            model = self._model()
            vel, y, _ = self._fit_data()
        except ValueError as e:
            QMessageBox.warning(self, 'Auto guess', str(e))
            return
        guesses = auto_guess(vel, y, self.ncomp_spin.value(), model, self._noise_rms())
        if not guesses:
            QMessageBox.information(self, 'Auto guess', 'No emission above 3 rms found in the fit range.')
            return
        if len(guesses) < self.ncomp_spin.value():
            self.status.showMessage(f'Only {len(guesses)} component(s) above 3 rms found')
        self._set_guesses(guesses)
        self.fit = None
        self.result_text.clear()
        self.redraw_spectrum()

    def fit_clicked(self):
        if self._need_spectrum():
            self._do_fit(quiet=False)

    def _do_fit(self, quiet, redraw=True):
        """Fit the current spectrum; returns an error message or None."""
        error = None
        try:
            vel, y, vrange = self._fit_data()
            rms = self._noise_rms()
            try:
                guesses = self._table_guesses(strict=True)
            except ValueError:
                guesses = auto_guess(vel, y, self.ncomp_spin.value(), self._model(), rms)
                if not guesses:
                    raise ValueError('No initial guesses: fill in the table, use "Auto guess" or "Pick on plot".')
                self._set_guesses(guesses)
            self.fit = fit_lines(vel, y, self._model(len(guesses)), guesses, vrange, rms)
            self.result_text.setPlainText(self.fit.report())
        except (ValueError, RuntimeError) as e:
            self.fit = None
            self.result_text.setPlainText(f'Fit failed: {e}')
            error = str(e)
            if not quiet:
                QMessageBox.warning(self, 'Fit failed', error)
        if redraw:
            self.redraw_spectrum()
        return error

    def clear_fit(self):
        self.fit = None
        self.result_text.clear()
        self._set_guesses([])
        if self.spec is not None:
            self.redraw_spectrum()


def _excepthook(exc_type, exc, tb):
    # PyQt5 aborts the application on unhandled exceptions in slots; show them instead
    traceback.print_exception(exc_type, exc, tb)
    if QtWidgets.QApplication.instance() is not None:
        QMessageBox.critical(None, 'Unexpected error', f'{exc_type.__name__}: {exc}')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='spectral-cube-viewer',
                                     description='Moment-0 maps, spectra, baselines and line fitting for FITS cubes.')
    parser.add_argument('cube', nargs='?', help='FITS cube to open')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    args, qt_args = parser.parse_known_args(argv)

    sys.excepthook = _excepthook
    app = QtWidgets.QApplication([sys.argv[0]] + qt_args)
    app.setApplicationName('Spectral Cube Viewer')
    win = SpectralCubeViewer()
    win.show()
    if args.cube:
        QtCore.QTimer.singleShot(0, lambda: win.load_cube(args.cube))
    sys.exit(app.exec_())
