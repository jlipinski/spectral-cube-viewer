# Spectral Cube Viewer

An interactive viewer for radio spectral-line FITS cubes (position–position–velocity). You can:

- make **moment-0 maps** over any velocity range;
- look at **spectra** of single pixels, of polygon regions, or averaged over the whole cube;
- remove **polynomial baselines** from one spectrum or from the whole cube;
- **fit lines** with several Gaussians, or with **hyperfine-structure models** (HCN, N2H+, CN, C2H, …);
- **mark lines** on a spectrum to identify them, using an offline catalog or Splatalogue.

It was written for TRAO and PMO data. It works with any cube that has a celestial and a spectral axis, including CLASS/GILDAS exports. It may not work without proper FITS header.

<!-- Add a screenshot here, e.g. docs/screenshot.png -->

## Installation

You need Python 3.10 or newer. The program runs on Windows, Linux and macOS.

**With pipx (recommended).** This installs the program in its own environment and adds a `spectral-cube-viewer` command:

```bash
pipx install git+https://github.com/jlipinski/spectral-cube-viewer
pipx inject spectral-cube-viewer astroquery      # optional: Splatalogue line search
```

**With pip** (preferably inside a virtual environment):

```bash
pip install "spectral-cube-viewer[online] @ git+https://github.com/jlipinski/spectral-cube-viewer"
```

With pip, the `[online]` part installs `astroquery`, which the Splatalogue line search needs. You can leave it out.

**With conda.** Use this if `pip install PyQt5` fails, which can happen on Apple-silicon Macs:

```bash
git clone https://github.com/jlipinski/spectral-cube-viewer
cd spectral-cube-viewer
conda env create -f environment.yml
conda activate spectral-cube-viewer
```

### Platform notes

- **Linux.** If Qt reports `could not load the Qt platform plugin "xcb"`, install the X11 libraries Qt needs:
  `sudo apt install libxcb-xinerama0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0`
  (Debian/Ubuntu package names). On Wayland, Qt5 runs through XWayland; if the window misbehaves, start it with `QT_QPA_PLATFORM=xcb spectral-cube-viewer`.
- **macOS.** If the pip installation of PyQt5 fails, use the conda route above.
- **Windows.** No extra steps. `spectral-cube-viewer` starts without a console window.

## Usage

```bash
spectral-cube-viewer                 # then use "Open FITS..."
spectral-cube-viewer cube.fits       # open a cube directly
python -m spectral_cube_viewer       # same thing, without the installed command
```

### Maps and spectra

- **Moment 0.** Enter *v min* and *v max* and press *Apply slab & show Moment0*. *Save moment-0 FITS...* writes the map, keeping its sky coordinates.
- **Spectra.** Left-click a pixel on the map. Clicks are ignored while the zoom or pan tool is active.
- **Region averages.** Press *Draw polygon*, click the vertices, then click the first vertex again to close the shape. You can draw several polygons; *Plot selected-region average* averages all of them together.
  - Polygons can be saved and loaded. The files store sky coordinates too, so a region drawn on one cube can be loaded onto another cube with a different pixel grid (e.g. 12CO and HCN maps of the same field).
  - *Smooth outline* replaces the straight edges with a smooth curve, used both for drawing and for choosing which pixels are inside.
- **Navigation.** The matplotlib toolbar above each plot zooms, pans and saves. *Hide edge channels* hides noisy channels at the ends of the band. *Esc* cancels drawing or picking.

### Baselines (*Baseline* tab)

1. Give line-free windows as velocity ranges, e.g. `-90:-40, 0:40`, or press *Select on plot* and drag across the spectrum.
2. Choose the polynomial order and press *Fit baseline*.
   - *Subtract* switches between the corrected spectrum and the raw spectrum with the baseline drawn on top.
   - *Auto-fit new spectra* fits the baseline of every spectrum you open.
   - The rms in the windows is used as the noise level in line fits.
3. *Apply baseline to cube* fits a baseline to every spectrum in the cube. Moment-0 maps and spectra then use the corrected cube. It keeps only the channels between the outermost windows, and can be saved with *Save cube FITS...* or undone with *Revert to original cube*.

### Line fitting (*Line fit* tab)

There are four models:

| Model | Parameters per component |
|---|---|
| Gaussians (independent) | T, V, FWHM (negative T allowed for absorption) |
| HFS – fixed LTE ratios | T of the strongest line, V, FWHM; optically thin relative intensities |
| HFS – free intensities | T of each hyperfine line; V and FWHM shared |
| HFS – with optical depth | A, V, FWHM, τ(main): T(v) = A·(1 − e^(−τ(v))) |

- **Hyperfine preset.** When a cube is opened, the preset matching its rest frequency is selected automatically. The hyperfine velocity offsets are computed from the cube's own rest frequency, so the fitted V is the systemic velocity, even when `RESTFRQ` is the unsplit line frequency. The line under the preset selector shows which frequency is used. Choose *Custom* to type your own velocity offsets and relative strengths.
- **Initial guesses.** Use *Auto guess*, *Pick on plot* (click the line peaks), or type them into the table.
- **Fit range.** Type a velocity range, or leave both fields empty to use the current zoom of the plot.
- **Results.** Each parameter is reported with its error, plus the integrated intensity with its error and the reduced χ². A residual panel appears below the spectrum. *Auto-fit new spectra* refits every spectrum you click on.
- **Export.** *Export spectrum (txt)...* saves the raw spectrum, the baseline, the corrected spectrum and the model, with the fit results in the header.
- **Several components with the optical-depth model.** The components are simply added together, with no radiative transfer between them.

### Line markers

Under *Line markers*, choose a source:
- **Offline catalog** (`line_catalog.json`);
- **Splatalogue (online)**, which needs an internet connection and `astroquery`.

Markers are placed for the source velocity *Vsys*. If you leave Vsys empty, the program uses the fitted V, or failing that the velocity of the brightest channel in view. *Eu max* hides transitions whose upper level is more energetic than the limit. *Known ISM species only* (Splatalogue) hides species never detected in space.

### Supported data

- **Axes.** The cube needs celestial axes and a velocity or frequency axis; Stokes axes of length 1 are ignored. A rest frequency (`RESTFRQ`/`RESTFREQ`) is needed for hyperfine presets and line markers.
- **Large cubes.** Cubes larger than half of the free RAM are read from disk as needed (memory-mapped) rather than loaded into memory, and every operation on the whole cube works in chunks.
- **Scaled-integer FITS.** Files with `BSCALE`/`BZERO`/`BLANK` (common in GILDAS/CLASS output) are supported. `BLANK` values and GILDAS's undeclared −1000 blanking value become NaN.

## Line catalog

The file [`src/spectral_cube_viewer/data/line_catalog.json`](src/spectral_cube_viewer/data/line_catalog.json) is shipped with the package. It covers 70–950 GHz: TRAO, PMO 13.7 m, IRAM 30 m, APEX and FYST/CCAT. It contains:

- **36 hyperfine presets**:
  - HCN, H13CN and DCN 1-0 to 3-2;
  - C17O 1-0 to 3-2;
  - CN 1-0 to 3-2 and C2H 1-0 to 4-3, one preset per fine-structure group;
  - N2H+ and N2D+ 1-0 to 3-2;
  - o- and p-NH2D at 86/110 and 333 GHz.

  Each preset stores the rest frequencies and relative line strengths of its components.
- **About 500 common lines** used for the offline markers. Among them: CO isotopologues, HCO+, HCN, HNC, N2H+, CS, SiO, SO, HC3N, H2CO, CH3OH, CH3CN, c-C3H2, [CI] and more.

To regenerate or extend it, edit the lists at the top of [`tools/build_line_catalog.py`](tools/build_line_catalog.py) and run:

```bash
pip install -e ".[catalog]"
python tools/build_line_catalog.py
```

The JSON file can also be edited by hand. To use your own catalog without reinstalling, set the environment variable `SPECTRAL_CUBE_VIEWER_CATALOG=/path/to/catalog.json`.

## Development

```bash
git clone https://github.com/jlipinski/spectral-cube-viewer
cd spectral-cube-viewer
pip install -e ".[online,catalog,test]"
QT_QPA_PLATFORM=offscreen pytest            # make test
RUN_NETWORK_TESTS=1 QT_QPA_PLATFORM=offscreen pytest   # also query Splatalogue
```

On Windows PowerShell, set the variable with `$env:QT_QPA_PLATFORM="offscreen"` before running `pytest`. The `Makefile` has shortcuts for Linux and macOS: `make dev`, `make test`, `make run`, `make catalog` and `make build`. GitHub Actions runs the tests on Linux, Windows and macOS.

The code is in `src/spectral_cube_viewer/`:

| Module | Contents |
|---|---|
| `core.py` | Opening cubes, moment 0, region averages, baselines; no Qt needed |
| `fitting.py` | Line models (Gaussian and hyperfine) and least-squares fitting |
| `lines.py` | Offline catalog and Splatalogue queries |
| `gui.py` | The Qt application |

## Acknowledgements

The line data come from:
- the CDMS (Müller et al. 2005, J. Mol. Struct. 742, 215; Endres et al. 2016, J. Mol. Spectrosc. 327, 95);
- the JPL catalog (Pickett et al. 1998, JQSRT 60, 883);
- Splatalogue (NRAO), accessed through astroquery (Ginsburg et al. 2019, AJ 157, 98);
- the hyperfine tables shipped with pyspeckit (Ginsburg & Mirocha 2011) for N2H+, N2D+ (Caselli et al. 1995; Pagani et al. 2009) and NH2D.

Cubes are read with astropy and spectral-cube.

## License

MIT; see [LICENSE](LICENSE).
