"""
Line catalogs: the offline catalog (data/line_catalog.json, hyperfine presets and common lines) and
online Splatalogue queries (needs astroquery and an internet connection).
"""

import json
import os
import re
import warnings

import numpy as np
from astropy import units as u

C_KMS = 299792.458
# the catalog shipped with the package; SPECTRAL_CUBE_VIEWER_CATALOG points to a user-supplied one instead
CATALOG_PATH = os.environ.get('SPECTRAL_CUBE_VIEWER_CATALOG') or \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'line_catalog.json')

# used when line_catalog.json is missing (CDMS frequencies [MHz], LTE relative strengths)
_BUILTIN_HFS = {
    'HCN 1-0': dict(source='built-in (CDMS)', components=[
        dict(label='F=1-1', freq_mhz=88630.4156, strength=0.6),
        dict(label='F=2-1', freq_mhz=88631.8475, strength=1.0),
        dict(label='F=0-1', freq_mhz=88633.9357, strength=0.2)]),
}


def load_line_catalog(path=CATALOG_PATH):
    """(HFS presets, line list, error message or None) from line_catalog.json."""
    try:
        with open(path, encoding='utf-8') as f:
            catalog = json.load(f)
        hfs = catalog.get('hfs') or {}
        if not hfs:
            raise ValueError('no "hfs" presets')
        return hfs, catalog.get('lines') or [], None
    except Exception as e:
        return dict(_BUILTIN_HFS), [], f'{os.path.basename(path)} not loaded ({e}); only the built-in HCN 1-0 preset'


HFS_PRESETS, CATALOG_LINES, CATALOG_ERROR = load_line_catalog()


def preset_reference(preset, rest_freq_mhz=None):
    """
    Frequency [MHz] the HFS velocity offsets are measured from: the cube's rest frequency when it lies
    within the multiplet (so the fitted V is the systemic velocity), else the strongest component.
    Returns (frequency, uses_cube_rest_freq).
    """
    freqs = [c['freq_mhz'] for c in preset['components']]
    if rest_freq_mhz and min(freqs) - 2.0 <= rest_freq_mhz <= max(freqs) + 2.0:
        return rest_freq_mhz, True
    return max(preset['components'], key=lambda c: c['strength'])['freq_mhz'], False


def preset_hfs(preset, ref_mhz):
    """Preset as velocity offsets [km/s] (radio convention) relative to ref_mhz."""
    comps = preset['components']
    return dict(offsets=[C_KMS * (ref_mhz - c['freq_mhz']) / ref_mhz for c in comps],
                strengths=[c['strength'] for c in comps], labels=[c['label'] for c in comps])


def find_preset(rest_freq_mhz):
    """Name of the HFS preset matching a cube's rest frequency, or None."""
    if not rest_freq_mhz:
        return None
    best, best_dist = None, None
    for name, preset in HFS_PRESETS.items():
        if preset_reference(preset, rest_freq_mhz)[1]:
            dist = min(abs(c['freq_mhz'] - rest_freq_mhz) for c in preset['components'])
            if best is None or dist < best_dist:
                best, best_dist = name, dist
    return best


def catalog_lines(eu_max):
    """Offline catalog lines with upper-level energy <= eu_max [K]."""
    return [dict(line) for line in CATALOG_LINES if line.get('eu_k') is None or line['eu_k'] <= eu_max]


def splatalogue_lines(fmin_mhz, fmax_mhz, eu_max, known_only=True):
    """Lines between fmin and fmax [MHz] from Splatalogue (CDMS + JPL). Needs internet and astroquery."""
    try:
        from astroquery.splatalogue import Splatalogue
    except ImportError:
        raise RuntimeError('astroquery is not installed (pip install astroquery)')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            table = Splatalogue.query_lines(fmin_mhz * u.MHz, fmax_mhz * u.MHz, line_lists=['CDMS', 'JPL'],
                                            energy_max=eu_max, energy_type='eu_k',
                                            only_astronomically_observed=known_only, export_stop=3000)
        except ValueError as e:
            if 'No matching' in str(e):
                return []
            raise
    strip = lambda s: re.sub(r'<[^>]+>', '', str(s)).strip()
    lines = []
    for r in table:
        freq = float(r['orderedfreq'])
        if not np.isfinite(freq):
            continue
        line = dict(species=strip(r['name']), transition=re.sub(r'\s+', '', strip(r['resolved_QNs'])),
                    freq_mhz=freq, eu_k=float(r['upper_state_energy_K']), source=str(r['linelist']))
        dup = next((x for x in lines if x['species'] == line['species'] and abs(x['freq_mhz'] - freq) < 0.05), None)
        if dup is None:
            lines.append(line)
        elif dup['source'] != 'CDMS' and line['source'] == 'CDMS':
            dup.update(line)
    return lines
