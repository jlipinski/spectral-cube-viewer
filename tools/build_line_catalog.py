"""
Build src/spectral_cube_viewer/data/line_catalog.json - the offline line data used by Spectral Cube Viewer.

    pip install -e ".[catalog]"
    python tools/build_line_catalog.py [output.json]     (needs internet, astroquery and pyspeckit)

The catalog has two parts:

* "hfs":   hyperfine multiplets used as fitting presets. Each entry lists the components'
           rest frequencies [MHz] and relative line strengths (Sij*mu^2, or the published relative
           intensities). Spectral Cube Viewer converts them to velocity offsets relative to the cube's
           rest frequency, so the fitted V is the systemic velocity.
* "lines": common single transitions (one entry per rotational line), used to mark lines on
           spectra without an internet connection.

Coverage: 70-950 GHz, i.e. TRAO and PMO 13.7 m (85-116 GHz), IRAM 30 m (3/2/1/0.8 mm),
APEX (to ~950 GHz) and FYST/CCAT (CHAI ~455-495 and ~800-860 GHz).

To add a transition, append it to HFS_SPLATALOGUE / HFS_PYSPECKIT / COMMON_SPECIES below and
re-run, or edit line_catalog.json by hand (same structure). Species keys are Splatalogue's names,
see astroquery.splatalogue.Splatalogue.get_species_ids().
"""

import datetime
import json
import os
import re
import sys
import warnings

import numpy as np
from astropy import units as u

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                   'src', 'spectral_cube_viewer', 'data', 'line_catalog.json')
BAND = (70.0, 950.0)            # GHz
MIN_REL_STRENGTH = 2e-3         # drop hyperfine components weaker than this (relative to the strongest)

HCN, H13CN, DCN = '02701 HCN v=0 - Hydrogen Cyanide', '02801 H13CN v = 0 - Hydrogen Cyanide', '02805 DCN v = 0 - Hydrogen Cyanide'
C17O, CN, CCH = '02913 C17O - Carbon Monoxide', '02601 CN v = 0  - Cyanide Radical', '02501 CCH v = 0 - Ethynyl'

# (preset name, Splatalogue species, fmin GHz, fmax GHz) - CDMS hyperfine components inside the window
HFS_SPLATALOGUE = [
    ('HCN 1-0', HCN, 88.62, 88.64),
    ('HCN 2-1', HCN, 177.25, 177.27),
    ('HCN 3-2', HCN, 265.88, 265.895),
    ('H13CN 1-0', H13CN, 86.33, 86.35),
    ('H13CN 2-1', H13CN, 172.67, 172.69),
    ('H13CN 3-2', H13CN, 259.00, 259.02),
    ('DCN 1-0', DCN, 72.40, 72.42),
    ('DCN 2-1', DCN, 144.82, 144.84),
    ('DCN 3-2', DCN, 217.23, 217.25),
    ('C17O 1-0', C17O, 112.35, 112.37),
    ('C17O 2-1', C17O, 224.70, 224.72),
    ('C17O 3-2', C17O, 337.05, 337.07),
    ('CN 1-0 J=1/2-1/2', CN, 113.11, 113.20),
    ('CN 1-0 J=3/2-1/2', CN, 113.48, 113.53),
    ('CN 2-1 J=3/2-1/2', CN, 226.28, 226.37),
    ('CN 2-1 J=5/2-3/2', CN, 226.86, 226.91),
    ('CN 3-2 J=5/2-3/2', CN, 339.99, 340.05),
    ('CN 3-2 J=7/2-5/2', CN, 340.24, 340.27),
    ('C2H 1-0 J=3/2-1/2', CCH, 87.28, 87.33),
    ('C2H 1-0 J=1/2-1/2', CCH, 87.40, 87.45),
    ('C2H 2-1 J=5/2-3/2', CCH, 174.65, 174.68),
    ('C2H 2-1 J=3/2-1/2', CCH, 174.71, 174.74),
    ('C2H 3-2 J=7/2-5/2', CCH, 261.99, 262.01),
    ('C2H 3-2 J=5/2-3/2', CCH, 262.06, 262.08),
    ('C2H 4-3 J=9/2-7/2', CCH, 349.33, 349.345),
    ('C2H 4-3 J=7/2-5/2', CCH, 349.39, 349.41),
]

# (preset name, pyspeckit model module, key prefix) - published tables shipped with pyspeckit
# (N2H+/N2D+: Caselli et al. 1995, Pagani et al. 2009; NH2D: Coudert & Roueff 2006)
HFS_PYSPECKIT = [
    ('N2H+ 1-0', 'n2hp', 'J1-0'),
    ('N2H+ 2-1', 'n2hp', 'J2-1'),
    ('N2H+ 3-2', 'n2hp', 'J3-2'),
    ('N2D+ 1-0', 'n2dp', 'J1-0'),
    ('N2D+ 2-1', 'n2dp', 'J2-1'),
    ('N2D+ 3-2', 'n2dp', 'J3-2'),
    ('o-NH2D 1(1,1)-1(0,1)', 'nh2d', 'o-1_11-1_01'),
    ('p-NH2D 1(1,1)-1(0,1)', 'nh2d', 'p-1_11-1_01'),
    ('o-NH2D 1(0,1)-0(0,0)', 'nh2d', 'o-1_01-0_00'),
    ('p-NH2D 1(0,1)-0(0,0)', 'nh2d', 'p-1_01-0_00'),
]

# (Splatalogue species, label, max upper-level energy [K]) - single lines for offline line markers
COMMON_SPECIES = [
    ('02812 CO v = 0 - Carbon Monoxide', 'CO', 160),
    ('02910 13CO v = 0 - Carbon Monoxide', '13CO', 160),
    ('03005 C18O - Carbon Monoxide', 'C18O', 160),
    ('02913 C17O - Carbon Monoxide', 'C17O', 160),
    ('02915 HCO+ v=0 - Formylium', 'HCO+', 120),
    ('03007 H13CO+ - Formylium', 'H13CO+', 120),
    ('03102 HC18O+ - Formylium', 'HC18O+', 60),
    ('03009 DCO+ v = 0 - Formylium', 'DCO+', 80),
    (HCN, 'HCN', 120),
    (H13CN, 'H13CN', 80),
    (DCN, 'DCN', 80),
    ('02712 HNC v=0 - Hydrogen Isocyanide', 'HNC', 120),
    ('02807 HN13C - Hydrogen Isocyanide', 'HN13C', 80),
    ('02809 DNC - Hydrogen Isocyanide', 'DNC', 80),
    ('02927 N2H+ v = 0 - Diazenylium', 'N2H+', 120),
    ('03022 N2D+ - Diazenylium', 'N2D+', 80),
    ('04413 CS v = 0  - Carbon Monosulfide', 'CS', 120),
    ('04604 C34S v = 0 - Carbon Monosulfide', 'C34S', 80),
    ('04504 13CS v = 0 - Carbon Monosulfide', '13CS', 60),
    ('04425 SiO v = 0  - Silicon Monoxide', 'SiO', 120),
    ('04811 SO 3Σ v = 0 - Sulfur Monoxide', 'SO', 80),
    ('06001 OCS v = 0 - Carbonyl Sulfide', 'OCS', 60),
    ('05101 HC3N v=0 - Cyanoacetylene', 'HC3N', 100),
    ('05610 CCS - Thioxoethenylidene', 'CCS', 30),
    ('03023 H2CO - Formaldehyde', 'H2CO', 80),
    ('04623 H2CS - Thioformaldehyde', 'H2CS', 40),
    ('03207 CH3OH vt = 0-2 - Methanol', 'CH3OH', 40),
    ('04107 CH3CN v = 0  - Methyl Cyanide', 'CH3CN', 80),
    ('04008 CH3CCH v = 0 - Methyl Acetylene', 'CH3CCH', 50),
    ('03813 c-HCCCH  v=0  - Cyclopropenylidene', 'c-C3H2', 30),
    ('04304 HNCO v=0 - Isocyanic Acid', 'HNCO', 50),
    ('03406 H2S - Hydrogen sulfide', 'H2S', 60),
    ('06409 SO2 v = 0 - Sulfur dioxide', 'SO2', 30),
    (CN, 'CN', 60),
    (CCH, 'C2H', 60),
    ('01812 NH2D - Ammonia', 'NH2D', 60),
]

# Atomic fine-structure lines (not in the molecular catalogs). CDMS / NIST values.
MANUAL_LINES = [
    dict(species='[CI]', transition='3P1-3P0', freq_mhz=492160.651, eu_k=23.6, source='CDMS'),
    dict(species='[CI]', transition='3P2-3P1', freq_mhz=809341.970, eu_k=62.5, source='CDMS'),
]

_QN_LABEL = re.compile(r'([A-Za-z]+)\s*(?:<sub>\s*(\d)\s*</sub>)?\s*=')


def _strip(text):
    return re.sub(r'<[^>]+>', '', str(text)).strip()


def _qn_labels(qns):
    """{'J', 'F', 'F1', ...} quantum-number names in a Splatalogue resolved_QNs string."""
    return {name + (sub or '') for name, sub in _QN_LABEL.findall(str(qns))}


def _splat(species, fmin_ghz, fmax_ghz, **kwargs):
    from astroquery.splatalogue import Splatalogue
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Splatalogue.query_lines(fmin_ghz * u.GHz, fmax_ghz * u.GHz, chemical_name=re.escape(species),
                                       export_stop=5000, **kwargs)


def _finish_components(comps):
    """Merge coincident components (same frequency within 1 kHz), drop very weak ones, sort by frequency."""
    comps = sorted(comps, key=lambda c: c['freq_mhz'])
    merged = []
    for c in comps:
        if merged and abs(c['freq_mhz'] - merged[-1]['freq_mhz']) < 1e-3:
            merged[-1]['strength'] += c['strength']
            if c['label'] not in merged[-1]['label'].split(' + '):
                merged[-1]['label'] += ' + ' + c['label']
        else:
            merged.append(dict(c))
    smax = max(c['strength'] for c in merged)
    return [dict(label=c['label'], freq_mhz=round(c['freq_mhz'], 5), strength=round(c['strength'] / smax, 6))
            for c in merged if c['strength'] >= MIN_REL_STRENGTH * smax]


def hfs_from_splatalogue(species, fmin, fmax):
    """CDMS hyperfine components in a window, at the coarsest level that resolves the hyperfine structure."""
    t = _splat(species, fmin, fmax, line_lists=['CDMS'])
    rows = [r for r in t if 'F' in _qn_labels(r['resolved_QNs'])]
    if not rows:
        raise ValueError(f'no hyperfine components for {species} in {fmin}-{fmax} GHz')
    # e.g. H13CN lists both N-hyperfine (F) and finer H/13C splittings (F1, F2): keep the coarsest level
    nlab = min(len(_qn_labels(r['resolved_QNs'])) for r in rows)
    rows = [r for r in rows if len(_qn_labels(r['resolved_QNs'])) == nlab]
    newest = {}                                  # duplicated entries from older catalog versions
    for r in rows:
        key = _strip(r['resolved_QNs']).replace(' ', '')
        if key not in newest or r['lineid'] > newest[key]['lineid']:
            newest[key] = r
    comps = []
    for qns, r in newest.items():
        label = ', '.join(p.strip() for p in _strip(r['resolved_QNs']).split(',') if p.strip().startswith('F'))
        comps.append(dict(label=label.replace(' ', ''), freq_mhz=float(r['orderedfreq']), strength=float(r['sijmu2'])))
    return _finish_components(comps)


def hfs_from_pyspeckit(module, prefix):
    import importlib
    mod = importlib.import_module(f'pyspeckit.spectrum.models.{module}')
    keys = [k for k in mod.freq_dict if k.startswith(prefix + '_')]
    if not keys:
        raise ValueError(f'{prefix} not in pyspeckit {module}')
    comps = [dict(label=f'{float(mod.freq_dict[k]) / 1e6:.4f}', freq_mhz=float(mod.freq_dict[k]) / 1e6,
                  strength=float(mod.line_strength_dict[k])) for k in keys]
    return _finish_components(comps)


_F_TERM = re.compile(r',?\s*F\s*\d?\s*=\s*[\d/]+\s*-\s*[\d/]+')


def common_lines(species, label, eu_max, cluster_mhz=1.5, hfs_span_mhz=15.0):
    """
    One entry per line below eu_max in the band. Hyperfine components are replaced by the unsplit line
    (or by their strongest component when the catalog has no unsplit entry), and duplicate catalog
    entries closer than cluster_mhz are merged (preferring CDMS).
    """
    t = _splat(species, *BAND, line_lists=['CDMS', 'JPL'], energy_max=eu_max, energy_type='eu_k')
    plain, hyperfine = [], {}
    for r in t:
        freq = float(r['orderedfreq'])
        if not np.isfinite(freq):
            continue
        transition = re.sub(r'\s+', '', _F_TERM.sub('', _strip(r['resolved_QNs']))).strip(',')
        row = dict(species=label, transition=transition, freq_mhz=round(freq, 4),
                   eu_k=round(float(r['upper_state_energy_K']), 2), source=str(r['linelist']),
                   _rank=(r['linelist'] != 'CDMS', -float(r['sijmu2'] or 0)))
        if any(lab.startswith('F') for lab in _qn_labels(r['resolved_QNs'])):
            hyperfine.setdefault(transition, []).append(row)
        else:
            plain.append(row)
    for transition, comps in hyperfine.items():
        strongest = min(comps, key=lambda c: c['_rank'])
        if not any(p['transition'] == transition and abs(p['freq_mhz'] - strongest['freq_mhz']) < hfs_span_mhz
                   for p in plain):
            plain.append(strongest)
    plain.sort(key=lambda c: c['freq_mhz'])
    clusters = []
    for row in plain:
        if clusters and row['freq_mhz'] - clusters[-1][-1]['freq_mhz'] < cluster_mhz:
            clusters[-1].append(row)
        else:
            clusters.append([row])
    out = []
    for cl in clusters:
        best = dict(min(cl, key=lambda c: c['_rank']))
        best.pop('_rank')
        out.append(best)
    return out


def main(out=OUT):
    catalog = dict(
        description='Offline line data for Spectral Cube Viewer, generated by tools/build_line_catalog.py. '
                    'hfs: hyperfine presets (component rest frequencies in MHz, relative strengths); '
                    'lines: common transitions for offline line markers. May be edited by hand.',
        generated=datetime.date.today().isoformat(),
        sources=['CDMS (Mueller et al. 2005; Endres et al. 2016) via Splatalogue',
                 'JPL (Pickett et al. 1998) via Splatalogue',
                 'pyspeckit hyperfine tables (N2H+, N2D+, NH2D)'],
        hfs={}, lines=[])
    for name, species, fmin, fmax in HFS_SPLATALOGUE:
        try:
            comps = hfs_from_splatalogue(species, fmin, fmax)
        except Exception as e:
            print(f'!! {name}: {e}')
            continue
        catalog['hfs'][name] = dict(source='CDMS via Splatalogue', components=comps)
        print(f'{name:22s} {len(comps):2d} components')
    for name, module, prefix in HFS_PYSPECKIT:
        try:
            comps = hfs_from_pyspeckit(module, prefix)
        except Exception as e:
            print(f'!! {name}: {e}')
            continue
        catalog['hfs'][name] = dict(source=f'pyspeckit ({module})', components=comps)
        print(f'{name:22s} {len(comps):2d} components')
    for species, label, eu_max in COMMON_SPECIES:
        try:
            lines = common_lines(species, label, eu_max)
        except Exception as e:
            print(f'!! {label}: {e}')
            continue
        catalog['lines'] += lines
        print(f'{label:8s} {len(lines):3d} lines (Eu < {eu_max} K)')
    catalog['lines'] += MANUAL_LINES
    catalog['lines'].sort(key=lambda c: c['freq_mhz'])
    catalog['hfs'] = dict(sorted(catalog['hfs'].items(),
                                 key=lambda kv: max(kv[1]['components'], key=lambda c: c['strength'])['freq_mhz']))
    out = os.path.normpath(out)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=1, ensure_ascii=False)
    print(f'\nWrote {out}: {len(catalog["hfs"])} HFS presets, {len(catalog["lines"])} lines')


if __name__ == '__main__':
    main(*sys.argv[1:2])
