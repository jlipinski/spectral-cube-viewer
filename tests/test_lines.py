import os

import numpy as np
import pytest

from spectral_cube_viewer.lines import (CATALOG_ERROR, CATALOG_LINES, HFS_PRESETS, catalog_lines, find_preset,
                                  load_line_catalog, preset_hfs, preset_reference, splatalogue_lines)


def test_packaged_catalog_loads():
    assert CATALOG_ERROR is None
    assert len(HFS_PRESETS) >= 30 and len(CATALOG_LINES) > 100
    for name, preset in HFS_PRESETS.items():
        comps = preset['components']
        assert comps, name
        assert max(c['strength'] for c in comps) == pytest.approx(1.0), name
        assert all(c['freq_mhz'] > 0 for c in comps), name


def test_missing_catalog_falls_back_to_builtin(tmp_path):
    hfs, lines, error = load_line_catalog(str(tmp_path / 'nope.json'))
    assert 'HCN 1-0' in hfs and lines == [] and error


def test_hcn_relative_strengths_and_offsets():
    preset = HFS_PRESETS['HCN 1-0']
    s = sorted(c['strength'] for c in preset['components'])
    np.testing.assert_allclose(np.array(s) / s[-1] * 5, [1, 3, 5], rtol=1e-3)     # LTE 1:3:5
    hfs = preset_hfs(preset, 88631.8475)
    np.testing.assert_allclose(sorted(hfs['offsets']), [-7.065, 0.0, 4.844], atol=2e-3)


def test_offsets_follow_cube_rest_frequency():
    preset = HFS_PRESETS['HCN 1-0']
    # cube rest frequency = unsplit J=1-0 line: the F=2-1 component is 0.83 km/s away
    ref, from_cube = preset_reference(preset, 88631.6023)
    assert from_cube and ref == 88631.6023
    offsets = preset_hfs(preset, ref)['offsets']
    assert min(abs(o + 0.83) for o in offsets) < 0.01
    ref, from_cube = preset_reference(preset, 110201.354)   # 13CO cube: not in the multiplet
    assert not from_cube


def test_find_preset():
    assert find_preset(88631.8473) == 'HCN 1-0'
    assert find_preset(93173.7637) == 'N2H+ 1-0'
    assert find_preset(113490.9702) == 'CN 1-0 J=3/2-1/2'
    assert find_preset(89188.526) is None          # HCO+ has no hyperfine structure
    assert find_preset(None) is None


def test_catalog_lines_filter():
    low = catalog_lines(20)
    assert low and all(line['eu_k'] is None or line['eu_k'] <= 20 for line in low)
    assert any(line['species'] == 'CO' and line['transition'].endswith('1-0') for line in low)


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get('RUN_NETWORK_TESTS'), reason='set RUN_NETWORK_TESTS=1 to query Splatalogue')
def test_splatalogue_finds_hcop():
    pytest.importorskip('astroquery')
    lines = splatalogue_lines(89180, 89200, 100)
    assert any('HCO+' in line['species'] for line in lines)
