"""Spectral line models (Gaussians, hyperfine multiplets) and least-squares fitting (no GUI)."""

import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

_trapezoid = getattr(np, 'trapezoid', None) or np.trapz

# Line models
# ----------------------------------------------------------------------------------------------

_FOUR_LN2 = 4 * np.log(2)

MODEL_GAUSS, MODEL_HFS_FIXED, MODEL_HFS_FREE, MODEL_HFS_TAU = 'gauss', 'hfs_fixed', 'hfs_free', 'hfs_tau'
MODEL_LABELS = {
    MODEL_GAUSS: 'Gaussians (independent)',
    MODEL_HFS_FIXED: 'HFS - fixed LTE ratios (thin)',
    MODEL_HFS_FREE: 'HFS - tied V/FWHM, free intensities',
    MODEL_HFS_TAU: 'HFS - with optical depth',
}


def gaussian(v, amp, v0, fwhm):
    return amp * np.exp(-_FOUR_LN2 * ((v - v0) / fwhm) ** 2)


class LineModel:
    """
    Sum of `ncomp` velocity components. Each component is a single Gaussian or a hyperfine
    pattern (lines at fixed velocity offsets sharing one velocity and FWHM):

    gauss      T, V, FWHM
    hfs_fixed  T, V, FWHM                 line i peak = T * r_i   (r normalised to the strongest)
    hfs_free   T(line 1..n), V, FWHM      independent line intensities
    hfs_tau    A, V, FWHM, tau_main       T(v) = A * (1 - exp(-tau(v))), tau_i = tau_main * r_i
    """

    def __init__(self, kind, ncomp, hfs=None):
        self.kind, self.ncomp, self.hfs = kind, ncomp, hfs
        if kind == MODEL_GAUSS:
            self.offsets, self.ratios, self.labels = np.zeros(1), np.ones(1), ['']
        else:
            self.offsets = np.asarray(hfs['offsets'], dtype=float)
            strengths = np.asarray(hfs['strengths'], dtype=float)
            self.ratios = strengths / strengths.max()
            self.labels = hfs.get('labels') or [f'hf{i + 1}' for i in range(len(strengths))]
        if kind == MODEL_HFS_FREE:
            self.par_names = [f'T({lab})' for lab in self.labels] + ['V', 'FWHM']
        elif kind == MODEL_HFS_TAU:
            self.par_names = ['A', 'V', 'FWHM', 'tau_main']
        else:
            self.par_names = ['T', 'V', 'FWHM']
        self.nper = len(self.par_names)
        self.npar = self.nper * ncomp
        self.iv, self.ifwhm = self.par_names.index('V'), self.par_names.index('FWHM')

    def _pattern(self, v, amps, v0, fwhm):
        v = np.asarray(v, dtype=float)
        dv = (v[None, :] - v0 - self.offsets[:, None]) / fwhm
        return (np.asarray(amps)[:, None] * np.exp(-_FOUR_LN2 * dv ** 2)).sum(axis=0)

    def group_profile(self, v, p):
        if self.kind == MODEL_GAUSS:
            return gaussian(np.asarray(v, dtype=float), *p)
        if self.kind == MODEL_HFS_FIXED:
            return self._pattern(v, p[0] * self.ratios, p[1], p[2])
        if self.kind == MODEL_HFS_FREE:
            return self._pattern(v, p[:-2], p[-2], p[-1])
        a, v0, fwhm, tau = p
        return -a * np.expm1(-self._pattern(v, tau * self.ratios, v0, fwhm))

    def split(self, params):
        return np.asarray(params, dtype=float).reshape(self.ncomp, self.nper)

    def groups(self, v, params):
        return [self.group_profile(v, p) for p in self.split(params)]

    def __call__(self, v, *params):
        return np.sum(self.groups(v, params), axis=0)

    def line_positions(self, p):
        return p[self.iv] + self.offsets

    def line_heights(self, p):
        if self.kind in (MODEL_GAUSS, MODEL_HFS_FIXED):
            return p[0] * self.ratios
        if self.kind == MODEL_HFS_FREE:
            return np.asarray(p[:-2])
        return -p[0] * np.expm1(-p[3] * self.ratios)

    def initial_params(self, guesses):
        """Convert (T_peak, V, FWHM) guesses (T of the strongest line) into model parameters."""
        p0 = []
        for t, v0, fwhm in guesses:
            if self.kind == MODEL_HFS_FREE:
                p0 += list(t * self.ratios) + [v0, fwhm]
            elif self.kind == MODEL_HFS_TAU:
                p0 += [t / -np.expm1(-1.0), v0, fwhm, 1.0]
            else:
                p0 += [t, v0, fwhm]
        return np.array(p0, dtype=float)

    def bounds(self, p0, vrange, chan):
        lo, hi = vrange
        pad = np.max(np.abs(self.offsets))
        lower, upper = [], []
        for name, val in zip(self.par_names * self.ncomp, p0):
            if name == 'V':
                b = (lo - pad, hi + pad)
            elif name == 'FWHM':
                b = (0.5 * chan, max(hi - lo, chan))
            elif name == 'tau_main':
                b = (1e-3, 300.0)
            elif self.kind == MODEL_GAUSS and val < 0:
                b = (-np.inf, 0.0)               # absorption component
            else:
                b = (0.0, np.inf)
            lower.append(b[0])
            upper.append(b[1])
        return np.array(lower), np.array(upper)

    def group_area(self, p):
        """Integrated intensity of one component (all its hyperfine lines) [K km/s]."""
        v0, fwhm = p[self.iv], abs(p[self.ifwhm])
        span = np.max(np.abs(self.offsets)) + 4 * fwhm
        vv = np.linspace(v0 - span, v0 + span, 4001)
        return _trapezoid(self.group_profile(vv, p), vv)


def auto_guess(vel, y, n, model, rms=None):
    """Greedy initial guesses: take the highest (smoothed) residual peak, estimate its FWHM, subtract it, repeat."""
    good = np.isfinite(y)
    order = np.argsort(vel[good])
    vel, y = vel[good][order], y[good][order]
    if vel.size < 3:
        return []
    chan = np.median(np.diff(vel))
    single = LineModel(model.kind, 1, model.hfs)
    resid = np.convolve(y, np.ones(3) / 3, mode='same')
    threshold = 3 * rms if rms else 0.0
    guesses = []
    for _ in range(n):
        i = int(np.argmax(resid))
        t = resid[i]
        if t <= threshold:
            break
        j, k = i, i
        while j > 0 and resid[j] > 0.5 * t:
            j -= 1
        while k < resid.size - 1 and resid[k] > 0.5 * t:
            k += 1
        guess = (t, vel[i], max(vel[k] - vel[j] - chan, 2 * chan))
        guesses.append(guess)
        resid = resid - single(vel, *single.initial_params([guess]))
    return guesses


class FitResult:
    def __init__(self, model, popt, pcov, vrange, rms, nchan, resid):
        self.model, self.popt, self.pcov, self.vrange, self.rms, self.nchan = model, popt, pcov, vrange, rms, nchan
        self.perr = np.sqrt(np.clip(np.diag(pcov), 0, None)) if np.all(np.isfinite(pcov)) \
            else np.full(popt.size, np.nan)
        self.resid_rms = float(np.sqrt(np.mean(resid ** 2)))
        dof = max(nchan - popt.size, 1)
        self.chi2r = float(np.sum((resid / rms) ** 2) / dof) if rms else None

    def group_area(self, g):
        """(value, error) of the integrated intensity of component g, error from the covariance."""
        m, n = self.model, self.model.nper
        p = self.popt[g * n:(g + 1) * n]
        area = m.group_area(p)
        cov = self.pcov[g * n:(g + 1) * n, g * n:(g + 1) * n]
        if not np.all(np.isfinite(cov)):
            return area, np.nan
        jac = np.empty(n)
        for i in range(n):
            h = 1e-6 * max(abs(p[i]), 1.0)
            dp = p.copy()
            dp[i] += h
            jac[i] = (m.group_area(dp) - area) / h
        return area, float(np.sqrt(max(jac @ cov @ jac, 0.0)))

    def report(self):
        m = self.model
        lines = [f'Model: {MODEL_LABELS[m.kind]}, {m.ncomp} component(s)',
                 f'Fit range: {self.vrange[0]:.2f} .. {self.vrange[1]:.2f} km/s ({self.nchan} channels)']
        if self.rms:
            lines.append(f'Noise rms (baseline windows): {self.rms:.4g} K')
        else:
            lines.append('Noise rms unknown - errors scaled by the residuals (define baseline windows)')
        lines.append(f'Residual rms: {self.resid_rms:.4g} K'
                     + (f',  reduced chi2: {self.chi2r:.3g}' if self.chi2r is not None else ''))
        units = {'V': 'km/s', 'FWHM': 'km/s', 'tau_main': ''}
        for g in range(m.ncomp):
            lines.append(f'\nComponent {g + 1}')
            sl = slice(g * m.nper, (g + 1) * m.nper)
            for name, val, err in zip(m.par_names, self.popt[sl], self.perr[sl]):
                lines.append(f'  {name:>12s} = {val:10.4f} +/- {err:<8.3g} {units.get(name, "K")}')
            area, aerr = self.group_area(g)
            lines.append(f'  {"int T dv":>12s} = {area:10.4f} +/- {aerr:<8.3g} K km/s')
        return '\n'.join(lines)


def fit_lines(vel, y, model, guesses, vrange, rms=None):
    """Least-squares fit of `model` to the channels vel/y (already restricted to the fit range)."""
    if vel.size <= model.npar:
        raise ValueError(f'Only {vel.size} channels in the fit range for {model.npar} free parameters.')
    chan = np.median(np.abs(np.diff(vel)))
    p0 = model.initial_params(guesses)
    lower, upper = model.bounds(p0, vrange, chan)
    p0 = np.clip(p0, lower, upper)
    sigma = np.full(vel.size, rms) if rms else None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', OptimizeWarning)
        popt, pcov = curve_fit(model, vel, y, p0=p0, bounds=(lower, upper), sigma=sigma,
                               absolute_sigma=bool(rms), x_scale='jac', max_nfev=20000)
    # order components by velocity
    order = np.argsort(model.split(popt)[:, model.iv])
    perm = (order[:, None] * model.nper + np.arange(model.nper)).ravel()
    popt, pcov = popt[perm], pcov[np.ix_(perm, perm)]
    return FitResult(model, popt, pcov, tuple(vrange), rms, vel.size, y - model(vel, *popt))
