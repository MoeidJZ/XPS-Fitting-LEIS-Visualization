"""
lineshapes.py — lmfit-backed XPS peak line-shape models for the XPS Analysis Tool.

This module replaces the former CasaXPS GL / SGL / LA / LF engine entirely.
Every shape now maps directly onto an lmfit model, so the GUI can run true
least-squares fits (Levenberg-Marquardt) and read physically meaningful
parameters back out.
"""

import re
import numpy as np
from lmfit.models import (GaussianModel, LorentzianModel, VoigtModel,
                          PseudoVoigtModel, DoniachModel, SkewedVoigtModel)

# ── family registry ───────────────────────────────────────────────────────────
FAMILIES = ('G', 'L', 'V', 'PV', 'DS', 'SV')

FAMILY_LABELS = {
    'G':  'Gaussian',
    'L':  'Lorentzian',
    'V':  'Voigt',
    'PV': 'pseudo-Voigt',
    'DS': 'Doniach-Sunjic',
    'SV': 'skewed Voigt',
}

# Extra shape parameters carried inside the shape string.
EXTRA_PARAMS = {
    'G':  [],
    'L':  [],
    'V':  [('gamma',    0.30, 0.0, 1e4)],
    'PV': [('fraction', 0.50, 0.0, 1.0)],
    'DS': [('gamma',    0.05, 0.0, 1.0)],
    'SV': [('gamma',    0.30, 0.0, 1e4), ('skew', 0.0, -10.0, 10.0)],
}

_MODELS = {
    'G':  GaussianModel,  'L':  LorentzianModel, 'V':  VoigtModel,
    'PV': PseudoVoigtModel, 'DS': DoniachModel,  'SV': SkewedVoigtModel,
}

GAMMA_TIED_TO_SIGMA = ('V', 'SV')
_LEGACY = {'GL': 'PV', 'SGL': 'PV', 'LA': 'DS', 'LF': 'DS'}
_NUM = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

# ── string helpers ─────────────────────────────────────────────────────────────
def split_lineshape(s):
    s = (s or '').strip()
    m = re.match(r'\s*([A-Za-z]+)', s)
    fam = m.group(1).upper() if m else 'G'
    if fam not in FAMILIES:
        fam = _LEGACY.get(fam, 'G')
    nums = [float(x) for x in re.findall(_NUM, s)]
    return fam, nums

def parse_lineshape(s):
    fam, nums = split_lineshape(s)
    extras = {}
    for i, (name, default, lo, hi) in enumerate(EXTRA_PARAMS[fam]):
        extras[name] = nums[i] if i < len(nums) else default
    return {'family': fam, 'extras': extras}

def shape_string(family, extras=None):
    family = family if family in FAMILIES else 'G'
    spec = EXTRA_PARAMS[family]
    if not spec:
        return family
    extras = extras or {}
    vals = [extras.get(name, default) for (name, default, lo, hi) in spec]
    return f"{family}({','.join(f'{v:g}' for v in vals)})"

def extra_param_names(family):
    return [name for (name, *_rest) in EXTRA_PARAMS.get(family, [])]

# ── lmfit model construction ───────────────────────────────────────────────────
def make_model(family, prefix=''):
    family = family if family in FAMILIES else 'G'
    return _MODELS[family](prefix=prefix)

# ── core evaluation ────────────────────────────────────────────────────────────
def _evaluate_model(family, x, center, sigma, amplitude, extras):
    """Consolidated math: Evaluates the line shape natively through lmfit."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    model = make_model(family, prefix='')
    params = model.make_params()
    
    params.add('center', value=float(center))
    params.add('amplitude', value=float(amplitude))
    params.add('sigma', value=max(float(sigma), 1e-6))
    
    if 'gamma' in extras:
        g_max = 1.0 if family == 'DS' else None
        params.add('gamma', value=max(float(extras['gamma']), 1e-6), min=0.0, max=g_max)
    if 'fraction' in extras:
        params.add('fraction', value=float(np.clip(extras['fraction'], 0.0, 1.0)), min=0.0, max=1.0)
    if 'skew' in extras:
        params.add('skew', value=float(extras.get('skew', 0.0)))
        
    return model.eval(params, x=x)

def evaluate(s, x, center, sigma, amplitude):
    p = parse_lineshape(s)
    y = _evaluate_model(p['family'], x, center, sigma, amplitude, p['extras'])
    return float(y[0]) if np.isscalar(x) else y

def fwhm_height(s, center, sigma, amplitude):
    p = parse_lineshape(s)
    s_ = max(float(sigma), 1e-6)
    span = 40.0 * s_
    xx = np.linspace(center - span, center + span, 20001)
    yy = _evaluate_model(p['family'], xx, center, s_, amplitude, p['extras'])
    ymax = float(np.max(yy)) if yy.size else 0.0
    if ymax <= 0:
        return 0.0, max(ymax, 0.0)
    half = ymax / 2.0
    above = np.where(yy >= half)[0]
    if above.size < 2:
        return 0.0, ymax
    fwhm = float(xx[above[-1]] - xx[above[0]])
    return fwhm, ymax

LINE_SHAPES = (
    'G', 'L',
    'V(0.3)', 'V(0.6)',
    'PV(0.3)', 'PV(0.5)', 'PV(0.7)',
    'DS(0.05)', 'DS(0.1)', 'DS(0.2)',
    'SV(0.3,0.3)', 'SV(0.3,-0.3)',
)