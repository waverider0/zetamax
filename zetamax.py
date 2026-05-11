#!/usr/bin/env python3

import math, cmath, random, threading
from fractions import Fraction
from typing import Callable

DURATION = 120
COMPLEX = False

# The naive way to build a math problem generator is rejection sampling:
# roll a random problem, compute the answer, then reject it if the answer
# is not "nice" -- too large, non-integral, ugly radical, ambiguous, etc.
#
# For basic arithmetic this is harmless. Addition, subtraction, and
# multiplication preserve integer/Gaussian-integer niceness automatically.
# Pick legal operands and the answer is legal. There is nothing sparse to
# hit.
#
# Outside basic arithmetic, naive rejection usually fails. "Nice problem"
# and "nice answer" often define two thin subsets whose intersection is
# sparse and irregular. The old eigenvalue approach was the bad pattern:
# sample a random construction, hope the matrix entries stay in bounds,
# then fall back to a hardcoded matrix when the search misses. That causes
# biased output, variable runtime, and mode collapse.
#
# The default for nontrivial algebra should be structure-first generation:
# choose the clean answer or algebraic structure first, then project
# forward to the displayed problem.
#
# Examples:
#   roots       -> choose factors/roots, multiply out the polynomial
#   eigenvalues -> choose trace/determinant or direct matrix normal form
#   inverse     -> choose a unimodular matrix or elementary row operations
#   GCD         -> choose gcd and coprime cofactors
#   Mod         -> choose modulus, remainder, and quotient
#
# This gives the important guarantee:
#
#   Soundness: every emitted problem has the advertised answer.
#
# It often also gives predictable cost, because the generator does not
# depend on accidentally hitting a rare intersection. But this is not a
# religion. Small rejection filters are allowed when they are local,
# measured, and boring:
#
#   - the main structure has already been constructed;
#   - the rejection predicate only removes edge cases;
#   - hit rate is high enough to be practically irrelevant;
#   - the retry count is small and bounded, or the valid candidates are
#     enumerated directly;
#   - failure never falls back to a hardcoded problem.
#
# Rejection is a smoothing tool, not the primary strategy.
#
# Also: constructive does not imply full coverage. Coverage depends on
# the parameterization.
#
# Direct/free-variable modes:
#   Plus, Minus, Times:
#     direct over chosen operands.
#
#   Divide:
#     direct over chosen integer/Gaussian-integer operands; answers may be
#     rational/Gaussian-rational.
#
#   Power:
#     direct over base and positive integer exponent. The answer set is
#     only perfect powers; that is inherent.
#
#   Det, matrix_multiply:
#     direct over chosen 2x2 matrices. Dimension is fixed by design.
#
#   Binomial, FactorialPower:
#     direct over chosen (n, k) domain. The answer values are sparse by
#     nature.
#
#   Mod:
#     direct over (modulus, remainder, quotient). To cover all dividends,
#     the quotient range must include negative values too.
#
# Constructed subfamily modes:
#   GCD(real):
#     chooses g, then finite-enumerates arbitrary coprime cofactors.
#
#   GCD(complex):
#     chooses Gaussian gcd g, then divides random cofactor-candidates by
#     their Gaussian gcd to make them coprime.
#
#   Log:
#     this is an exact discrete-log-style problem: choose base z and
#     exponent n, emit Log[z, z^n]. Exclude 0 and units directly; do not
#     rely on broad rejection.
#
#   Roots:
#     covers the chosen factor families: integer, rational, repeated,
#     Gaussian-pair, and selected surd factors. This is not the full space
#     of all integer polynomials. Add factor families when broader
#     coverage is desired.
#
#   Inverse:
#     integer-valued inverse means determinant is a unit: +-1 in Z, or
#     +-1, +-I in Z[i]. That is not a limitation; it is the exact condition.
#     If rational inverse answers are allowed, use a separate rational mode.
#
#   Eigenvalues:
#     uses upper/lower triangular matrices for distinct real eigenvalues
#     and real 2x2 block form for complex conjugate pairs. Broader coverage
#     can use companion matrices plus bounded unimodular conjugation.
#
# Numeric/grid modes:
#   Sin, Cos, Tan, ArcSin, ArcCos, ArcTan, complex_rotation:
#     sampled on a decimal grid with approximate numeric answers. These
#     are not algebraic full-coverage modes.
#
# Many algebra modes work because there are integer-preserving forward
# maps: factor multiplication, trace/determinant construction,
# unimodular row operations, and matrix multiplication. COMPLEX usually
# lifts these to Gaussian integers, but it does not automatically imply
# identical coverage.
#
# Calculus is different. Constructive generation is still the right
# default -- e.g. choose F and differentiate to make an integral problem.
# But expression growth, simplification, and "niceness" dominate. Randomly
# sampling an integrand and hoping for an elementary antiderivative is the
# same bad sparse-hit pattern as naive eigenvalue rejection.

def _rng(low: int, high: int, imag: bool = True) -> complex:
  real = random.randint(low, high)
  if not imag or not COMPLEX: return complex(real, 0)
  imag_part = random.randint(low, high)
  if real == 0 and imag_part == 0: imag_part = 1
  return complex(real, imag_part)

def _rng_float(low: float, high: float, imag: bool = True) -> complex:
  real = round(random.uniform(low, high), 2)
  if not imag or not COMPLEX: return complex(real, 0)
  imag_part = round(random.uniform(low, high), 2)
  if abs(real) < 1e-9 and abs(imag_part) < 1e-9: imag_part = 1.0
  return complex(real, imag_part)

#
# Parsing and formatting
#

def _strip_outer_parens(value: str) -> str:
  while len(value) >= 2 and value[0] == '(' and value[-1] == ')':
    depth = 0
    wraps = True
    for i, ch in enumerate(value):
      if ch == '(':
        depth += 1
      elif ch == ')':
        depth -= 1
        if depth == 0 and i != len(value) - 1:
          wraps = False
          break
      if depth < 0:
        wraps = False
        break
    if not wraps or depth != 0: break
    value = value[1:-1]
  return value

def _parse_imag_coeff(value: str) -> float:
  if value in ('', '+'): return 1.0
  if value == '-': return -1.0
  return float(Fraction(value))

def parse_complex_from_user(user_input: str) -> complex:
  cleaned = user_input.strip().replace(' ', '').lower()
  cleaned = _strip_outer_parens(cleaned)
  if cleaned in ('i', '+i', 'j', '+j'): return 1j
  if cleaned in ('-i', '-j'): return -1j
  cleaned = cleaned.replace('i', 'j')
  if cleaned in ('j', '+j'): return 1j
  if cleaned == '-j': return -1j
  try:
    return complex(cleaned)
  except ValueError:
    if cleaned.endswith('j'):
      body = cleaned[:-1]
      sep = -1
      for i in range(1, len(body)):
        if body[i] in '+-': sep = i
      if sep != -1:
        real = float(Fraction(body[:sep]))
        imag = _parse_imag_coeff(body[sep:])
        return complex(real, imag)
      return complex(0, _parse_imag_coeff(body))
    return complex(float(Fraction(cleaned)), 0)

def _answer_tokens(user_input: str) -> list[str]:
  tokens: list[str] = []
  current: list[str] = []
  depth = 0
  for ch in user_input.strip():
    if ch in '{[]}':
      if depth == 0 and current:
        tokens.append(''.join(current))
        current = []
      continue
    if ch == '(':
      depth += 1
      current.append(ch)
      continue
    if ch == ')':
      depth = max(0, depth - 1)
      current.append(ch)
      continue
    if depth == 0 and (ch.isspace() or ch in ',;|'):
      if current:
        tokens.append(''.join(current))
        current = []
      continue
    current.append(ch)
  if current: tokens.append(''.join(current))
  return tokens

def _top_level_comma_parts(user_input: str) -> list[str]:
  parts: list[str] = []
  current: list[str] = []
  depth = 0
  for ch in user_input:
    if ch == '(':
      depth += 1
    elif ch == ')':
      depth = max(0, depth - 1)
    if ch == ',' and depth == 0:
      part = ''.join(current).strip()
      if part: parts.append(part)
      current = []
    else:
      current.append(ch)
  part = ''.join(current).strip()
  if part: parts.append(part)
  return parts

def parse_complex_sequence(user_input: str) -> list[complex]:
  if ',' in user_input and '|' not in user_input and '{' not in user_input and '}' not in user_input:
    parts = _top_level_comma_parts(user_input)
  else:
    parts = _answer_tokens(user_input)
  return [parse_complex_from_user(part) for part in parts]

def parse_tuple_answer(user_input: str) -> tuple:
  return tuple(parse_complex_sequence(user_input))

def format_complex_cartesian(real_value: float, imag_value: float) -> str:
  if abs(real_value) < 1e-12: real_value = 0
  if abs(imag_value) < 1e-12: imag_value = 0
  if imag_value == 0: return str(int(real_value)) if real_value == int(real_value) else f'{real_value:.2f}'
  if real_value == 0:
    if imag_value == 1: return 'I'
    if imag_value == -1: return '-I'
    return f'{int(imag_value)}I' if imag_value == int(imag_value) else f'{imag_value:.2f}I'
  sign = '+' if imag_value >= 0 else '-'
  real_string = str(int(real_value)) if real_value == int(real_value) else f'{real_value:.2f}'
  abs_imag = abs(imag_value)
  if abs_imag == 1: return f'({real_string} {sign} I)'
  imag_string = str(int(abs_imag)) if abs_imag == int(abs_imag) else f'{abs_imag:.2f}'
  return f'({real_string} {sign} {imag_string}I)'

def format_complex_result(value: complex) -> str:
  if abs(value.imag) < 1e-9:
    real = value.real
    return str(int(real)) if real == int(real) else f'{real:.2f}'
  if abs(value.real) < 1e-9:
    imag = value.imag
    return f'{int(imag)}I' if imag == int(imag) else f'{imag:.2f}I'
  sign = '+' if value.imag >= 0 else '-'
  real = value.real
  real_str = str(int(real)) if real == int(real) else f'{real:.2f}'
  imag = abs(value.imag)
  imag_str = str(int(imag)) if imag == int(imag) else f'{imag:.2f}'
  return f'{real_str} {sign} {imag_str}I'

def _fmt(z: complex) -> str:
  return format_complex_cartesian(z.real, z.imag)

def _fres(z: complex) -> str:
  return format_complex_result(z)

def _fentry(value: complex) -> str:
  if abs(value.imag) < 1e-9:
    real = value.real
    return str(int(real)) if real == int(real) else f'{real:.2f}'
  return format_complex_result(value)

def format_matrix(rows: list[list]) -> str:
  # {x0 y0 | x1 y1}
  return '{' + ' | '.join(' '.join(_fmt(v) for v in row) for row in rows) + '}'

#
# Generators
#

def Plus() -> tuple[str, complex]:
  z1 = _rng(2, 200)
  z2 = _rng(2, 200)
  return f'{_fmt(z1)} + {_fmt(z2)}', z1 + z2

def Minus() -> tuple[str, complex]:
  z1 = _rng(2, 200)
  z2 = _rng(2, 200)
  return f'{_fmt(z1)} - {_fmt(z2)}', z1 - z2

def Times() -> tuple[str, complex]:
  z1 = _rng(2, 100)
  z2 = _rng(2, 20)
  return f'{_fmt(z1)} * {_fmt(z2)}', z1 * z2

def Divide() -> tuple[str, complex]:
  z1 = _rng(2, 100)
  z2 = _rng(2, 100)
  return f'{_fmt(z1)} / {_fmt(z2)}', z1 / z2

def Power() -> tuple[str, complex]:
  z = _rng(2, 10)
  x = random.randint(2, 8)
  return f'{_fmt(z)}^{x}', z ** x

def Log():
  n = random.randint(2, 6)
  # Exclude 0 (ambiguous: 0^n = 0 for all n) and Gaussian units
  # (i^1 = i^5 = i, making Log ambiguous with multiple integer answers).
  if COMPLEX:
    pool = [complex(r, i) for r in range(-8, 9) for i in range(-8, 9)
            if complex(r, i) != 0 and abs(complex(r, i)) != 1]
  else:
    pool = [complex(r, 0) for r in range(-8, 9) if r not in (-1, 0, 1)]
  z1 = random.choice(pool)
  z2 = z1 ** n
  return f'Log[{_fmt(z1)}, {_fmt(z2)}]', complex(n, 0)

## NT

class _UnitTolerantComplex:
  """Stores the canonical form of a Gaussian integer answer.
  The grading loop normalizes the user's answer before comparison,
  so any associate (+-1, +-i multiples) is accepted."""
  def __init__(self, canonical: complex):
    self.canonical = canonical

class _CongruenceCheck:
  """Stores (a, m) so the grading loop can verify m | (a - answer)
  rather than comparing against a specific remainder."""
  def __init__(self, a: complex, m: complex, remainder: complex):
    self.a = a
    self.m = m
    self.remainder = remainder

def _gaussian_divmod(a: complex, b: complex) -> tuple[complex, complex]:
  q = complex(round((a / b).real), round((a / b).imag))
  return q, a - q * b

def _gaussian_gcd(a: complex, b: complex) -> complex:
  while b != 0:
    _, r = _gaussian_divmod(a, b)
    a, b = b, r
  return a

def _is_gaussian_unit(z: complex) -> bool:
  return abs(abs(z) - 1) < 1e-9

def _gaussian_exact_div(a: complex, b: complex) -> complex:
  """Exact division of Gaussian integers (a/b). Raises if not exact."""
  z = a / b
  q = complex(round(z.real), round(z.imag))
  if abs(a - q * b) > 1e-9:
    raise ArithmeticError(f'non-exact Gaussian division: {a} / {b}')
  return q

def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
  """Returns (g, x, y) such that a*x + b*y = g = gcd(a,b) >= 0."""
  old_r, r = abs(a), abs(b)
  old_s, s = 1, 0
  old_t, t = 0, 1
  while r != 0:
    q = old_r // r
    old_r, r = r, old_r - q * r
    old_s, s = s, old_s - q * s
    old_t, t = t, old_t - q * t
  if a < 0: old_s = -old_s
  if b < 0: old_t = -old_t
  return (old_r, old_s, old_t)

def _gaussian_extended_gcd(a: complex, b: complex
                           ) -> tuple[complex, complex, complex]:
  """Returns (g, x, y) such that a*x + b*y = g = gaussian_gcd(a,b)."""
  old_r, r = a, b
  old_s, s = complex(1, 0), complex(0, 0)
  old_t, t = complex(0, 0), complex(1, 0)
  while abs(r) > 1e-9:
    q, _ = _gaussian_divmod(old_r, r)
    old_r, r = r, old_r - q * r
    old_s, s = s, old_s - q * s
    old_t, t = t, old_t - q * t
  return (old_r, old_s, old_t)

def _canonical_gaussian(z: complex) -> complex:
  # gcd in Z[i] is only defined up to units (+-1, +-i).
  # Normalizing to the first quadrant (re > 0, im >= 0)
  # picks a unique representative for grading.
  if z == 0: return z
  for u in [1, -1, 1j, -1j]:
    w = z * u
    if w.real > 0 and w.imag >= 0: return w
  for u in [1, -1, 1j, -1j]:
    w = z * u
    if w.real >= 0 and w.imag > 0: return w
  if z.real != 0: return complex(abs(z.real), z.imag)
  return complex(z.real, abs(z.imag))

def GCD() -> tuple[str, object]:
  if not COMPLEX:
    g = random.randint(2, 20)
    u = random.randint(1, 15)
    vs = [v for v in range(1, 16) if math.gcd(u, v) == 1 and v != u]
    v = random.choice(vs)
    x = g * u
    y = g * v
    return f'GCD[{x}, {y}]', complex(g, 0)

  # COMPLEX: choose gcd g, then construct coprime cofactors by
  # dividing random cofactor-candidates by their Gaussian gcd.
  g_pool = [complex(r, i) for r in range(-5, 6) for i in range(-5, 6)
            if complex(r, i) != 0 and abs(complex(r, i)) > 1]
  g_raw = random.choice(g_pool)
  g = _canonical_gaussian(g_raw)

  raw_pool = [complex(r, i) for r in range(1, 7) for i in range(-3, 4)]
  pairs = [(a, b) for a in raw_pool for b in raw_pool if a != b]
  a_raw, b_raw = random.choice(pairs)

  h = _gaussian_gcd(a_raw, b_raw)
  a = _gaussian_exact_div(a_raw, h)
  b = _gaussian_exact_div(b_raw, h)
  x = g * a
  y = g * b
  return f'GCD[{_fmt(x)}, {_fmt(y)}]', _UnitTolerantComplex(g)

def Mod() -> tuple[str, object]:
  if not COMPLEX:
    m = random.randint(5, 30)
    r = random.randint(0, m - 1)
    q = random.choice([q for q in range(-30, 31) if q != 0])
    a = q * m + r
    return f'Mod[{a}, {m}]', complex(r, 0)

  m = complex(random.randint(2, 5), random.randint(0, 3))
  r_raw = complex(random.randint(-4, 4), random.randint(-4, 4))
  if r_raw == 0: r_raw = 1
  _, r = _gaussian_divmod(r_raw, m)
  q_pool = [complex(r, i) for r in range(-5, 6) for i in range(-3, 4)
            if r != 0 or i != 0]
  q = random.choice(q_pool)
  a = q * m + r
  return f'Mod[{_fmt(a)}, {_fmt(m)}]', _CongruenceCheck(a, m, r)

# Combo

def Binomial() -> tuple[str, complex]:
  n = random.randint(2, 12)
  k = random.randint(1, n)
  return f'Binomial[{n}, {k}]', complex(math.comb(n, k), 0)

def FactorialPower() -> tuple[str, complex]:
  n = random.randint(2, 12)
  k = random.randint(1, n)
  return f'FactorialPower[{n}, {k}]', complex(math.perm(n, k), 0)

## Roots

def _multiply_polynomials(p: list[int], q: list[int]) -> list[int]:
  result = [0] * (len(p) + len(q) - 1)
  for i, pi in enumerate(p):
    for j, qj in enumerate(q):
      result[i + j] += pi * qj
  return result

def _format_polynomial(coeffs: list[int]) -> str:
  terms = []
  for i in range(len(coeffs) - 1, -1, -1):
    c = coeffs[i]
    if c == 0: continue
    sign = '-' if c < 0 else ('+' if terms else '')
    ac = abs(c)
    if i == 0: term = f'{sign} {ac}'
    elif i == 1: term = f'{sign} {ac}x' if ac != 1 else f'{sign} x'
    else: term = f'{sign} {ac}x^{i}' if ac != 1 else f'{sign} x^{i}'
    terms.append(term)
  return ' '.join(terms).strip()

def _roots_integer(degree: int) -> tuple[list[int], list[complex]]:
  roots_int = [random.randint(-10, 10) for _ in range(degree)]
  coeffs = [1]
  for r in roots_int: coeffs = _multiply_polynomials(coeffs, [-r, 1])
  return coeffs, [complex(r, 0) for r in roots_int]

def _roots_rational(degree: int) -> tuple[list[int], list[complex]]:
  roots_frac = [Fraction(random.randint(-6, 6), random.randint(2, 4)) for _ in range(degree)]
  coeffs = [1]
  for r in roots_frac: coeffs = _multiply_polynomials(coeffs, [-r.numerator, r.denominator])
  g = coeffs[0]
  for c in coeffs[1:]: g = math.gcd(g, c)
  if g > 1: coeffs = [c // g for c in coeffs]
  return coeffs, [complex(float(r), 0) for r in roots_frac]

def _roots_gaussian(degree: int) -> tuple[list[int], list[complex]]:
  roots: list[complex] = []
  coeffs = [1]
  rem = degree
  while rem > 0:
    if COMPLEX and rem >= 2 and random.choice([True, False]):
      a = random.randint(-10, 10)
      b = random.randint(1, 10)
      roots.extend([complex(a, b), complex(a, -b)])
      coeffs = _multiply_polynomials(coeffs, [a * a + b * b, -2 * a, 1])
      rem -= 2
    else:
      r = random.randint(-10, 10)
      roots.append(complex(r, 0))
      coeffs = _multiply_polynomials(coeffs, [-r, 1])
      rem -= 1
  return coeffs, roots

SQUAREFREE = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23]

def _roots_surd_real(degree: int) -> tuple[list[int], list[complex]]:
  max_coeff = 300 if degree <= 3 else 500
  # Empirics (20k trials): coeff bound always satisfied within 5 tries; 0 failures.
  for _ in range(5):
    roots: list[complex] = []
    coeffs = [1]
    rem = degree
    while rem > 0:
      if rem >= 2 and random.choice([True, rem >= 4]):
        p = random.randint(-4, 4)
        q = 1
        r = random.randint(1, 3)
        d = random.choice(SQUAREFREE[:8] if degree >= 3 else SQUAREFREE)
        # Roots (p +- q*sqrt(d))/r:
        # factor = r^2 x^2 - 2pr x + (p^2 - q^2 d)
        factor = [p * p - q * q * d, -2 * p * r, r * r]
        coeffs = _multiply_polynomials(coeffs, factor)
        sqrt_d = math.sqrt(d)
        roots.append(complex((p + q * sqrt_d) / r, 0))
        roots.append(complex((p - q * sqrt_d) / r, 0))
        rem -= 2
      else:
        n = random.randint(-10, 10)
        roots.append(complex(n, 0))
        coeffs = _multiply_polynomials(coeffs, [-n, 1])
        rem -= 1
    if max(abs(c) for c in coeffs) <= max_coeff: return coeffs, roots
  return coeffs, roots

def _roots_surd_complex(degree: int) -> tuple[list[int], list[complex]]:
  max_coeff = 300 if degree <= 3 else 500
  # Empirics (20k trials): coeff bound always satisfied within 5 tries; 0 failures.
  for _ in range(5):
    roots: list[complex] = []
    coeffs = [1]
    rem = degree
    while rem > 0:
      if rem >= 2 and random.choice([True, rem >= 4]):
        p = random.randint(-4, 4)
        q = 1
        r = random.randint(1, 3)
        d = random.choice(SQUAREFREE[:8] if degree >= 3 else SQUAREFREE)
        # Roots (p +- qi*sqrt(d))/r:
        # factor = r^2 x^2 - 2pr x + (p^2 + q^2 d)
        factor = [p * p + q * q * d, -2 * p * r, r * r]
        coeffs = _multiply_polynomials(coeffs, factor)
        sqrt_d = math.sqrt(d)
        roots.append(complex(p / r, q * sqrt_d / r))
        roots.append(complex(p / r, -q * sqrt_d / r))
        rem -= 2
      else:
        n = random.randint(-10, 10)
        roots.append(complex(n, 0))
        coeffs = _multiply_polynomials(coeffs, [-n, 1])
        rem -= 1
    if max(abs(c) for c in coeffs) <= max_coeff: return coeffs, roots
  return coeffs, roots

def _roots_mixed(degree: int) -> tuple[list[int], list[complex]]:
  """Interleave integer, rational, and surd factors in one polynomial."""
  max_coeff = 500
  for _ in range(5):
    roots: list[complex] = []
    coeffs = [1]
    rem = degree
    while rem > 0:
      kind = random.choice(['int', 'rat', 'surd'])
      if kind == 'int' or rem == 1:
        r = random.randint(-10, 10)
        roots.append(complex(r, 0))
        coeffs = _multiply_polynomials(coeffs, [-r, 1])
        rem -= 1
      elif kind == 'rat':
        r = Fraction(random.randint(-6, 6), random.randint(2, 4))
        roots.append(complex(float(r), 0))
        coeffs = _multiply_polynomials(coeffs, [-r.numerator, r.denominator])
        rem -= 1
      elif kind == 'surd' and rem >= 2:
        p = random.randint(-4, 4)
        q = 1
        rd = random.randint(1, 3)
        d = random.choice(SQUAREFREE[:8] if degree >= 3 else SQUAREFREE)
        factor = [p * p - q * q * d, -2 * p * rd, rd * rd]
        coeffs = _multiply_polynomials(coeffs, factor)
        sqrt_d = math.sqrt(d)
        roots.append(complex((p + q * sqrt_d) / rd, 0))
        roots.append(complex((p - q * sqrt_d) / rd, 0))
        rem -= 2
    if max(abs(c) for c in coeffs) <= max_coeff: return coeffs, roots
  return coeffs, roots

def _roots_repeated(degree: int) -> tuple[list[int], list[complex]]:
  unique_n = random.randint(1, max(1, degree - 1))
  unique = [random.randint(-10, 10) for _ in range(unique_n)]
  roots_int = unique[:]
  while len(roots_int) < degree: roots_int.append(random.choice(unique))
  random.shuffle(roots_int)
  coeffs = [1]
  for r in roots_int: coeffs = _multiply_polynomials(coeffs, [-r, 1])
  return coeffs, [complex(r, 0) for r in roots_int]

def Roots() -> tuple[str, list[complex]]:
  degree = random.choices([2, 3, 4], weights=[90, 10, 1], k=1)[0]
  strategies = ['integer', 'rational', 'surd_real', 'repeated', 'mixed']
  if COMPLEX: strategies.extend(['gaussian', 'surd_complex'])
  strategy = random.choice(strategies)
  if strategy == 'integer': coeffs, roots = _roots_integer(degree)
  elif strategy == 'rational': coeffs, roots = _roots_rational(degree)
  elif strategy == 'gaussian': coeffs, roots = _roots_gaussian(degree)
  elif strategy == 'surd_real': coeffs, roots = _roots_surd_real(degree)
  elif strategy == 'surd_complex': coeffs, roots = _roots_surd_complex(degree)
  elif strategy == 'repeated': coeffs, roots = _roots_repeated(degree)
  elif strategy == 'mixed': coeffs, roots = _roots_mixed(degree)
  else: coeffs, roots = _roots_integer(degree)

  # Optional scalar leading coefficient (non-monic polynomials).
  if random.random() < 0.3:
    s = random.randint(2, 3)
    scaled = [c * s for c in coeffs]
    if max(abs(c) for c in scaled) <= 500:
      coeffs = scaled

  return f'Roots[{_format_polynomial(coeffs)} == 0, x]', roots

# Matrix 2D

def matrix_multiply():
  a11, a12 = _rng(-4, 4), _rng(-4, 4)
  a21, a22 = _rng(-4, 4), _rng(-4, 4)
  b11, b12 = _rng(-4, 4), _rng(-4, 4)
  b21, b22 = _rng(-4, 4), _rng(-4, 4)
  r11 = a11 * b11 + a12 * b21
  r12 = a11 * b12 + a12 * b22
  r21 = a21 * b11 + a22 * b21
  r22 = a21 * b12 + a22 * b22
  A = format_matrix([[a11, a12], [a21, a22]])
  B = format_matrix([[b11, b12], [b21, b22]])
  return f'{A} * {B}', (r11, r12, r21, r22)

_gaussian_inv_cache: dict = {}

def _gaussian_inverse_candidates(det: complex, bound: int) -> list:
  """Enumerate bounded Gaussian unimodular matrices by completing primitive top rows."""
  key = (round(det.real), round(det.imag), bound)
  if key in _gaussian_inv_cache:
    return _gaussian_inv_cache[key]

  pool = [complex(r, i) for r in range(-bound, bound + 1)
          for i in range(-bound, bound + 1)
          if r != 0 or i != 0]
  candidates = []
  for a in pool:
    for b in pool:
      if a == b: continue
      g, x, y = _gaussian_extended_gcd(a, b)
      if not _is_gaussian_unit(g): continue
      c = _gaussian_exact_div(-det * y, g)
      d = _gaussian_exact_div(det * x, g)
      if (abs(c.real) <= bound and abs(c.imag) <= bound and
          abs(d.real) <= bound and abs(d.imag) <= bound):
        candidates.append((a, b, c, d))

  _gaussian_inv_cache[key] = candidates
  return candidates

def Inverse():
  # Restrict |det| = 1 so A^-1 = adj(A)/det has integer entries.
  bound = 8

  if not COMPLEX:
    det = random.choice([-1, 1])
    candidates = []
    for a_val in range(1, bound + 1):
      for b_val in range(-bound, bound + 1):
        if b_val == 0: continue
        if math.gcd(abs(a_val), abs(b_val)) != 1: continue
        g, x, y = _extended_gcd(a_val, b_val)
        if g != 1: continue
        c_val = -det * y
        d_val = det * x
        if abs(c_val) <= bound and abs(d_val) <= bound:
          candidates.append((a_val, b_val, c_val, d_val))

    if not candidates:
      raise RuntimeError('no real unimodular inverse candidates within bounds')

    a_val, b_val, c_val, d_val = random.choice(candidates)
    a = complex(a_val, 0); b = complex(b_val, 0)
    c = complex(c_val, 0); d = complex(d_val, 0)
    scale = 1 / det
    inv = (scale * d, -scale * b, -scale * c, scale * a)
    return f'Inverse{format_matrix([[a, b], [c, d]])}', inv

  det = random.choice([-1, 1, 1j, -1j])
  candidates = _gaussian_inverse_candidates(det, bound)
  if not candidates:
    raise RuntimeError('no Gaussian unimodular inverse candidates within bounds')

  a, b, c, d = random.choice(candidates)
  scale = 1 / det
  inv = (scale * d, -scale * b, -scale * c, scale * a)
  return f'Inverse{format_matrix([[a, b], [c, d]])}', inv

def Det():
  a, b = _rng(-8, 8), _rng(-8, 8)
  c, d = _rng(-8, 8), _rng(-8, 8)
  return f'Det{format_matrix([[a, b], [c, d]])}', a * d - b * c

def Eigenvalues():
  strategy = random.choice(['real_diagonalizable', 'complex_conjugate_pair'])

  if strategy == 'real_diagonalizable':
    l1 = random.randint(-8, 8)
    l2 = random.choice([x for x in range(-8, 9) if x != l1])

    # Triangular construction: eigenvalues are on the diagonal.
    # Small off-diagonal entry for variety; covers diagonal case (k=0).
    k = random.randint(-3, 3)
    if random.choice([True, False]):
      a, b, c, d = l1, k, 0, l2
    else:
      a, b, c, d = l1, 0, k, l2

    return f'Eigenvalues{format_matrix([[a, b], [c, d]])}', (complex(l1, 0), complex(l2, 0))

  # Real block form [[re, -im], [im, re]] has eigenvalues re +- i*im.
  re = random.randint(-3, 3)
  im = random.randint(1, 3)
  a, b, c, d = re, -im, im, re
  return f'Eigenvalues{format_matrix([[a, b], [c, d]])}', (complex(re, im), complex(re, -im))

# Transcendentals

def Sin():
  z = _rng_float(0, math.pi / 2)
  return f'Sin[{_fmt(z)}]', cmath.sin(z)

def Cos():
  z = _rng_float(0, math.pi / 2)
  return f'Cos[{_fmt(z)}]', cmath.cos(z)

def Tan():
  z = _rng_float(0, math.pi / 3)
  return f'Tan[{_fmt(z)}]', cmath.tan(z)

def ArcSin():
  x = _rng_float(0, 1, imag=False)
  return f'ArcSin[{x.real}]', complex(math.asin(x.real), 0)

def ArcCos():
  x = _rng_float(0, 1, imag=False)
  return f'ArcCos[{x.real}]', complex(math.acos(x.real), 0)

def ArcTan():
  x = _rng_float(0, 1, imag=False)
  return f'ArcTan[{x.real}]', complex(math.atan(x.real), 0)

def complex_rotation():
  z = _rng_float(-5, 5)
  if abs(z) < 1e-9: z = complex(1, 0)
  deg = round(random.uniform(10, 350), 1)
  rad = math.radians(deg)
  result = z * complex(math.cos(rad), math.sin(rad))
  return f'{_fmt(z)} * Exp[I {deg:.1f} Degree]', result

#
# Game settings and main loop
#

ENABLED_MODES: list[Callable[[], tuple[str, object]]] = [
  Plus,
  Minus,
  Times,
  Divide,
  #Power,
  #Log,
  #GCD,
  #Mod,
  #Binomial,
  #FactorialPower,
  #Roots,
  #matrix_multiply,
  #Inverse,
  #Det,
  #Eigenvalues,
  #Sin,
  #Cos,
  #Tan,
  #ArcSin,
  #ArcCos,
  #ArcTan,
  #complex_rotation,
]

def _stop_game(stop_event: threading.Event, timer: threading.Timer) -> None:
  stop_event.clear()
  timer.cancel()

if __name__ == '__main__':
  score = 0
  correct_answer: object | None = None
  keep_running = threading.Event()
  keep_running.set()
  game_timer = threading.Timer(DURATION, keep_running.clear)
  game_timer.start()

  def quit_game():
    _stop_game(keep_running, game_timer)

  def format_final_answer(answer: object) -> str:
    if isinstance(answer, _UnitTolerantComplex): return _fres(answer.canonical)
    if isinstance(answer, _CongruenceCheck): return _fres(answer.remainder)
    if isinstance(answer, list): return ', '.join(_fmt(v) for v in answer)
    if isinstance(answer, tuple):
      if len(answer) == 4: return format_matrix([[answer[0], answer[1]], [answer[2], answer[3]]])
      if len(answer) == 9: return format_matrix([[answer[0], answer[1], answer[2]],
                                                 [answer[3], answer[4], answer[5]],
                                                 [answer[6], answer[7], answer[8]]])
      return ', '.join(_fmt(v) for v in answer)
    if isinstance(answer, complex): return _fres(answer)
    return str(answer)

  try:
    while keep_running.is_set():
      prompt, correct_answer = random.choice(ENABLED_MODES)()

      while keep_running.is_set():
        try: user_input = input(f'{prompt}\n> ')
        except (EOFError, KeyboardInterrupt): quit_game(); break
        if user_input.strip().lower() == 'q': quit_game(); break

        try:
          if isinstance(correct_answer, _UnitTolerantComplex):
            parsed = parse_complex_from_user(user_input)
            if abs(_canonical_gaussian(parsed) - correct_answer.canonical) < 0.01: break

          elif isinstance(correct_answer, _CongruenceCheck):
            parsed = parse_complex_from_user(user_input)
            diff = correct_answer.a - parsed
            if diff == 0: break
            q = diff / correct_answer.m
            if abs(q.real - round(q.real)) < 1e-9 and abs(q.imag - round(q.imag)) < 1e-9: break

          elif isinstance(correct_answer, list):
            parsed = parse_complex_sequence(user_input)
            if len(parsed) != len(correct_answer): continue
            matched = [False] * len(correct_answer)
            for u in parsed:
              found = False
              for i, expected in enumerate(correct_answer):
                if not matched[i] and abs(u - expected) < 0.01:
                  matched[i] = True
                  found = True
                  break
              if not found: break
            else:
              if all(matched): break

          elif isinstance(correct_answer, tuple):
            parsed = parse_tuple_answer(user_input)
            if len(parsed) == len(correct_answer):
              if len(correct_answer) in (2, 3):
                used = [False] * len(correct_answer)
                ok = True
                for u in parsed:
                  found = False
                  for i, e in enumerate(correct_answer):
                    if not used[i] and abs(u - e) < 0.01:
                      used[i] = True
                      found = True
                      break
                  if not found:
                    ok = False
                    break
                if ok: break
              elif parsed == correct_answer:
                break

          elif isinstance(correct_answer, complex):
            parsed = parse_complex_from_user(user_input)
            if abs(parsed - correct_answer) / (abs(correct_answer) or 1) <= 0.01: break

          else:
            parsed = float(user_input)
            if abs(parsed - correct_answer) < 0.01: break

        except (ValueError, TypeError, ZeroDivisionError): continue

      if keep_running.is_set():
        score += 1
        print(f'Score: {score}')

  except (KeyboardInterrupt, EOFError):
    quit_game()

  if correct_answer is not None:
    print(f'Answer: {format_final_answer(correct_answer)}')
