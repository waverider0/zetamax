#!/usr/bin/env python3

import math, cmath, random, threading
from fractions import Fraction
from typing import Callable

# Opus 4.6:
#
# Every generator emits a (problem, answer) pair by running a computation in
# its cheap direction. The student inverts it.
#
#   arithmetic    choose operands, apply the operation
#   roots         choose factors, multiply out
#   eigenvalues   choose spectrum, build A
#   inverse       choose a unimodular matrix
#   GCD           choose gcd and coprime cofactors, multiply up
#   Mod           choose modulus, remainder, quotient
#   integrals     choose antiderivative template, differentiate
#
# Arithmetic is trivially dense. Integers are closed under addition,
# subtraction, and multiplication. Every random operand pair yields a clean
# answer. No construction machinery needed.
#
# Algebra is sparse but constructible. Random integer polynomials almost never
# factor nicely. Random matrices almost never have integer inverses or clean
# eigenvalues. Rejection sampling hits a thin irregular target. But the forward
# maps are total and integer-preserving: polynomial multiplication takes integer
# roots to integer coefficients, Bezout completion takes a unit determinant to a
# unimodular matrix, multiplying a gcd by coprime cofactors gives the displayed
# pair. Start with a nice answer, push forward, land in a clean problem.
# Soundness is automatic.
#
# Coverage is separate. Each forward construction is a coordinate patch on the
# space of valid problems. Triangular eigenvalue builders cover triangular
# matrices. Surd factor families cover specific quadratic extensions. Broaden
# coverage by writing constructions, not by widening rejection. Equivalences
# need deliberate handling: Gaussian gcds up to unit associates, modular answers
# as congruence classes, roots and eigenvalues as unordered multisets, log bases
# chosen to avoid ambiguity.
#
# Calculus is a different kind of problem. The forward map no longer preserves
# the target region. In algebra, the constraint is integrality — a lattice that
# polynomial multiplication, Bezout completion, and matrix products all respect.
# In calculus, the constraint is displayability: short, recognizable,
# pedagogically conventional expressions. This is a property of syntax trees.
# It has no algebraic characterization. Its boundary is nowhere smooth. The
# chain rule, product rule, and quotient rule grow expressions freely.
# Differentiating a short clean antiderivative routinely produces something
# longer and unrecognizable. Starting from a nice answer and pushing forward is
# still necessary, but it is not sufficient on its own. Calculus generators need
# curated template families with strict expression budgets, machinery with no
# analogue in the algebra generators.
#
# Small rejection filters stay useful after construction — trim coefficient
# bounds, exclude degeneracies — but hit rate must be high, retries use fresh
# structure, and failure never falls back to a canned problem. Rejection is
# cleanup, not generation.

def _rng(low: int, high: int, imag: bool = True) -> complex:
  real = random.randint(low, high)
  if not imag or not COMPLEX: return complex(real, 0)
  return complex(real, random.randint(low, high))

def _rng_float(low: float, high: float, imag: bool = True) -> complex:
  real = round(random.uniform(low, high), 2)
  if not imag or not COMPLEX: return complex(real, 0)
  return complex(real, round(random.uniform(low, high), 2))

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
  return float(value)

def parse_complex_from_user(user_input: str) -> complex:
  cleaned = user_input.strip().replace(' ', '').lower()
  cleaned = _strip_outer_parens(cleaned)
  if cleaned in ('i', '+i', 'j', '+j'): return 1j
  if cleaned in ('-i', '-j'): return -1j
  cleaned = cleaned.replace('i', 'j')
  try:
    return complex(cleaned)
  except ValueError:
    if cleaned.endswith('j'):
      body = cleaned[:-1]
      sep = -1
      for i in range(1, len(body)):
        if body[i] in '+-': sep = i
      if sep != -1:
        real = float(body[:sep])
        imag = _parse_imag_coeff(body[sep:])
        return complex(real, imag)
      return complex(0, _parse_imag_coeff(body))
    return complex(float(cleaned), 0)

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
    if imag == 1: return 'I'
    if imag == -1: return '-I'
    return f'{int(imag)}I' if imag == int(imag) else f'{imag:.2f}I'
  sign = '+' if value.imag >= 0 else '-'
  real = value.real
  real_str = str(int(real)) if real == int(real) else f'{real:.2f}'
  imag = abs(value.imag)
  if imag == 1: return f'{real_str} {sign} I'
  imag_str = str(int(imag)) if imag == int(imag) else f'{imag:.2f}'
  return f'{real_str} {sign} {imag_str}I'

def _fmt(z: complex) -> str:
  return format_complex_cartesian(z.real, z.imag)

def _fres(z: complex) -> str:
  return format_complex_result(z)

def format_matrix(rows: list[list]) -> str:
  return '{' + ' | '.join(' '.join(_fmt(v) for v in row) for row in rows) + '}'

#
# Generators
#

def Plus() -> tuple[str, complex]:
  z1 = _rng(-200, 200)
  z2 = _rng(-200, 200)
  return f'{_fmt(z1)} + {_fmt(z2)}', z1 + z2

def Minus() -> tuple[str, complex]:
  z1 = _rng(-200, 200)
  z2 = _rng(-200, 200)
  return f'{_fmt(z1)} - {_fmt(z2)}', z1 - z2

def Times() -> tuple[str, complex]:
  z1 = _rng(-100, 100)
  z2 = _rng(-20, 20)
  return f'{_fmt(z1)} * {_fmt(z2)}', z1 * z2

def Divide() -> tuple[str, complex]:
  z1 = _rng(-100, 100)
  z2 = 0
  while z2 == 0: z2 = _rng(-100, 100)
  return f'{_fmt(z1)} / {_fmt(z2)}', z1 / z2

def Power() -> tuple[str, complex]:
  z = _rng(-10, 10)
  x = random.randint(2, 8)
  return f'{_fmt(z)}^{x}', z ** x

def Log() -> tuple[str, complex]:
  """Exact discrete exponent drill. Exclude 0 and units so the exponent is not ambiguous."""
  n = random.randint(2, 6)
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
  """Stores the canonical Gaussian gcd. Grading accepts all unit associates."""
  def __init__(self, canonical: complex):
    self.canonical = canonical

class _CongruenceCheck:
  """Stores a congruence class. Grading checks m | (a - answer)."""
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
  z = a / b
  q = complex(round(z.real), round(z.imag))
  if abs(a - q * b) > 1e-9:
    raise ArithmeticError(f'non-exact Gaussian division: {a} / {b}')
  return q

def _gaussian_extended_gcd(a: complex, b: complex) -> tuple[complex, complex, complex]:
  """Returns (g, x, y) with a*x + b*y = g = gcd(a,b) in Z[i]."""
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
  """Choose gcd first, then coprime cofactors. Complex answers are up to units."""
  if not COMPLEX:
    g = random.randint(2, 20)
    u = random.choice([n for n in range(-15, 16) if n])
    v = random.choice([n for n in range(-15, 16) if n and n != u])
    h = math.gcd(abs(u), abs(v))
    x = g * (u // h)
    y = g * (v // h)
    return f'GCD[{x}, {y}]', complex(g, 0)

  g_pool = [complex(r, i) for r in range(-5, 6) for i in range(-5, 6)
            if complex(r, i) != 0 and abs(complex(r, i)) > 1]
  g = _canonical_gaussian(random.choice(g_pool))

  raw_pool = [complex(r, i) for r in range(-6, 7) for i in range(-6, 7)
              if r != 0 or i != 0]
  a_raw = random.choice(raw_pool)
  b_raw = random.choice(raw_pool)
  while b_raw == a_raw:
    b_raw = random.choice(raw_pool)

  h = _gaussian_gcd(a_raw, b_raw)
  a = _gaussian_exact_div(a_raw, h)
  b = _gaussian_exact_div(b_raw, h)
  x = g * a
  y = g * b
  return f'GCD[{_fmt(x)}, {_fmt(y)}]', _UnitTolerantComplex(g)

def Mod() -> tuple[str, object]:
  """Choose modulus, remainder, and quotient before forming the dividend."""
  if not COMPLEX:
    m = random.randint(5, 30)
    r = random.randint(0, m - 1)
    q = random.choice([q for q in range(-30, 31) if q != 0])
    a = q * m + r
    return f'Mod[{a}, {m}]', complex(r, 0)

  m_pool = [complex(r, i) for r in range(-5, 6) for i in range(-5, 6)
            if abs(complex(r, i)) > 1]
  m = random.choice(m_pool)
  r_raw = complex(random.randint(-4, 4), random.randint(-4, 4))
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

def _primitive_coeffs(coeffs: list[int]) -> list[int]:
  g = 0
  for c in coeffs: g = math.gcd(g, c)
  if g > 1: return [c // g for c in coeffs]
  return coeffs

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
  roots: list[complex] = []
  coeffs = [1]
  rem = degree
  while rem > 0:
    if COMPLEX and rem >= 2:
      a = random.randint(-10, 10)
      b = random.randint(-10, 10)
      roots.extend([complex(a, b), complex(a, -b)])
      coeffs = _multiply_polynomials(coeffs, [a * a + b * b, -2 * a, 1])
      rem -= 2
    else:
      r = random.randint(-10, 10)
      roots.append(complex(r, 0))
      coeffs = _multiply_polynomials(coeffs, [-r, 1])
      rem -= 1
  return coeffs, roots

def _roots_rational(degree: int) -> tuple[list[int], list[complex]]:
  roots: list[complex] = []
  coeffs = [1]
  rem = degree
  while rem > 0:
    if COMPLEX and rem >= 2:
      a = Fraction(random.randint(-4, 4), random.randint(2, 4))
      b = Fraction(random.randint(-4, 4), random.randint(2, 4))
      c0 = a * a + b * b
      c1 = -2 * a
      l = math.lcm(c0.denominator, c1.denominator)
      factor = _primitive_coeffs([int(c0 * l), int(c1 * l), l])
      roots.extend([complex(float(a), float(b)), complex(float(a), -float(b))])
      coeffs = _multiply_polynomials(coeffs, factor)
      rem -= 2
    else:
      r = Fraction(random.randint(-6, 6), random.randint(2, 4))
      roots.append(complex(float(r), 0))
      coeffs = _multiply_polynomials(coeffs, _primitive_coeffs([-r.numerator, r.denominator]))
      rem -= 1
  return _primitive_coeffs(coeffs), roots

SQUAREFREE = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23]

def _roots_surd(degree: int) -> tuple[list[int], list[complex]]:
  max_coeff = 300 if degree <= 3 else 500
  for _ in range(5):
    roots: list[complex] = []
    coeffs = [1]
    rem = degree
    while rem > 0:
      if rem >= 2:
        p = random.randint(-4, 4)
        q = 1
        r = random.randint(1, 3)
        d = random.choice(SQUAREFREE[:8] if degree >= 3 else SQUAREFREE)
        sqrt_d = math.sqrt(d)
        if COMPLEX:
          factor = [p * p + q * q * d, -2 * p * r, r * r]
          roots.append(complex(p / r, q * sqrt_d / r))
          roots.append(complex(p / r, -q * sqrt_d / r))
        else:
          factor = [p * p - q * q * d, -2 * p * r, r * r]
          roots.append(complex((p + q * sqrt_d) / r, 0))
          roots.append(complex((p - q * sqrt_d) / r, 0))
        coeffs = _multiply_polynomials(coeffs, _primitive_coeffs(factor))
        rem -= 2
      else:
        n = random.randint(-10, 10)
        roots.append(complex(n, 0))
        coeffs = _multiply_polynomials(coeffs, [-n, 1])
        rem -= 1
    coeffs = _primitive_coeffs(coeffs)
    if max(abs(c) for c in coeffs) <= max_coeff: return coeffs, roots
  return coeffs, roots

def _roots_repeated(degree: int) -> tuple[list[int], list[complex]]:
  if COMPLEX and degree >= 4:
    roots: list[complex] = []
    coeffs = [1]
    a = random.randint(-3, 3)
    b = random.randint(-3, 3)
    factor = [a * a + b * b, -2 * a, 1]
    for _ in range(2):
      roots.extend([complex(a, b), complex(a, -b)])
      coeffs = _multiply_polynomials(coeffs, factor)
    rem = degree - 4
    while rem > 0:
      r = random.randint(-10, 10)
      roots.append(complex(r, 0))
      coeffs = _multiply_polynomials(coeffs, [-r, 1])
      rem -= 1
    return coeffs, roots

  unique_n = random.randint(1, max(1, degree - 1))
  unique = random.sample(range(-10, 11), unique_n)
  roots_int = unique[:]
  while len(roots_int) < degree: roots_int.append(random.choice(unique))
  random.shuffle(roots_int)
  coeffs = [1]
  for r in roots_int: coeffs = _multiply_polynomials(coeffs, [-r, 1])
  return coeffs, [complex(r, 0) for r in roots_int]

def _roots_mixed(degree: int) -> tuple[list[int], list[complex]]:
  max_coeff = 500
  for _ in range(5):
    roots: list[complex] = []
    coeffs = [1]
    rem = degree
    while rem > 0:
      if rem == 1:
        kind = random.choice(['int', 'rat'])
      elif COMPLEX:
        kind = random.choice(['gaussian', 'gaussian_rational', 'surd_complex'])
      else:
        kind = random.choice(['int', 'rat', 'surd'])

      if kind == 'int':
        r = random.randint(-10, 10)
        roots.append(complex(r, 0))
        coeffs = _multiply_polynomials(coeffs, [-r, 1])
        rem -= 1
      elif kind == 'rat':
        r = Fraction(random.randint(-6, 6), random.randint(2, 4))
        roots.append(complex(float(r), 0))
        coeffs = _multiply_polynomials(coeffs, _primitive_coeffs([-r.numerator, r.denominator]))
        rem -= 1
      elif kind == 'surd':
        p = random.randint(-4, 4)
        q = 1
        rd = random.randint(1, 3)
        d = random.choice(SQUAREFREE[:8] if degree >= 3 else SQUAREFREE)
        factor = [p * p - q * q * d, -2 * p * rd, rd * rd]
        coeffs = _multiply_polynomials(coeffs, _primitive_coeffs(factor))
        sqrt_d = math.sqrt(d)
        roots.append(complex((p + q * sqrt_d) / rd, 0))
        roots.append(complex((p - q * sqrt_d) / rd, 0))
        rem -= 2
      elif kind == 'gaussian':
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        roots.extend([complex(a, b), complex(a, -b)])
        coeffs = _multiply_polynomials(coeffs, [a * a + b * b, -2 * a, 1])
        rem -= 2
      elif kind == 'gaussian_rational':
        a = Fraction(random.randint(-4, 4), random.randint(2, 4))
        b = Fraction(random.randint(-4, 4), random.randint(2, 4))
        c0 = a * a + b * b
        c1 = -2 * a
        l = math.lcm(c0.denominator, c1.denominator)
        factor = _primitive_coeffs([int(c0 * l), int(c1 * l), l])
        coeffs = _multiply_polynomials(coeffs, factor)
        roots.extend([complex(float(a), float(b)), complex(float(a), -float(b))])
        rem -= 2
      elif kind == 'surd_complex':
        p = random.randint(-4, 4)
        q = 1
        rd = random.randint(1, 3)
        d = random.choice(SQUAREFREE[:8] if degree >= 3 else SQUAREFREE)
        factor = [p * p + q * q * d, -2 * p * rd, rd * rd]
        coeffs = _multiply_polynomials(coeffs, _primitive_coeffs(factor))
        sqrt_d = math.sqrt(d)
        roots.append(complex(p / rd, q * sqrt_d / rd))
        roots.append(complex(p / rd, -q * sqrt_d / rd))
        rem -= 2

    coeffs = _primitive_coeffs(coeffs)
    if max(abs(c) for c in coeffs) <= max_coeff: return coeffs, roots
  return coeffs, roots

def Roots() -> tuple[str, list[complex]]:
  """Choose factors first. Answers are numeric; users enter decimal approximations."""
  degree = random.choices([2, 3, 4], weights=[90, 10, 1], k=1)[0]
  strategy = random.choice(['integer', 'rational', 'surd', 'repeated', 'mixed'])
  if strategy == 'integer': coeffs, roots = _roots_integer(degree)
  elif strategy == 'rational': coeffs, roots = _roots_rational(degree)
  elif strategy == 'surd': coeffs, roots = _roots_surd(degree)
  elif strategy == 'repeated': coeffs, roots = _roots_repeated(degree)
  elif strategy == 'mixed': coeffs, roots = _roots_mixed(degree)
  else: coeffs, roots = _roots_integer(degree)
  return f'Roots[{_format_polynomial(coeffs)} == 0, x]', roots

# Matrices

def matrix_multiply():
  n = 2
  A = [[_rng(-4, 4) for _ in range(n)] for _ in range(n)]
  B = [[_rng(-4, 4) for _ in range(n)] for _ in range(n)]
  C = tuple(sum(A[i][k] * B[k][j] for k in range(n))
            for i in range(n) for j in range(n))
  return f'{format_matrix(A)} * {format_matrix(B)}', C

_gaussian_inv_cache: dict = {}

def _gaussian_inverse_candidates(det: complex, bound: int) -> list:
  """Enumerate bounded unimodular matrices by completing primitive top rows."""
  key = (round(det.real), round(det.imag), bound, COMPLEX)
  if key in _gaussian_inv_cache:
    return _gaussian_inv_cache[key]

  imag_values = range(-bound, bound + 1) if COMPLEX else range(1)
  pool = [complex(r, i) for r in range(-bound, bound + 1) for i in imag_values]
  candidates = []
  for a in pool:
    for b in pool:
      if a == 0 and b == 0: continue
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
  """Integer inverse iff determinant is a unit. Complete a primitive row by Bezout."""
  bound = 8
  det = random.choice([-1, 1, 1j, -1j] if COMPLEX else [-1, 1])
  candidates = _gaussian_inverse_candidates(det, bound)
  if not candidates:
    raise RuntimeError('no unimodular inverse candidates within bounds')

  a, b, c, d = random.choice(candidates)
  scale = 1 / det
  inv = (scale * d, -scale * b, -scale * c, scale * a)
  return f'Inverse{format_matrix([[a, b], [c, d]])}', inv

def Det():
  a, b = _rng(-8, 8), _rng(-8, 8)
  c, d = _rng(-8, 8), _rng(-8, 8)
  return f'Det{format_matrix([[a, b], [c, d]])}', a * d - b * c

def Eigenvalues():
  """Choose eigenvalues first, then build a triangular matrix."""
  l1 = _rng(-8, 8)
  l2 = _rng(-8, 8)
  while abs(l2 - l1) < 1e-9:
    l2 = _rng(-8, 8)
  k = _rng(-3, 3)
  return f'Eigenvalues{format_matrix([[l1, k], [0, l2]])}', (l1, l2)

# Transcendentals

def Sin():
  z = _rng_float(-math.pi, math.pi)
  return f'Sin[{_fmt(z)}]', cmath.sin(z)

def Cos():
  z = _rng_float(-math.pi, math.pi)
  return f'Cos[{_fmt(z)}]', cmath.cos(z)

def Tan():
  z = _rng_float(-math.pi / 3, math.pi / 3)
  return f'Tan[{_fmt(z)}]', cmath.tan(z)

def ArcSin():
  z = _rng_float(-1, 1)
  return f'ArcSin[{_fmt(z)}]', cmath.asin(z)

def ArcCos():
  z = _rng_float(-1, 1)
  return f'ArcCos[{_fmt(z)}]', cmath.acos(z)

def ArcTan():
  z = _rng_float(-1, 1)
  while COMPLEX and abs(z.real) < 1e-12 and abs(abs(z.imag) - 1) < 1e-12:
    z = _rng_float(-1, 1)
  return f'ArcTan[{_fmt(z)}]', cmath.atan(z)

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

DURATION = 120
COMPLEX = False

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
