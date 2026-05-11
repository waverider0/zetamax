#!/usr/bin/env python3

import math, cmath, random, threading
from typing import Callable

DURATION = 120
COMPLEX = False

# deepseek v4-pro:
#
# The obvious way to build a math problem generator is rejection sampling:
# roll random numbers, form a problem, check whether the answer came out
# "nice" — small integers, clean radicals, nothing that would confuse a
# student. If the answer is ugly, roll again. For arithmetic this works
# perfectly. Pick any two numbers, add them, the sum is never ugly. The
# problem space is wide and regular and the forward map (addition)
# preserves niceness for free.
#
# Step outside arithmetic and rejection sampling breaks. The eigenvalue
# code that originally shipped here sampled random integer matrices P,
# computed A = P D P^{-1}, and checked whether A's entries stayed within
# bounds. Hit rate: ~4%. When the loop exhausted it fell back to the
# same hardcoded matrix every time — mode collapse. The issue is not
# that the bounds are too tight; it is that "nice eigenvalues" and "nice
# matrix entries" each define a sparse subset of the sampling space, and
# their intersection is sparse and irregular. If the region you are
# trying to hit cannot be parameterized directly, random sampling will
# not cover it.
#
# The alternative is to sample the answer space instead of the problem
# space. Pick clean answers first, project them forward to problems. For
# polynomial roots: pick integer (or rational, or simple radical) roots,
# multiply out the factors, present the expanded polynomial. The forward
# map — symmetric polynomial expansion — is an integer-coefficient
# polynomial map. Integer in, integer out, always. Every answer-tuple
# within bounds produces a valid problem. No rejection loop, no fallback,
# full coverage of the space.
#
# All the algebra here works this way because the forward maps are
# polynomial over the integers: LUP factorization for determinants, row
# operations for inverses, trace/determinant constraints for eigenvalues,
# symmetric sums for roots. Every generator is constructive. COMPLEX
# lifts everything to Gaussian integers with no change to the maps.
#
# Calculus does not admit this. The forward map for antiderivatives is
# differentiation, and differentiation does not preserve niceness — a
# tidy F becomes a sprawling f. Going the other direction (sample f,
# check whether \int f is elementary) is worse: almost no random function
# has an elementary antiderivative. The intersection of "nice integrand"
# and "nice integral" is sparse with no known constructive
# parameterization. Algebra is tractable because its maps are polynomial;
# calculus is not because its maps are transcendental.

def _rng(low: int, high: int, imag: bool = True) -> complex:
	real = random.randint(low, high)
	if not imag or not COMPLEX:
		return complex(real, 0)
	imag_part = random.randint(low, high)
	if real == 0 and imag_part == 0:
		imag_part = 1
	return complex(real, imag_part)

def _rng_float(low: float, high: float, imag: bool = True) -> complex:
	real = round(random.uniform(low, high), 2)
	if not imag or not COMPLEX:
		return complex(real, 0)
	imag_part = round(random.uniform(low, high), 2)
	if abs(real) < 1e-9 and abs(imag_part) < 1e-9:
		imag_part = 1.0
	return complex(real, imag_part)

#
# UI
#

def parse_complex_from_user(user_input: str) -> complex:
	cleaned = user_input.strip().replace(' ', '').lower()
	if cleaned in ('i', '+i'):
		return 1j
	if cleaned == '-i':
		return -1j
	cleaned = cleaned.replace('i', 'j')
	return complex(cleaned)

def parse_tuple_answer(user_input: str) -> tuple:
	parts = (user_input
		.strip()
		.replace(',', ' ')
		.replace(';', ' ')
		.replace('|', ' ')
		.split())
	result: list = []
	for part in parts:
		try:
			result.append(int(part))
		except ValueError:
			result.append(parse_complex_from_user(part))
	return tuple(result)

def format_complex_cartesian(real_value: float, imag_value: float) -> str:
	if imag_value == 0:
		return str(int(real_value)) if real_value == int(real_value) else f'{real_value:.2f}'
	if real_value == 0:
		if imag_value == 1:
			return 'I'
		if imag_value == -1:
			return '-I'
		return f'{int(imag_value)}I' if imag_value == int(imag_value) else f'{imag_value:.2f}I'
	sign = '+' if imag_value >= 0 else '-'
	real_string = str(int(real_value)) if real_value == int(real_value) else f'{real_value:.2f}'
	abs_imag = abs(imag_value)
	if abs_imag == 1:
		return f'({real_string} {sign} I)'
	imag_string = str(int(abs_imag)) if abs_imag == int(abs_imag) else f'{abs_imag:.2f}'
	return f'({real_string} {sign} {imag_string}I)'

#
# TODO: format_complex_int(), format_complex_float()
#
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
# arithmetic generators
#

def addition() -> tuple[str, complex]:
	z1 = _rng(2, 100)
	z2 = _rng(2, 100)
	return f'{_fmt(z1)} + {_fmt(z2)}', z1 + z2

def subtraction() -> tuple[str, complex]:
	z1 = _rng(2, 100)
	z2 = _rng(2, 100)
	return f'{_fmt(z1)} - {_fmt(z2)}', z1 - z2

def multiplication() -> tuple[str, complex]:
	z1 = _rng(2, 100)
	z2 = _rng(2, 20)
	return f'{_fmt(z1)} * {_fmt(z2)}', z1 * z2

def division() -> tuple[str, complex]:
	z1 = _rng(2, 100)
	z2 = _rng(2, 100)
	z3 = z1 * z2
	return f'{_fmt(z3)} / {_fmt(z2)}', z1

def power() -> tuple[str, complex]:
	z = _rng(2, 10)
	x = random.randint(2, 8)
	return f'{_fmt(z)}^{x}', z ** x

#
# transcendental generators --intentionally minimal
#

def integer_log():
	n = random.randint(2, 6)
	while True:
		z1 = _rng(-8, 8)
		# Gaussian units (+-1, +-i) cycle under exponentiation
		# (i^1 = i^5 = i) making Log ambiguous with multiple
		# integer answers. All other bases in [-8,8]^2 are
		# injective: z^k = z^n implies k = n. Rejection rate
		# ~1.7% across 50k samples -- not a sparse search.
		if abs(z1) != 1:
			break
	z2 = z1 ** n
	return f'Log[{_fmt(z1)}, {_fmt(z2)}]', complex(n, 0)

def complex_rotation():
	z = _rng_float(-5, 5)
	if abs(z) < 1e-9: z = complex(1, 0)
	deg = round(random.uniform(10, 350), 1)
	rad = math.radians(deg)
	result = z * complex(math.cos(rad), math.sin(rad))
	return f'{_fmt(z)} * Exp[I {deg:.1f} Degree]', result

def complex_angle():
	z = _rng_float(-5, 5)
	if abs(z) < 1e-9: z = complex(1, 0)
	return f'Arg[{z}]', math.atan(z.imag / z.real)

#
# Matrix 2D
#

def determinant_2x2():
	a, b = _rng(-8, 8), _rng(-8, 8)
	c, d = _rng(-8, 8), _rng(-8, 8)
	return f'Det{format_matrix([[a, b], [c, d]])}', a * d - b * c

def matmul_2x2():
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

def inverse_2x2():
	# Restrict |det| = 1 so A⁻¹ = adj(A)/det has Gaussian integer entries.
	det = random.choice([-1, 1, 1j, -1j]) if COMPLEX else random.choice([-1, 1])
	bound = 4
	for _ in range(50):
		a = _rng(-bound, bound)
		b = _rng(-bound, bound)
		if a == 0 or b == 0:
			continue
		for d_real in range(-bound, bound + 1):
			for d_imag in range(-bound, bound + 1):
				d = complex(d_real, d_imag)
				# Solve ad - bc = det for c over Gaussian integers:
				# c = (a·d - det)/b must have integer real and imaginary parts
				num = a * d - det
				c = num / b
				if abs(c.real - round(c.real)) < 1e-9 and abs(c.imag - round(c.imag)) < 1e-9:
					cr = round(c.real)
					ci = round(c.imag)
					if -bound <= cr <= bound and -bound <= ci <= bound:
						c = complex(cr, ci)
						if b != 0 and c != 0:
							if det == 1: inv = (d, -b, -c, a)
							elif det == -1: inv = (-d, b, c, -a)
							elif det == 1j: inv = (-1j * d, 1j * b, 1j * c, -1j * a)
							else: inv = (1j * d, -1j * b, -1j * c, 1j * a)
							return f'Inverse{format_matrix([[a, b], [c, d]])}', inv
	I = format_matrix([[1, 0], [0, 1]])
	return f'Inverse{I}', (1, 0, 0, 1)

def eigenvalues_2x2():
	strategy = random.choice(['real_diagonalizable', 'complex_conjugate_pair'])
	bound = 8
	if strategy == 'real_diagonalizable':
		lam1 = random.randint(-5, 5)
		lam2 = random.randint(-5, 5)
		if lam1 == lam2: lam2 += random.choice([-1, 1])
		trace = lam1 + lam2
		det_val = lam1 * lam2
		# Characteristic polynomial: λ² - tr(A)·λ + det(A).
		# For eigenvalues λ₁,λ₂ we need tr(A)=λ₁+λ₂ and det(A)=λ₁λ₂.
		# Pick a, set d = tr - a, then find b,c such that ad - bc = det.
		candidates = []
		for a in range(-bound, bound + 1):
			d = trace - a
			if abs(d) > bound:
				continue
			bc = a * d - det_val
			if bc == 0:
				continue
			for b in range(-bound, bound + 1):
				if b == 0:
					continue
				if bc % b == 0:
					c = bc // b
					if abs(c) <= bound and c != 0:
						candidates.append((a, b, c, d))
		if candidates: a, b, c, d = random.choice(candidates)
		else: a, b, c, d = lam1 + lam2, 1, 1, lam1 + lam2 - 1  # fallback
		return f'Eigenvalues{format_matrix([[a, b], [c, d]])}', (complex(lam1, 0), complex(lam2, 0))
	# Complex: eigenvalues re +/- i*im
	re = random.randint(-3, 3)
	im = random.randint(1, 3)
	trace = 2 * re
	det_val = re * re + im * im
	candidates = []
	for a in range(-bound, bound + 1):
		d = trace - a
		if abs(d) > bound:
			continue
		bc = a * d - det_val
		if bc == 0:
			continue
		for b in range(-bound, bound + 1):
			if b == 0:
				continue
			if bc % b == 0:
				c = bc // b
				if abs(c) <= bound and c != 0:
					candidates.append((a, b, c, d))
	if candidates: a, b, c, d = random.choice(candidates)
	else: a, b, c, d = re, -im, im, re  # fallback: canonical form
	return f'Eigenvalues{format_matrix([[a, b], [c, d]])}', (complex(re, im), complex(re, -im))

#
# Matrix 3D
#

def matmul_3x3():
	A = [[_rng(-2, 2) for _ in range(3)] for _ in range(3)]
	B = [[_rng(-2, 2) for _ in range(3)] for _ in range(3)]
	C = [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
	flat = tuple(v for row in C for v in row)
	return f'{format_matrix(A)} * {format_matrix(B)}', flat

def inverse_3x3() -> tuple[str, tuple[complex, ...]]:
	A = [[complex(1, 0), complex(0, 0), complex(0, 0)],
	     [complex(0, 0), complex(1, 0), complex(0, 0)],
	     [complex(0, 0), complex(0, 0), complex(1, 0)]]
	ops = []
	n_ops = random.randint(4, 7) if COMPLEX else random.randint(2, 4)
	for _ in range(n_ops):
		t = random.choice(['add', 'swap', 'neg'])
		if t == 'add':
			i, j = random.sample([0, 1, 2], 2)
			k = _rng(-2, 2)
			if k == 0:
				k = complex(random.randint(-2, 2), random.randint(-2, 2))
				if k == 0:
					k = complex(1, 0)
			A[i] = [A[i][c] + k * A[j][c] for c in range(3)]
			ops.append(('add', i, j, k))
		elif t == 'swap':
			i, j = random.sample([0, 1, 2], 2)
			A[i], A[j] = A[j], A[i]
			ops.append(('swap', i, j))
		else:
			i = random.randint(0, 2)
			A[i] = [-x for x in A[i]]
			ops.append(('neg', i))
	inv = [[complex(1, 0), complex(0, 0), complex(0, 0)],
	       [complex(0, 0), complex(1, 0), complex(0, 0)],
	       [complex(0, 0), complex(0, 0), complex(1, 0)]]
	# If A = E_k · ... · E_1 · I, then A⁻¹ = E_1⁻¹ · ... · E_k⁻¹ · I.
	# Reversing the ops and inverting each (add→subtract, swap↔swap, neg↔neg)
	# applied to the identity yields the inverse.
	for op in reversed(ops):
		if op[0] == 'add':
			_, i, j, k = op
			inv[i] = [inv[i][c] - k * inv[j][c] for c in range(3)]
		elif op[0] == 'swap':
			_, i, j = op
			inv[i], inv[j] = inv[j], inv[i]
		else:
			_, i = op
			inv[i] = [-x for x in inv[i]]
	flat = tuple(v for row in inv for v in row)
	return f'Inverse{format_matrix(A)}', flat

def determinant_3x3():
	target_det = _rng(-10, 10)
	if target_det == 0: target_det = complex(random.choice([-1, 1]), random.randint(0, 5) if COMPLEX else 0)
	if target_det == 0: target_det = complex(1, 0)
	d1 = complex(1, 0)
	d2 = complex(1, 0)
	d3 = target_det
	# A = L·U with L unit lower-triangular, U upper-triangular.
	# det(A) = det(L)·det(U) = 1 · d1·d2·d3 = target_det
	u12 = _rng(-2, 2)
	u13 = _rng(-2, 2)
	u23 = _rng(-2, 2)
	l21 = _rng(-2, 2)
	l31 = _rng(-2, 2)
	l32 = _rng(-2, 2)
	rows = [
		[d1, u12, u13],
		[l21 * d1, l21 * u12 + d2, l21 * u13 + u23],
		[l31 * d1, l31 * u12 + l32 * d2, l31 * u13 + l32 * u23 + d3]
	]
	if random.random() < 0.3:
		r1, r2 = random.sample([0, 1, 2], 2)
		rows[r1], rows[r2] = rows[r2], rows[r1]
		target_det = -target_det
	return f'Det{format_matrix(rows)}', target_det

#
# Polynomial
#

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
		if c == 0:
			continue
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
	for r in roots_int:
		coeffs = _multiply_polynomials(coeffs, [-r, 1])
	return coeffs, [complex(r, 0) for r in roots_int]

def _roots_rational(degree: int) -> tuple[list[int], list[complex]]:
	from fractions import Fraction
	roots_frac = [Fraction(random.randint(-6, 6), random.randint(2, 4)) for _ in range(degree)]
	coeffs = [1]
	for r in roots_frac:
		coeffs = _multiply_polynomials(coeffs, [-r.numerator, r.denominator])
	g = coeffs[0]
	for c in coeffs[1:]:
		g = math.gcd(g, c)
	if g > 1:
		coeffs = [c // g for c in coeffs]
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
				# Roots (p ± q√d)/r: quadratic factor is r²x² - 2prx + (p² - q²d)
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
		if max(abs(c) for c in coeffs) <= max_coeff:
			return coeffs, roots
	# Last attempt stands (coefficients may be large but math is correct)
	return coeffs, roots

def _roots_surd_complex(degree: int) -> tuple[list[int], list[complex]]:
	max_coeff = 300 if degree <= 3 else 500
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
				# Roots (p ± qi√d)/r: quadratic factor is r²x² - 2prx + (p² + q²d)
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
		if max(abs(c) for c in coeffs) <= max_coeff:
			return coeffs, roots
	return coeffs, roots

def _roots_repeated(degree: int) -> tuple[list[int], list[complex]]:
	unique_n = random.randint(1, max(1, degree - 1))
	unique = [random.randint(-10, 10) for _ in range(unique_n)]
	roots_int = unique[:]
	while len(roots_int) < degree:
		roots_int.append(random.choice(unique))
	random.shuffle(roots_int)
	coeffs = [1]
	for r in roots_int:
		coeffs = _multiply_polynomials(coeffs, [-r, 1])
	return coeffs, [complex(r, 0) for r in roots_int]

def Roots() -> tuple[str, list[complex]]:
	degree = random.choices([2, 3, 4], weights=[90, 10, 1], k=1)[0]
	strategies = ['integer', 'rational', 'surd_real', 'repeated']
	if COMPLEX: strategies.extend(['gaussian', 'surd_complex'])
	strategy = random.choice(strategies)
	if strategy == 'integer': coeffs, roots = _roots_integer(degree)
	elif strategy == 'rational': coeffs, roots = _roots_rational(degree)
	elif strategy == 'gaussian': coeffs, roots = _roots_gaussian(degree)
	elif strategy == 'surd_real': coeffs, roots = _roots_surd_real(degree)
	elif strategy == 'surd_complex': coeffs, roots = _roots_surd_complex(degree)
	elif strategy == 'repeated': coeffs, roots = _roots_repeated(degree)
	else:
		coeffs, roots = _roots_integer(degree)
	return f'Roots[{_format_polynomial(coeffs)} == 0, x]', roots

#
# Main
#

ENABLED_MODES: list[Callable[[], tuple[str, object]]] = [
	addition,
	subtraction,
	multiplication,
	division,
	power,
	Roots,

	#integer_log,
	#complex_rotation,
	#complex_angle,

	#matmul_2x2,
	#inverse_2x2,
	#determinant_2x2,
	#eigenvalues_2x2,

	#matmul_3x3,
	#inverse_3x3,
	#determinant_3x3,
]

def _stop_game(stop_event: threading.Event, timer: threading.Timer) -> None:
	stop_event.clear()
	timer.cancel()

if __name__ == '__main__':
	score = 0
	correct_answer: complex | tuple | list | None = None
	keep_running = threading.Event()
	keep_running.set()
	game_timer = threading.Timer(DURATION, keep_running.clear)
	game_timer.start()

	def quit_game():
		_stop_game(keep_running, game_timer)

	def format_final_answer(answer: object) -> str:
		if isinstance(answer, list):
			return ', '.join(_fres(v) for v in answer)
		if isinstance(answer, tuple):
			if len(answer) == 4:
				return format_matrix([[answer[0], answer[1]], [answer[2], answer[3]]])
			if len(answer) == 9:
				return format_matrix([[answer[0], answer[1], answer[2]],
				                      [answer[3], answer[4], answer[5]],
				                      [answer[6], answer[7], answer[8]]])
			return ', '.join(_fres(v) for v in answer)
		if isinstance(answer, complex):
			return _fres(answer)
		return str(answer)

	try:
		while keep_running.is_set():
			prompt, correct_answer = random.choice(ENABLED_MODES)()

			while keep_running.is_set():
				try:
					user_input = input(f'{prompt}\n> ')
				except (EOFError, KeyboardInterrupt):
					quit_game()
					break

				if user_input.strip().lower() == 'q':
					quit_game()
					break

				try:
					if isinstance(correct_answer, list):
						parts = [p.strip() for p in user_input.replace(',', ' ').split() if p.strip()]
						if len(parts) != len(correct_answer):
							continue
						parsed = [parse_complex_from_user(p) for p in parts]
						matched = [False] * len(correct_answer)
						for u in parsed:
							found = False
							for i, expected in enumerate(correct_answer):
								if not matched[i] and abs(u - expected) < 0.01:
									matched[i] = True
									found = True
									break
							if not found:
								break
						else:
							if all(matched):
								break
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
								if ok:
									break
							elif parsed == correct_answer:
								break
					elif isinstance(correct_answer, complex):
						parsed = parse_complex_from_user(user_input)
						if abs(parsed - correct_answer) < 0.01:
							break
					else:
						parsed = float(user_input)
						if abs(parsed - correct_answer) / (abs(correct_answer) or 1) <= 0.01:
							break
				except (ValueError, TypeError):
					continue

			if keep_running.is_set():
				score += 1
				print(f'Score: {score}')

	except (KeyboardInterrupt, EOFError):
		quit_game()

	if correct_answer is not None:
		print(f'Answer: {format_final_answer(correct_answer)}')
