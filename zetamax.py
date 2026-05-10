#!/usr/bin/env python3

import math, cmath, random, threading
from typing import Callable

DURATION = 120
COMPLEX = False

#
# RNG
#

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
	# ( x0 y0 | x1 y1 )
	return '( ' + ' | '.join(' '.join(_fentry(v) for v in row) for row in rows) + ' )'

#
# arithmetic
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
	x = random.randint(2, 4)
	return f'{_fmt(z)}^{x}', z ** x

#
# transcendental
#

def Exp() -> tuple[str, complex]:
	z = _rng_float(-2, 2)
	return f'Exp[{_fmt(z)}]', cmath.exp(z)

def Log() -> tuple[str, complex]:
	x = _rng(2, 1000, imag=False)
	return f'Log[{_fmt(x)}]', cmath.log(x)

def LogBase() -> tuple[str, complex]:
	b = _rng(2, 16, imag=False)
	x = _rng(2, 1000, imag=False)
	return f'Log[{_fmt(b)}, {_fmt(x)}]', cmath.log(x) / cmath.log(b)

def Sin() -> tuple[str, complex]:
	z = _rng_float(0, math.pi / 2)
	return f'Sin[{_fmt(z)}]', cmath.sin(z)

def Cos() -> tuple[str, complex]:
	z = _rng_float(0, math.pi / 2)
	return f'Cos[{_fmt(z)}]', cmath.cos(z)

def Tan() -> tuple[str, complex]:
	z = _rng_float(0, math.pi / 3)
	return f'Tan[{_fmt(z)}]', cmath.tan(z)

def ArcSin() -> tuple[str, complex]:
	x = round(random.uniform(0, 1), 2)
	return f'ArcSin[{x}]', complex(math.degrees(math.asin(x)), 0)

def ArcCos() -> tuple[str, complex]:
	x = round(random.uniform(0, 1), 2)
	return f'ArcCos[{x}]', complex(math.degrees(math.acos(x)), 0)

def ArcTan() -> tuple[str, complex]:
	x = round(random.uniform(0, 5), 2)
	return f'ArcTan[{x}]', complex(math.degrees(math.atan(x)), 0)

def Arg() -> tuple[str, complex]:
	z1 = _rng_float(-5, 5)
	if abs(z1) < 1e-9:
		z1 = complex(1, 0)
	rad = round(random.uniform(10, 350), 1) * math.pi / 180
	z2 = z1 * complex(math.cos(rad), math.sin(rad))
	return f'Arg[{_fres(z2)} / {_fmt(z1)}]', complex(rad, 0)

def complex_rotation() -> tuple[str, complex]:
	z = _rng_float(-5, 5)
	if abs(z) < 1e-9:
		z = complex(1, 0)
	deg = round(random.uniform(10, 350), 1)
	rad = math.radians(deg)
	result = z * complex(math.cos(rad), math.sin(rad))
	return f'{_fmt(z)} * Exp[I {deg:.1f} Degree]', result

#
# matrix
#

def determinant_2x2() -> tuple[str, complex]:
	a, b = _rng(-8, 8, imag=False), _rng(-8, 8, imag=False)
	c, d = _rng(-8, 8, imag=False), _rng(-8, 8, imag=False)
	return f'Det{format_matrix([[a, b], [c, d]])}', a * d - b * c

def determinant_3x3() -> tuple[str, complex]:
	for _ in range(100):
		rows = [[_rng(-2, 2, imag=False) for _ in range(3)] for _ in range(3)]
		a, b, c = rows[0]
		d, e, f = rows[1]
		g, h, i = rows[2]
		det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
		if det != 0 and abs(det) <= 20:
			break
	return f'Det{format_matrix(rows)}', complex(det, 0)

def matmul_2x2() -> tuple[str, tuple[complex, complex, complex, complex]]:
	a11, a12 = _rng(-4, 4, imag=False), _rng(-4, 4, imag=False)
	a21, a22 = _rng(-4, 4, imag=False), _rng(-4, 4, imag=False)
	b11, b12 = _rng(-4, 4, imag=False), _rng(-4, 4, imag=False)
	b21, b22 = _rng(-4, 4, imag=False), _rng(-4, 4, imag=False)
	r11 = a11 * b11 + a12 * b21
	r12 = a11 * b12 + a12 * b22
	r21 = a21 * b11 + a22 * b21
	r22 = a21 * b12 + a22 * b22
	A = format_matrix([[a11, a12], [a21, a22]])
	B = format_matrix([[b11, b12], [b21, b22]])
	return f'{A} * {B}', (r11, r12, r21, r22)

def matmul_3x3() -> tuple[str, tuple[complex, ...]]:
	A = [[_rng(-2, 2, imag=False) for _ in range(3)] for _ in range(3)]
	B = [[_rng(-2, 2, imag=False) for _ in range(3)] for _ in range(3)]
	C = [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
	flat = tuple(complex(v, 0) for row in C for v in row)
	return f'{format_matrix(A)} * {format_matrix(B)}', flat

def inverse_2x2() -> tuple[str, tuple[complex, complex, complex, complex]]:
	for _ in range(200):
		a, b = _rng(-4, 4, imag=False), _rng(-4, 4, imag=False)
		c, d = _rng(-4, 4, imag=False), _rng(-4, 4, imag=False)
		det = a * d - b * c
		if abs(det) == 1:
			if det == 1:
				inv = (d, -b, -c, a)
			else:
				inv = (-d, b, c, -a)
			return f'Inverse{format_matrix([[a, b], [c, d]])}', inv
	I = format_matrix([[1, 0], [0, 1]])
	return f'Inverse{I}', (1, 0, 0, 1)

def inverse_3x3() -> tuple[str, tuple[complex, ...]]:
	A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
	ops = []
	for _ in range(random.randint(2, 4)):
		t = random.choice(['add', 'swap', 'neg'])
		if t == 'add':
			i, j = random.sample([0, 1, 2], 2)
			k = random.randint(-2, 2)
			if k == 0: k = 1
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
	inv = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
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
	flat = tuple(complex(v, 0) for row in inv for v in row)
	return f'Inverse{format_matrix(A)}', flat

def _real_eig_2x2(lam_range, P_range, entry_bound):
	for _ in range(200):
		lam1 = random.randint(-lam_range, lam_range)
		lam2 = random.randint(-lam_range, lam_range)
		if lam1 == lam2:
			continue
		p11 = random.randint(-P_range, P_range)
		p12 = random.randint(-P_range, P_range)
		p21 = random.randint(-P_range, P_range)
		p22 = random.randint(-P_range, P_range)
		det_p = p11 * p22 - p12 * p21
		if abs(det_p) != 1:
			continue
		a = det_p * (p11 * p22 * lam1 - p12 * p21 * lam2)
		b = det_p * p11 * p12 * (lam2 - lam1)
		c = det_p * p21 * p22 * (lam1 - lam2)
		d = det_p * (-p21 * p12 * lam1 + p22 * p11 * lam2)
		if all(-entry_bound <= v <= entry_bound for v in (a, b, c, d)):
			return a, b, c, d, lam1, lam2
	return None

def eigenvalues_2x2() -> tuple[str, tuple[complex, complex]]:
	strategy = random.choice(['real_diagonalizable', 'complex_conjugate_pair'])
	if strategy == 'real_diagonalizable':
		result = _real_eig_2x2(5, 3, 8)
		if result:
			a, b, c, d, lam1, lam2 = result
			return f'Eigenvalues{format_matrix([[a, b], [c, d]])}', (complex(lam1, 0), complex(lam2, 0))
		return f'Eigenvalues{format_matrix([[3, 1], [0, 2]])}', (complex(3, 0), complex(2, 0))
	re = random.randint(-3, 3)
	im = random.randint(1, 3)
	M = format_matrix([[re, -im], [im, re]])
	return f'Eigenvalues{M}', (complex(re, im), complex(re, -im))

def eigenvalues_3x3() -> tuple[str, tuple[complex, complex, complex]]:
	obvious = random.randint(-3, 3)
	if obvious == 0:
		obvious = 1
	strategy = random.choice(['real_diagonalizable', 'complex_conjugate_pair'])
	if strategy == 'real_diagonalizable':
		result = _real_eig_2x2(3, 2, 5)
		if result:
			a, b, c, d, lam1, lam2 = result
			rows = [[a, b, 0], [c, d, 0], [0, 0, obvious]]
			return f'Eigenvalues{format_matrix(rows)}', (complex(lam1, 0), complex(lam2, 0), complex(obvious, 0))
		rows = [[3, 1, 0], [0, 2, 0], [0, 0, obvious]]
		return f'Eigenvalues{format_matrix(rows)}', (complex(3, 0), complex(2, 0), complex(obvious, 0))
	re = random.randint(-2, 2)
	im = random.randint(1, 3)
	rows = [[re, -im, 0], [im, re, 0], [0, 0, obvious]]
	return f'Eigenvalues{format_matrix(rows)}', (complex(re, im), complex(re, -im), complex(obvious, 0))

#
# polynomial
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

def Roots() -> tuple[str, list[complex]]:
	degree = random.choices([2,3,4], weights=[90,9,1], k=1)[0]
	roots = []
	remaining = degree
	while remaining > 0:
		if COMPLEX and remaining >= 2 and random.choice([True, False]):
			a = random.randint(-10, 10)
			b = random.randint(1, 10)
			roots.extend([complex(a, b), complex(a, -b)])
			remaining -= 2
		else:
			roots.append(complex(random.randint(-10, 10), 0))
			remaining -= 1

	coeffs = [1]
	i = 0
	while i < len(roots):
		r = roots[i]
		if abs(r.imag) < 1e-9:
			coeffs = _multiply_polynomials(coeffs, [-int(r.real), 1])
			i += 1
		else:
			a = int(r.real)
			b = int(abs(r.imag))
			coeffs = _multiply_polynomials(coeffs, [a * a + b * b, -2 * a, 1])
			i += 2
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

	#Exp,
	#Log,
	#LogBase,
	#Sin,
	#Cos,
	#Tan,
	#ArcSin,
	#ArcCos,
	#ArcTan,
	#Arg,
	#complex_rotation,

	#determinant_2x2,
	#determinant_3x3,
	#matmul_2x2,
	#matmul_3x3,
	#inverse_2x2,
	#inverse_3x3,
	#eigenvalues_2x2,
	#eigenvalues_3x3,

	Roots,
]

def _stop_game(stop_event: threading.Event, timer: threading.Timer) -> None:
	stop_event.clear()
	timer.cancel()

if __name__ == '__main__':
	score = 0
	correct_answer: complex | tuple | list = complex(0, 0)
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

	if correct_answer != complex(0, 0):
		print(f'Answer: {format_final_answer(correct_answer)}')
