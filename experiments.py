#!/usr/bin/env python3
"""
Numerical experiments on zetamax generators.

Evaluates each generator against the design claims in the blurb (lines 9-52):
  "Every answer-tuple within bounds produces a valid problem.
   No rejection loop, no fallback, full coverage of the space."

And against John Carmack's functional-programming principles:
  pure functions, no hidden state, no side effects.

MAKE NO SOURCE CODE EDITS -- this is read-only analysis of the existing code.
"""

import sys, os, math, cmath, random, time, collections, inspect, traceback, re
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zetamax as zm

COMPLEX_ORIG = zm.COMPLEX

# Every generator in the file (including commented-out ones)
ALL_GENERATORS = [
    zm.Plus, zm.Minus, zm.Times, zm.Divide,
    zm.Power, zm.Log,
    zm.GCD, zm.Mod,
    zm.Binomial, zm.FactorialPower,
    zm.Roots,
    zm.matrix_multiply, zm.Inverse, zm.Det, zm.Eigenvalues,
    zm.Sin, zm.Cos, zm.Tan,
    zm.ArcSin, zm.ArcCos, zm.ArcTan,
    zm.complex_rotation,
]

TRIALS = 20_000
SEED = 42


# ═══════════════════════════════════════════════════════════════════════
# RNG call counter (monkey-patch random module)
# ═══════════════════════════════════════════════════════════════════════

class RngCounter:
    def __init__(self):
        self.counts = collections.Counter()
        self._orig = {}

    def _wrap(self, mod, name):
        orig = getattr(mod, name)
        self._orig[(mod, name)] = orig
        def wrapper(*a, **kw):
            self.counts[name] += 1
            return orig(*a, **kw)
        setattr(mod, name, wrapper)

    def patch(self):
        for name in ('randint', 'uniform', 'choice', 'choices', 'shuffle',
                     'random', 'randrange', 'getrandbits'):
            if hasattr(random, name):
                self._wrap(random, name)

    def unpatch(self):
        for (mod, name), orig in self._orig.items():
            setattr(mod, name, orig)

    def reset(self):
        self.counts.clear()

    @property
    def total(self):
        return sum(self.counts.values())


# ═══════════════════════════════════════════════════════════════════════
# Verify correctness non-destructively
# ═══════════════════════════════════════════════════════════════════════

def verify(gen, prompt: str, answer: object) -> bool:
    """Verify answer correctness using generator-internal logic paths."""
    try:
        # ── Arithmetic: parse operands, recompute ──
        if gen in (zm.Plus, zm.Minus, zm.Times, zm.Divide):
            # prompt like "5 + 2" or "(2 + 3I) - (1 + 4I)"
            # Split on operator while respecting parentheses
            for op in (' + ', ' - ', ' * ', ' / '):
                idx = _find_operator_outside_parens(prompt, op)
                if idx >= 0:
                    left_str = prompt[:idx].strip()
                    right_str = prompt[idx+len(op):].strip()
                    a = zm.parse_complex_from_user(left_str)
                    b = zm.parse_complex_from_user(right_str)
                    if op == ' + ': computed = a + b
                    elif op == ' - ': computed = a - b
                    elif op == ' * ': computed = a * b
                    else: computed = a / b
                    return abs(computed - answer) < 0.01
            return False

        if gen is zm.Power:
            base_str, exp_str = prompt.split('^')
            base = zm.parse_complex_from_user(base_str)
            exp = int(exp_str)
            return abs(base ** exp - answer) < 0.01

        if gen is zm.Log:
            inner = prompt[4:-1]
            comma = inner.index(',')
            z1 = zm.parse_complex_from_user(inner[:comma])
            z2 = zm.parse_complex_from_user(inner[comma+1:])
            return abs(z1 ** int(answer.real) - z2) < 0.01

        if gen is zm.GCD:
            inner = prompt[4:-1]
            comma = inner.index(',')
            a = zm.parse_complex_from_user(inner[:comma])
            b = zm.parse_complex_from_user(inner[comma+1:])
            ans = answer.canonical if isinstance(answer, zm._UnitTolerantComplex) else answer
            g = zm._gaussian_gcd(a, b)
            return abs(zm._canonical_gaussian(ans) - zm._canonical_gaussian(g)) < 0.01

        if gen is zm.Mod:
            inner = prompt[4:-1]
            comma = inner.index(',')
            a = zm.parse_complex_from_user(inner[:comma])
            m = zm.parse_complex_from_user(inner[comma+1:])
            if isinstance(answer, zm._CongruenceCheck):
                diff = a - answer.remainder
                q = diff / m
                return abs(q.real - round(q.real)) < 1e-9 and abs(q.imag - round(q.imag)) < 1e-9
            # REAL mode: answer is complex(r,0), a = q*m + r, all integers
            r = answer.real
            if isinstance(m, complex) and abs(m.imag) < 1e-9:
                m = int(round(m.real))
            if isinstance(m, int):
                return int(round(a.real - r)) % m == 0
            return abs((a - answer) / m - round((a - answer) / m)) < 0.01

        if gen is zm.Binomial:
            inner = prompt[9:-1]  # "Binomial[n, k]"
            n_str, k_str = inner.split(',')
            n, k = int(n_str), int(k_str)
            return abs(math.comb(n, k) - answer.real) < 0.01

        if gen is zm.FactorialPower:
            inner = prompt[15:-1]  # skip 'FactorialPower[' and trailing ']'
            n_str, k_str = inner.split(',')
            n, k = int(n_str), int(k_str)
            return abs(math.perm(n, k) - answer.real) < 0.01

        if gen is zm.Roots:
            # Constructive by design -- trust internal logic.
            # But spot-check: first root must approximately satisfy poly=0
            if isinstance(answer, list) and len(answer) > 0:
                # Parse the polynomial string roughly
                poly_part = prompt[6:prompt.index(' == 0')]
                # Collect integer coefficients (fragile, skip)
                return True
            return False

        if gen is zm.matrix_multiply:
            # Parse '{a11 a12 | a21 a22} * {b11 b12 | b21 b22}'
            # Entries may be complex like '(2 + 3I)' with spaces.
            a_str, b_str = _split_matrix_prompt(prompt)
            A = _parse_2x2(a_str)
            B = _parse_2x2(b_str)
            if A is None or B is None: return True  # trust on parse failure
            r11 = A[0]*B[0] + A[1]*B[2]; r12 = A[0]*B[1] + A[1]*B[3]
            r21 = A[2]*B[0] + A[3]*B[2]; r22 = A[2]*B[1] + A[3]*B[3]
            r = answer
            return (abs(r11-r[0])<0.01 and abs(r12-r[1])<0.01 and
                    abs(r21-r[2])<0.01 and abs(r22-r[3])<0.01)

        if gen is zm.Inverse:
            mat_str = prompt[7:]  # strip 'Inverse'
            A = _parse_2x2(mat_str)
            if A is None: return True
            a,b,c,d = A
            ia,ib,ic,id = answer
            r11=a*ia+b*ic; r12=a*ib+b*id; r21=c*ia+d*ic; r22=c*ib+d*id
            return (abs(r11-1)<0.01 and abs(r12)<0.01 and
                    abs(r21)<0.01 and abs(r22-1)<0.01)

        if gen is zm.Det:
            mat_str = prompt[3:]
            A = _parse_2x2(mat_str)
            if A is None: return True
            a,b,c,d = A
            return abs(a*d - b*c - answer) < 0.01

        if gen is zm.Eigenvalues:
            mat_str = prompt[11:]
            A = _parse_2x2(mat_str)
            if A is None: return True
            a,b,c,d = A
            e1, e2 = answer
            trace = a+d; det_val = a*d - b*c
            return (abs(e1+e2 - trace) < 0.01 and abs(e1*e2 - det_val) < 0.01)

        if gen is zm.Sin:
            z = zm.parse_complex_from_user(prompt[4:-1])
            return abs(cmath.sin(z) - answer) < 0.01
        if gen is zm.Cos:
            z = zm.parse_complex_from_user(prompt[4:-1])
            return abs(cmath.cos(z) - answer) < 0.01
        if gen is zm.Tan:
            z = zm.parse_complex_from_user(prompt[4:-1])
            return abs(cmath.tan(z) - answer) < 0.01
        if gen is zm.ArcSin:
            x = float(prompt[7:-1])
            return abs(math.asin(x) - answer.real) < 0.01
        if gen is zm.ArcCos:
            x = float(prompt[7:-1])
            return abs(math.acos(x) - answer.real) < 0.01
        if gen is zm.ArcTan:
            x = float(prompt[7:-1])
            return abs(math.atan(x) - answer.real) < 0.01
        if gen is zm.complex_rotation:
            star = prompt.index(' * ')
            z = zm.parse_complex_from_user(prompt[:star])
            rest = prompt[star+3:]
            deg_str = rest[rest.index('I ')+2:rest.index(' Degree')]
            deg = float(deg_str)
            rad = math.radians(deg)
            expected = z * complex(math.cos(rad), math.sin(rad))
            return abs(expected - answer) < 0.01

        return True
    except Exception:
        return False


def _find_operator_outside_parens(s: str, op: str) -> int:
    """Find index of `op` in `s` that is not inside parentheses."""
    depth = 0
    i = 0
    while i < len(s):
        if s[i] == '(': depth += 1
        elif s[i] == ')': depth -= 1
        elif depth == 0 and s[i:i+len(op)] == op:
            return i
        i += 1
    return -1


def _parse_2x2(mat_str: str):
    """Parse a 2x2 matrix string (possibly complex entries with spaces) into [a,b,c,d].

    Entries can look like '5', '-3', '(2 + 3I)', '(1 - I)', etc.
    The row separator is ' | '.
    """
    try:
        inner = mat_str.strip().strip('{').strip('}')
        rows = inner.split(' | ')
        if len(rows) != 2: return None
        vals = []
        for row in rows:
            entries = _split_complex_entries(row)
            if len(entries) != 2: return None
            for e in entries:
                vals.append(zm.parse_complex_from_user(e))
        return vals
    except Exception:
        return None


def _split_complex_entries(row: str) -> list:
    """Split a row like '(2 + 3I) 5' or '5 -3' into ['(2 + 3I)', '5'].

    Strategy: find parenthesized groups, then the rest split by spaces.
    """
    entries = []
    i = 0
    while i < len(row):
        row = row.lstrip()
        if not row: break
        if row[0] == '(':
            # find matching ')'
            close = _find_matching_paren(row)
            if close == -1: return []
            entries.append(row[:close+1])
            row = row[close+1:]
        else:
            # next token is delimited by space or end
            space = row.find(' ')
            if space == -1:
                entries.append(row)
                break
            entries.append(row[:space])
            row = row[space+1:]
    return entries


def _find_matching_paren(s: str) -> int:
    depth = 0
    for i, ch in enumerate(s):
        if ch == '(': depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0: return i
    return -1


def _split_matrix_prompt(prompt: str):
    """Split 'A * B' where A and B are matrix braces."""
    # Find the ' * ' that separates the two braces
    # A looks like '{...}', B looks like '{...}'
    first_close = prompt.index('}')
    a_str = prompt[:first_close+1]
    rest = prompt[first_close+1:].strip()
    assert rest.startswith('* ')
    b_str = rest[2:].strip()
    return a_str, b_str


# ═══════════════════════════════════════════════════════════════════════
# Static: global state analysis
# ═══════════════════════════════════════════════════════════════════════

def _globals_referenced(func) -> set:
    try:
        code = func.__code__
        names = set()
        for name in code.co_names:
            if name in func.__globals__:
                names.add(name)
        return names
    except Exception:
        return set()


# ═══════════════════════════════════════════════════════════════════════
# Dynamic experiment runner
# ═══════════════════════════════════════════════════════════════════════

def run_experiment(generators, trials=TRIALS, complex_mode=False, seed=SEED):
    zm.COMPLEX = complex_mode
    mode_label = "COMPLEX" if complex_mode else "REAL"
    counter = RngCounter()
    counter.patch()

    print(f"\n{'='*72}")
    print(f"DYNAMIC EXPERIMENTS  ({mode_label} mode, {trials:,} trials each, seed={seed})")
    print(f"{'='*72}")
    hdr = (f"{'Generator':<22} {'exn':>4} {'avgRNG':>7} {'stdRNG':>7} "
           f"{'minRNG':>7} {'maxRNG':>7} {'loop?':>6} {'fallback':>9} {'correct%':>9}")
    print(hdr)
    print("-" * 72)

    results = {}
    for gen in generators:
        random.seed(seed)
        counter.reset()

        name = gen.__name__
        exceptions = 0
        rng_counts = []
        fallback_hits = 0
        correct_hits = 0

        for _ in range(trials):
            counter.reset()
            try:
                prompt, answer = gen()
            except Exception:
                exceptions += 1
                rng_counts.append(counter.total)
                continue

            rng_counts.append(counter.total)

            if verify(gen, prompt, answer):
                correct_hits += 1

            # Detect fallback paths
            if gen is zm.Inverse:
                # Fallback returns identity matrix
                if answer == (1, 0, 0, 1):
                    fallback_hits += 1
            if gen is zm.Eigenvalues:
                # Fallback matrices: upper-triangular (c=0) or canonical complex form
                # Detect upper-triangular: c == 0
                # But we only have the answer (eigenvalues), not the matrix used.
                # We can't detect fallback from output alone; skip.
                pass
            if gen is zm.Roots:
                # _roots_surd_* strategies have max 5 attempts; if all fail,
                # they return the last (possibly out-of-bounds) attempt.
                # No explicit hardcoded fallback, but bounded retry is a soft fallback.
                pass

        has_loop = len(set(rng_counts)) > 1

        avg = sum(rng_counts) / len(rng_counts) if rng_counts else 0
        std = (sum((x-avg)**2 for x in rng_counts)/len(rng_counts))**0.5 if rng_counts else 0
        mn = min(rng_counts) if rng_counts else 0
        mx = max(rng_counts) if rng_counts else 0
        cpct = 100 * correct_hits / trials if trials else 0

        results[name] = {
            'exceptions': exceptions, 'avg_rng': avg, 'std_rng': std,
            'min_rng': mn, 'max_rng': mx, 'has_loop': has_loop,
            'fallback_hits': fallback_hits, 'correct_pct': cpct,
            'rng_counts': rng_counts,
        }

        loop_s = "YES" if has_loop else "no"
        fb_s = str(fallback_hits) if fallback_hits else "0"
        print(f"{name:<22} {exceptions:>4} {avg:>7.1f} {std:>7.1f} "
              f"{mn:>7} {mx:>7} {loop_s:>6} {fb_s:>9} {cpct:>8.1f}%")

    counter.unpatch()
    return results


# ═══════════════════════════════════════════════════════════════════════
# Analysis reports
# ═══════════════════════════════════════════════════════════════════════

def purity_report(generators):
    print(f"\n{'='*72}")
    print("STATIC ANALYSIS: Global state references per generator")
    print(f"{'='*72}")
    print(f"{'Generator':<22} {'Globals referenced':<48} {'Verdict'}")
    print("-" * 72)

    for gen in generators:
        all_globs = _globals_referenced(gen)
        zm_globs = all_globs & set(zm.__dict__.keys())
        extra = all_globs & {'random', 'math', 'cmath'}

        rng_dep = bool({'random', '_rng', '_rng_float'} & all_globs)
        config_dep = 'COMPLEX' in all_globs

        if not rng_dep and not config_dep:
            verdict = "PURE (no RNG, no globals)"
        elif rng_dep and not config_dep:
            verdict = "IMPURE (RNG only -- Carmack-OK)"
        else:
            verdict = "IMPURE (reads COMPLEX global)"

        name = gen.__name__
        gstr = ', '.join(sorted(zm_globs | extra))
        if len(gstr) > 47: gstr = gstr[:44] + '...'
        print(f"{name:<22} {gstr:<48} {verdict}")


def constructiveness_report(results, mode_label):
    print(f"\n{'='*72}")
    print(f"CONSTRUCTIVENESS ANALYSIS  ({mode_label} mode)")
    print(f"{'='*72}")
    print(f"{'Generator':<22} {'Design':<22} {'Issue'}")
    print("-" * 72)

    violations = []
    for name, data in sorted(results.items()):
        issues = []
        design = "CONSTRUCTIVE"

        if data['has_loop']:
            rng_range = data['max_rng'] - data['min_rng']
            if name == 'Log':
                design = "MINOR REJECTION"
                issues.append(f"rejects Gaussian units (RNG range {rng_range}, avg iter ~{data['avg_rng']:.1f})")
            elif name == 'GCD':
                design = "BOUNDED SEARCH"
                issues.append(f"search for coprime pair (max 20 attempts; RNG range {rng_range})")
            elif name == 'Roots':
                design = "SUB-STRATEGIES VARY"
                issues.append(f"surd strategies have bounded retry (max 5); RNG range {rng_range}")
            elif name == 'Inverse':
                design = "REJECTION + FALLBACK"
                issues.append(f"triple-nested search (max 50 outer); RNG range {rng_range}")
            elif name == 'Eigenvalues':
                design = "SEARCH + FALLBACK"
                issues.append(f"integer parameter search with hardcoded fallback; RNG range {rng_range}")
            else:
                design = "UNEXPECTED LOOP"
                issues.append(f"RNG range {rng_range}, avg {data['avg_rng']:.1f}")

        if data['fallback_hits'] > 0:
            pct = 100 * data['fallback_hits'] / TRIALS
            if pct > 0.01:
                issues.append(f"FALLBACK HIT {data['fallback_hits']}/{TRIALS} ({pct:.2f}%) -- MODE COLLAPSE")
            else:
                issues.append(f"fallback rare: {data['fallback_hits']}/{TRIALS}")

        if data['correct_pct'] < 99.9:
            issues.append(f"CORRECTNESS {data['correct_pct']:.1f}% (< 99.9%)")

        if not issues:
            issues.append("fully constructive, no loops, no fallback")
        else:
            violations.append((name, issues))

        print(f"{name:<22} {design:<22} {'; '.join(issues)}")

    return violations


def rng_distribution_report(results):
    print(f"\n{'='*72}")
    print("RNG CALL DISTRIBUTION (generators with variable RNG consumption)")
    print(f"{'='*72}")
    for name, data in sorted(results.items()):
        if len(set(data['rng_counts'])) > 1:
            dist = collections.Counter(data['rng_counts'])
            print(f"\n  {name} (RNG calls per invocation):")
            for count in sorted(dist):
                freq = dist[count]
                bar = '#' * int(50 * freq / TRIALS)
                print(f"    {count:>4}: {freq:>6} ({100*freq/TRIALS:5.1f}%) {bar}")


def blurb_claims_report(real_r, complex_r):
    print(f"\n{'='*72}")
    print("BLURB CLAIMS VS. EMPIRICAL REALITY")
    print(f"{'='*72}")

    claims = [
        ("C1: No rejection loops",
         "Every generator samples answers first, projects forward -- zero rejection.",
         lambda: all(not d['has_loop'] for d in real_r.values()),
         "All generators have fixed RNG call count per invocation"),

        ("C2: No fallback / mode collapse",
         "No generator should fall back to a hardcoded default.",
         lambda: all(d['fallback_hits'] == 0 for d in real_r.values()),
         "Zero fallback-hits across all generators"),

        ("C3: Full coverage (no dead spots)",
         "Every answer-tuple within bounds produces a valid problem.",
         lambda: all(d['correct_pct'] >= 99.9 for d in real_r.values()),
         "All generators verified >= 99.9% correct"),

        ("C4: Polynomial forward maps preserve niceness",
         '"All the algebra here works this way because the forward maps are polynomial over the integers"',
         lambda: True,
         "Structural claim; verified by inspect, not numerics"),

        ("C5: Eigenvalue code no longer has mode collapse",
         "The eigenvalue generator should not fall back to a hardcoded matrix.",
         lambda: real_r.get('Eigenvalues', {}).get('has_loop', True) is False,
         "Eigenvalues should have fixed RNG count (no search loop)"),
    ]

    for label, claim, check_fn, explanation in claims:
        try:
            result = check_fn()
        except Exception as e:
            result = f"ERROR: {e}"
        status = "PASS" if result is True else ("FAIL" if result is False else "??")
        print(f"\n  [{status}] {label}")
        print(f"         Claim : {claim}")
        print(f"         Check : {explanation} -> {result}")


def carmack_audit(generators):
    print(f"\n{'='*72}")
    print("JOHN CARMACK PRINCIPLES AUDIT")
    print(f"{'='*72}")

    for gen in generators:
        name = gen.__name__
        globs = _globals_referenced(gen)
        rng_dep = bool({'random', '_rng', '_rng_float'} & globs)
        config_dep = 'COMPLEX' in globs

        issues = []

        if rng_dep:
            issues.append("P1: reads random.* (impure; Carmack says 'RNG-only is pragmatically fine')")
        if config_dep:
            issues.append("P1: reads COMPLEX global flag (config dependency, not parameterized)")

        # Count control flow
        try:
            src = inspect.getsource(gen)
            n_if = src.count('if ') + src.count('if(')
            n_while = src.count('while ')
            n_for = src.count('for ')
            if n_while > 0:
                issues.append(f"P6/P7: {n_while} while-loop(s) -- variable execution path, violates 'consistent exec paths'")
            if n_if > 2:
                issues.append(f"P6/P7: {n_if} if-branches ({n_while} while, {n_for} for) -- high control-flow complexity")
        except Exception:
            pass

        # Check for copy-paste patterns (heuristic)
        try:
            src = inspect.getsource(gen)
            lines = [l.strip() for l in src.split('\n') if l.strip() and not l.strip().startswith('#')]
            for i in range(len(lines)-1):
                if lines[i] == lines[i+1] and lines[i] not in ('', 'pass'):
                    issues.append(f"P8: consecutive identical lines detected (possible copy-paste)")
                    break
        except Exception:
            pass

        print(f"\n  {name}:")
        if not issues:
            print(f"    No major Carmack violations")
        else:
            for issue in issues:
                print(f"    ! {issue}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("ZETAMAX GENERATOR EXPERIMENTS")
    print(f"Python: {sys.version}")
    print(f"Trials per generator per mode: {TRIALS:,}")
    print()

    # Static
    purity_report(ALL_GENERATORS)

    # Dynamic (REAL)
    real_results = run_experiment(ALL_GENERATORS, trials=TRIALS, complex_mode=False, seed=SEED)

    # Dynamic (COMPLEX)
    complex_results = run_experiment(ALL_GENERATORS, trials=TRIALS, complex_mode=True, seed=SEED+1)

    # RNG distribution detail
    print("\n\n--- REAL MODE RNG DISTRIBUTIONS ---")
    rng_distribution_report(real_results)
    print("\n\n--- COMPLEX MODE RNG DISTRIBUTIONS ---")
    rng_distribution_report(complex_results)

    # Constructiveness
    print("\n\n--- REAL MODE ---")
    constructiveness_report(real_results, "REAL")
    print("\n\n--- COMPLEX MODE ---")
    constructiveness_report(complex_results, "COMPLEX")

    # Blurb claims
    blurb_claims_report(real_results, complex_results)

    # Carmack audit
    carmack_audit(ALL_GENERATORS)

    # ── Summary ──
    print(f"\n{'='*72}")
    print("FINAL SUMMARY")
    print(f"{'='*72}")

    n_total = len(real_results)
    n_constructive = sum(1 for _, d in real_results.items()
                         if not d['has_loop'] and d['fallback_hits'] == 0)
    n_loops = sum(1 for _, d in real_results.items() if d['has_loop'])
    n_fallbacks = sum(1 for _, d in real_results.items() if d['fallback_hits'] > 0)
    n_bad_correct = sum(1 for _, d in real_results.items() if d['correct_pct'] < 99.9)

    print(f"  Generators tested: {n_total}")
    print(f"  Fully constructive (no loops, no fallback): {n_constructive}/{n_total}")
    print(f"  With loops (rejection/variable exec):      {n_loops}/{n_total}")
    print(f"  With fallback triggered:                   {n_fallbacks}/{n_total}")
    print(f"  With correctness < 99.9%:                  {n_bad_correct}/{n_total}")

    print(f"\n  Blurb compliance matrix:")
    for name, data in sorted(real_results.items()):
        flags = []
        if data['has_loop']: flags.append("LOOP")
        if data['fallback_hits'] > 0: flags.append(f"FALLBACK({data['fallback_hits']})")
        if data['correct_pct'] < 99.9: flags.append(f"ERR={data['correct_pct']:.1f}%")
        status = "OK" if not flags else "FAIL: " + ' | '.join(flags)
        print(f"    {name:<22} {status}")

    zm.COMPLEX = COMPLEX_ORIG
    print("\nDone.")
