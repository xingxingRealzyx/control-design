# Control design library (C++17)

Controller design and analysis for the host side of an embedded workflow: model
inspection, discretization, and (next) gain synthesis, with the numbers a
firmware engineer needs printed or exported rather than described.

C++17, Eigen, Google style with four-space indentation. No C output on purpose:
the deliverable of this library is a *design*, and the design is consumed as a
small C++17 header, not as generated C.

Sibling project: [kalman-filter](https://github.com/xingxingRealzyx/kalman-filter)
covers the estimation half (KF / EKF / IEKF / UKF). This one covers the design
half, so a full workflow reads:

```
model -> analyse -> discretize -> design gains -> export -> firmware
                        ^^^^^^^^^^^^^^^^^^^^^^^^  this repository
```

## What is here now, and what is not

| module | state |
|---|---|
| `control/types.h` | state-space container, validation, method enumerations - **done** |
| `control/lti.h` | poles, zeros, stability, margins, DC gain, frequency response, Lyapunov solvers, Gramians, Hankel values, balanced truncation - **done** |
| `control/discretize.h` | ZOH (van Loan), Tustin, forward/backward Euler, matched pole-zero, sample-time rule, Padé delay, series connection, conversion report - **done** |
| `control/analysis.h` | controllability/observability ranks, PBH per mode, Kalman decomposition, minimal realisation - **done** |
| `control/design.h` | pole placement (Ackermann), LQR/DLQR, observer by duality, Kalman steady-state gain, integral action, reference feedforward - **done** |
| `control/codegen.h` | export the design as a deployable C++17 controller header - **done** |

The pipeline is complete end to end:

```
model -> analyse -> discretize -> design -> emit -> firmware
```

`examples/generate_controller.cpp` emits the deployable header, and the test
suite compiles that emitted header and checks it against an independent
implementation (see "The generated code is verified, not trusted").

Build and test:

```bash
cmake --preset release            # or: debug, ci, sanitizers
cmake --build --preset release -j
ctest --preset release            # 58 tests, two executables

./build/release/design_workflow_example
```

## Why "not controllable" is three different answers

`AnalyseControllability` does not return a boolean. It distinguishes:

1. **fully controllable** - arbitrary pole placement is available;
2. **stabilizable** - some state is unreachable, but every unreachable mode is
   already stable: a stabilising controller exists and the offending states can
   only be left to decay;
3. **unstabilizable** - an unstable mode the input cannot influence. No
   controller fixes this. It is a statement about where the actuator is, and the
   tool says so instead of producing gains for a plant that will diverge.

The classification is per mode (PBH test, smallest singular value of
`[A - lambda I, B]`), not by matrix rank, because "the mode at +1.0 is
unreachable" is actionable and "rank 3 < 4" is not. The dual statements hold for
observability (`detectable` vs `undetectable`), and a *marginal* result is
reported separately: a system that is controllable only through a `1e-8` entry is
full rank in exact arithmetic and wants gains of order `1e8` in practice.

## The generated code is verified, not trusted

`control/codegen.h` emits a self-contained C++17 header: no exceptions, no RTTI,
no heap, no recursion, all state in the object, compile-time dimensions so every
loop has a fixed trip count, one `using Real = ...` to switch the target between
float and double, saturation and a NaN guard on the input, and a provenance
comment on every gain.

None of that is taken on trust. `generate_controller` runs at **build time** and
its output is compiled into `generated_controller_test`, which runs the emitted
class against a separately written Eigen reference over the same input sequence -
3000 ticks covering a step, a reversal and a segment that drives the actuator
into its limit - and compares the commands:

| check | result |
|---|---|
| ticks compared | 3000 |
| ticks with the actuator saturated | 237 |
| peak command | 20.000000 N (the limit) |
| largest deviation from the reference | **3.553e-15** (relative 1.776e-16) |

That is round-off, against a protocol tolerance of 1e-9 relative for double. The
comparison is the point: the emitted code is hand-unrolled loops over
hand-indexed arrays, and a transposed index or an inverted sign compiles cleanly
and produces a controller that is *almost* right - the kind that survives review
and fails on hardware. The integrator state is compared too, not just the
command, because a divergence there can stay invisible in the output for a long
time.

Two things the emit path refuses, because the design and the code must not drift:

* a gain with n+1 columns (a design that included integral action) is rejected by
  the plain entry point rather than emitted with the integral gain dropped;
* a continuous model is rejected outright - the emitted observer steps with the
  plant's own matrices, so the caller has to hand over the sampled model the
  firmware will actually run against.

The float target gets specific attention: literals are emitted at 17 significant
digits and wrapped in `static_cast<Real>(...)`, because `float k[]{0.30000000000000004}`
is a narrowing conversion and a build that fails in the field over one awkward
gain is not a hypothetical.

## What the implementation promises

* **A refused answer instead of a wrong one.** `PolesAndZeros` throws for a
  discrete model and for a non-square model rather than returning numbers it
  cannot verify. The reason is in the source: zero-order hold introduces sampling
  zeros (a minimum-phase continuous plant commonly becomes non-minimum-phase once
  sampled), and the pencil formulation used here does not reproduce them. A wrong
  zero silently corrupts a bandwidth or inversion calculation downstream, so it
  is refused.
* **A conversion report, not a conversion.** `Convert` returns the method used,
  whether it is exact for the class of input it assumes, the largest deviation of
  the discrete step response from the continuous one, and which continuous poles
  a cheap method pushed outside the unit circle. `ForwardEuler` at 1.5 s on a
  pole at -2 maps it to -2 and says so; discovering that on hardware is how a
  commissioning afternoon disappears.
* **A sample time derived from a rule.** `RequiredSampleTime(bandwidth, 20)` is
  the only place a sample rate comes from, because "20 samples per period of the
  fastest frequency of interest" is a sentence that can be argued with, and a
  number pulled from the air cannot.
* **Residual-based tests.** Every numerical routine is asserted against the
  equation it claims to solve (Lyapunov residuals, the Padé polynomial identity,
  the closed-form ZOH matrices, the analytic transfer function), not against a
  previously recorded output. Recorded outputs freeze bugs in place.

## Measured behaviour, not claims

The example designs a DC motor speed loop; the discretization is checked against
the closed form.

| check | result |
|---|---|
| ZOH `Ad`, `Bd` vs closed form for `1/(s^2+3s+2)` | error `1.1e-16` |
| ZOH pole mapping | `e^{-Ts}` and `e^{-2Ts}` to `1e-12` |
| ZOH step-response deviation from continuous | `0.0` (exact at the sample instants) |
| Tustin at 50 samples/s, deviation | `< 1e-3` |
| Padé `n=1..3` DC gain | `1.000000000` |
| Padé `n=3` phase at 1 rad/s | `-5.731 deg` (exact `-5.730`) |
| balanced truncation of a 3rd-order system, DC gain | `1.020002` vs `1.020010` full |
| forward Euler at `Ts = 1.5 s` | destabilized pole reported, not hidden |
| LQR on the double integrator vs the analytic solution | P and K match to `3e-11` |
| CARE residual | `5.7e-14` |
| DARE residual (this implementation) | `5.0e-13` (scipy's published value: `2.1e-9`) |
| pole placement, observer, integral gain vs scipy reference | exact to `1e-9` |
| generated controller vs the reference over 3000 ticks | `3.6e-15` |
| DLQR closed-loop spectral radius | `0.8149195372` |
| PBH measure, reachable vs unreachable mode | `> 1e-3` vs `< 1e-12` |
| Kalman decomposition, four subspace dimensions | sum to `n` in all three test systems |
| minimal realisation, frequency response vs full model | `< 1e-10` relative over three decades |

## Bugs this repository's tests caught while it was written

Recorded because each one was invisible in a plot and would have cost real time:

1. **The Padé delay was the reciprocal of the right answer.** The first version
   approximated `exp(+sT)`, giving a DC gain of 0.95 and a positive phase; the
   coefficient direction is invisible unless the phase is checked against `-wT`.
2. **The Padé realization was wrong in the output matrix.** Three attempts at the
   companion-form coefficient mapping failed (DC gain 0.9, then -0.8, then 1.1
   instead of 1.0). The working version builds the numerator and denominator from
   binomial coefficients and solves for `C`, `D` from the polynomial identity
   `(C (sI-A)^-1 B + D) D(s) = N(s)`, which is checkable, instead of a table of
   indices, which is not.
3. **A zero-finding rank test that kept spurious eigenvalues.** It reported a zero
   for a plant that has none. Now the left/right null vectors must genuinely
   annihilate `B` and `C`, and the search is scoped to SISO continuous models.
4. **`BalancedTruncation` lost 2 percent of the DC gain** because the balancing
   transformation was assembled wrong - the reduced model still had the right
   order and was still stable, so only the accuracy check caught it.
5. **`Series` composed the matrices wrong** (`C2*D1` instead of `D2*D1 + D2`).
6. **The Kalman decomposition summed to more than `n`.** Two separate faults: a
   loop that tested candidate direction `i` but copied direction `j` (so nothing
   was ever rejected), and a relative rank test applied to a *nearly zero*
   projection - which always finds a unit direction, because its own largest
   singular value sets the scale. The intersection of two subspaces meeting only
   at the origin came back one-dimensional. Fixed by deciding "this projection is
   negligible" before asking for a basis, against the size of the subspace being
   projected.
7. **The Ackermann polynomial was evaluated backwards.** `coefficients(0)` holds
   the *leading* coefficient, and the first version multiplied it by `A^0`. For
   poles -2, -3 that computes `6A^2 + 5A + I` instead of `A^2 + 5A + 6I`, giving
   K = [-11, -13] where [4, 2] was asked for - and an *unstable* closed loop. The
   achieved-pole comparison caught it; the pole list is reported rather than
   assumed for exactly this reason.
8. **Complex poles were collapsed onto the real axis.** The characteristic
   polynomial was built using `pole.real()` only, so `-1 +/- 2j` produced
   `(s+1)^2` instead of `(s+1)^2 + 4`: K = [1, 2] instead of [5, 2]. The loop was
   still stable, which is why nothing but an exact-value check would notice. The
   product is now carried out in complex arithmetic and a non-conjugate request
   is refused.
9. **A `JacobiSVD` was used without `ComputeThinU`/`ComputeThinV`**, which aborts
   in a debug build and reads out of bounds in a release build. Found by the test
   suite segfaulting, which is exactly why the test framework has a `KCTL_TRACE`
   switch: when a test crashes the process instead of failing an assertion,
   buffered stdout is lost and there is no other way to tell which test died.

Cross-checking: the Kalman subspace dimensions are verified against an
independent NumPy implementation of the same projection-based intersection, and
that reference is what caught two wrong expectations in the tests - the same
algorithm written twice is what makes a dimension count trustworthy.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

* Ubuntu and macOS, each in `ci` (`-Werror`) and `debug`;
* AddressSanitizer + UndefinedBehaviorSanitizer on Ubuntu.

Everything is driven by the same `CMakePresets.json` used locally. The Linux
matrix is also reproduced locally with `gcc:13` containers on `linux/amd64` and
`linux/arm64` while iterating, because a second compiler finds things the first
one does not: the sibling project's `-Wshadow` and over-tight-threshold failures
were both GCC-only.
