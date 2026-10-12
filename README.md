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
| `control/pid.h` | tuning from a model (critical proportion, pole placement), margins, and the runtime with derivative-on-measurement, filtering, anti-windup and rate limiting - **done** |
| `control/mpc.h` | receding-horizon QP in velocity form: prediction matrices, terminal cost, input and slew constraints, bounded solve - **done** |
| `control/adrc.h` | linear active disturbance rejection: bandwidth-parameterized ESO, disturbance cancellation, noise-gain reporting - **done** |

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
ctest --preset release            # 88 tests, two executables

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

## ADRC: one number with a physical meaning, and two bandwidths

`control/adrc.h` implements linear active disturbance rejection control. Rather
than model the plant well enough to invert it, it treats everything unmodelled -
parameter error, external disturbance, unmodelled coupling - as one additive term
entering through the input, estimates it with an extended state observer, and
cancels it. The design needs `b0` (the input gain) and two bandwidths.

The observer gains come from a single bandwidth so that the estimation-error
dynamics have **every pole at `-wo`**:

```
beta = [2 wo, wo^2]          2nd-order ESO   (first-order plant)
beta = [3 wo, 3 wo^2, wo^3]  3rd-order ESO   (second-order plant)
```

That identity is checked **algebraically** in the test suite: "every pole at
`-wo`" is exactly the statement that the characteristic polynomial is
`(lambda + wo)^m`, which is exactly the statement that `beta_i = C(m, i+1) wo^(i+1)`.
It is not checked through an eigensolver, because a repeated root makes the matrix
defective and the eigenvalue error then grows like the m-th root of round-off
(about `1e-5` for a triple root) - the test would be measuring the eigensolver
rather than the design. The observer's *speed* is verified in the time domain
instead, where a defective root is harmless.

What the tests establish, with the measured numbers:

| claim | result |
|---|---|
| observer poles, both ESO orders | exactly `-wo` (algebraic identity) |
| discrete observer = exact ZOH of the continuous one | agrees with a fine integrator to `1e-5` (the integrator's own error) |
| constant disturbance rejection, first-order plant | steady-state error `~1e-4`; a proportional controller matched to the same bandwidth leaves `>0.05` |
| tracking approaches the ideal `wc/(s+wc)` as `wo/wc` grows | settling `1.477 s` at ratio 3 against an ideal `1.000 s`, no overshoot |
| second-order law follows `zeta` | `zeta = 1` gives `0.17 %` overshoot, `zeta = 0.5` gives `18.2 %` (theory: 16.3 %) |
| `b0` wrong by a factor of two | still settles, zero steady-state error, speed spread under 10 % |

That last row is the property ADRC is chosen for, and it is worth being precise
about what the test does *not* claim: the ordering of the settling times under a
wrong `b0` is reported rather than asserted, because measured it is **not** the
naive prediction (a larger `b0` settles marginally faster, since the observer's own
model gain moves with `b0` and dominates the transient). Asserting the naive
ordering would have been asserting a guess about a subtle interaction.

`AnalyseAdrc` reports the **noise gain** - the multiple of the controller bandwidth
by which measurement noise reaches the control signal - because that is the real
cost of raising `wo`, and a trade with a number attached is a trade that can be
argued with.

The observer is discretized by **exact zero-order hold**, not Euler: at the
bandwidths ADRC runs at, an Euler observer can be unstable for a sample time the
exact one handles.

## PID: the four structures that are not optional

`PidController` implements all four of the things a PID needs to survive an
actuator, because a badly *structured* PID looks perfectly tuned in simulation and
winds up, overshoots and chatters on hardware:

1. **Derivative on the measurement, negated** - never on the error, so a setpoint
   step does not differentiate into a spike. The test asserts the first command
   after a step is `Kp*e` exactly, with `Kd = 100` in place.
2. **Derivative filtering** with `Td/N`, `N` configurable (8..20 is the usual
   range, lower for a noisy sensor).
3. **Anti-windup by clamping**: the integrator is held while the output is
   saturated in the direction the error would push it, plus an independent bound
   on the integral term. The test drives 500 ticks of a 100-unit error into a
   +/-1 actuator and checks the integral term stays bounded (without the hold it
   would reach 5000).
4. **Output and rate limiting**, with `saturated()` and `rate_limited()` reported
   so the caller can see which one is active.

Tuning comes from a model rather than only from an experiment:

* `TuneZieglerNicholsFromModel` finds the ultimate gain and period at the phase
  crossover, and **refuses** a plant whose phase never reaches -180 degrees
  instead of inventing a gain - a first-order lag has no ultimate gain, and this
  method does not apply to it. Verified against the textbook case `1/(s+1)^3`,
  where `Ku = 8` and `Pu = 2 pi / sqrt(3)`.
* `TunePolePlacement` matches coefficients exactly for a first-order plant with PI
  and a second-order plant with PID, refusing other shapes with a pointer to
  `PlacePoles`/`LinearQuadraticRegulator` rather than approximating silently.

One honest caveat is built into the API: **pole placement places poles, and the
controller's own zeros still shape the response.** Asking for 10 percent overshoot
on a second-order plant yields about 27 percent, because the PID introduces zeros
the characteristic polynomial matching knows nothing about. So `PidDesign` carries
the **simulated** overshoot and settling time and warns when they miss the
request - which is the difference between a design note and a surprise on the
test bench.

## MPC: verified by a theorem, not by a step response

`control/mpc.h` uses the velocity form (the decision variable is the input
increment), which puts integral action into the structure rather than into an
extra state someone has to remember to add. The prediction stacks into
`Y = Phi xi + Gamma dU` and the cost becomes a small dense QP per tick.

The claim that makes it trustworthy is checkable without a simulation:

> With the terminal cost taken from the DARE solution of the **augmented** problem
> and the stage cost covering the steps the terminal cost does not, the
> unconstrained optimum **equals the LQR gain for any horizon**.

That holds here to `5.6e-9` at horizons 1, 5 and 20 - the residual being the DARE's
own convergence tolerance. Three separate mistakes were caught by exactly this
test, all of which produced a stable controller and would have passed any step-
response check:

1. the terminal cost was computed from the **plant's** DARE instead of the
   augmented one (the cost penalises `du`, so the equivalent LQR problem is on the
   augmented model);
2. the terminal cost was **never added to the QP** at all - the design stored it
   and then ignored it;
3. the stage cost covered steps `1..N` while the terminal cost prices step `N`,
   double-counting that step.

A fourth bug was found by the tracking test rather than the equivalence test, and
is the most instructive: the terminal penalty was measured from the **origin**, so
for a non-zero reference the stage cost pulled the output towards `r` while the
terminal cost pulled the state towards `0`. The loop settled quietly at
`y = 1.034` for a reference of `2.0` - a wrong answer that looks exactly like a
converged one. The penalty is now measured from the equilibrium that holds the
reference, solved from the same `[(A-I) B; C 0]` system the feedforward uses, and
tracking is exact (`y = 2.0000000000`, `u = 4.0000000000`).

Constraints, with the guarantee stated precisely rather than implied:

| constraint | guarantee | verified |
|---|---|---|
| input increment (slew) | exact, enforced inside the QP | `max abs(du) = 0.5000000000` for a limit of 0.5 |
| absolute input | exact on the command that leaves the controller | `peak abs(u) = 1.0000000000` for a limit of 1.0 |
| solver | bounded iterations, convergence reported | worst case 240 of 500, all converged |

The distinction in that table is deliberate. The absolute limit is applied as a
saturation on the outgoing command (the standard embedded compromise) rather than
inside the QP; doing both exactly needs a projection onto a band in the cumulative
sum, which is a QP of its own. The violation is zero either way, and the move is
not the exact constrained optimum while the limit binds - so the header says
which of the two is guaranteed instead of leaving it to be discovered.

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
9. **The PID pole-placement characteristic polynomial was missing its leading
   `s`.** The gains came out right (`Kp = 1.9`, `Ki = 4.5`) while the reported
   poles were the roots of the wrong equation, because the plant denominator was
   never multiplied by `s`. A test that checked only the gains would have passed.
10. **The MPC terminal cost was measured from the origin**, so a non-zero
   reference settled at a compromise between the stage cost and the terminal cost
   (see above).
11. **The ADRC observer matrix was built with its gains on the diagonal**
   (`diag(-b1, -b2)`) instead of in the first column with the integrator chain on
   the superdiagonal. The eigenvalues were then `-b_i` rather than `-wo` (measured:
   `-40` and `-400` instead of `-20` twice), so the disturbance estimate was wrong
   and the loop settled at `0.548` for a reference of `1.0`. One mistake, six
   failing tests - and the identity the design is *named* for is what caught it.
12. **The `ci` preset never enabled `-Werror`.** `CMakePresets.json` set
   `KALMAN_WERROR` while `CMakeLists.txt` read `CONTROL_WERROR`, so the preset
   silently did nothing and CMake printed a discarded `unused-cli` warning. All the
   `-Werror` verification in this repository's history was done by passing
   `-DCONTROL_WERROR=ON` explicitly or in the containers, so the code is
   warning-free - but the *guarantee* in CI was weaker than it appeared. Found by
   the agent scaffolding the sibling `sysid` repository, which noticed the discarded
   warning while copying the presets. Fixed by renaming the preset entry.
13. **A `JacobiSVD` was used without `ComputeThinU`/`ComputeThinV`**, which aborts
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
