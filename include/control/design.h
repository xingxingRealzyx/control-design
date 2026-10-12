// Copyright 2024 The Control Design Library Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CONTROL_DESIGN_H_
#define CONTROL_DESIGN_H_

// Synthesis: given a model, produce gains.
//
//   pole placement   - put the closed loop exactly where you want it
//   LQR              - let a cost function choose, and get guaranteed margins
//   observer         - the dual of pole placement, for estimating state
//   integral action  - remove steady-state error against a constant reference
//   reference gain   - feedforward that makes the output track without offset
//
// Every entry point validates first and refuses when the model cannot support
// the request (an uncontrollable mode cannot be placed, an unobservable mode
// cannot be observed, an unstable-mode check uses the analysis module so the
// refusal names the offending eigenvalue).
//
// What the caller gets back is a `DesignResult`: the gains, plus the numbers
// that decide whether the design is deployable - achieved closed-loop poles,
// their damping, the gain magnitudes, and any warning. A gain vector on its own
// is not enough to judge a design.

#include <Eigen/Core>
#include <complex>
#include <string>
#include <vector>

#include "control/analysis.h"
#include "control/types.h"

namespace control {

struct DesignOptions {
    AnalysisOptions analysis;
    // Pole placement on a nearly uncontrollable model produces enormous gains.
    // When the requested poles need a gain larger than this, the result is
    // returned with a warning rather than silently.
    double gain_warning_threshold = 1e5;
    // Iteration limit for the iterative discrete Riccati solver.
    int riccati_max_iterations = 20000;
    double riccati_tolerance = 1e-14;
};

struct DesignResult {
    // u = -K x for state feedback; u = -K [x; integral of error] for the
    // augmented design; e_hat = L (y - C x_hat) for an observer.
    Matrix gain;
    // Solution of the Riccati equation, when the design came from one (LQR).
    Matrix riccati_solution;
    // Achieved closed-loop poles, sorted by dominance. For pole placement this
    // should match the request; reporting the achieved value rather than the
    // requested one is the difference between a design and a hope.
    ComplexVector closed_loop_poles;
    // Damping ratios of those poles (continuous-time equivalent for a discrete
    // design), in the same order.
    Vector damping_ratios;
    // Largest absolute gain entry: the number that decides whether the design
    // survives contact with a 12-bit DAC or a saturation limit.
    double gain_magnitude = 0.0;
    bool ok = false;
    std::vector<std::string> warnings;
    std::string notes;
};

// Exact pole placement, by Ackermann's formula. Works for any controllable
// `n`-state model with a single input; for a multi-input model it throws and
// says so, because the multi-input problem has a family of solutions (the extra
// freedom is what the LQR weights are for).
DesignResult PlacePoles(const LtiSystem& system,
                        const std::vector<std::complex<double>>& desired_poles,
                        const DesignOptions& options = {});

// Continuous-time LQR: minimise the integral of x' Q x + u' R u, giving
// u = -K x with K = R^-1 B' P and P the stabilising solution of the CARE
//
//   A' P + P A - P B R^-1 B' P + Q = 0.
//
// Requires (A, B) stabilizable and Q, R symmetric with R positive definite and
// Q positive semidefinite; each is checked rather than assumed.
DesignResult LinearQuadraticRegulator(
    const LtiSystem& system, const Matrix& state_weight,
    const Matrix& input_weight, RiccatiMethod method = RiccatiMethod::kAuto,
    const DesignOptions& options = {});

// Discrete-time LQR for a discrete model: minimise sum of x' Q x + u' R u,
// u = -K x with K = (R + B' P B)^-1 B' P A and P solving the DARE
//
//   A' P A - P - A' P B (R + B' P B)^-1 B' P A + Q = 0.
DesignResult DiscreteLinearQuadraticRegulator(
    const LtiSystem& system, const Matrix& state_weight,
    const Matrix& input_weight, RiccatiMethod method = RiccatiMethod::kAuto,
    const DesignOptions& options = {});

// Solves the CARE / DARE and returns the stabilising P. Exposed because P is
// worth inspecting on its own (its diagonal is the cost-to-go per state, which
// is how an LQR design is usually sanity-checked) and because the residual is
// what the tests assert.
bool SolveContinuousRiccati(const Matrix& a, const Matrix& b, const Matrix& q,
                            const Matrix& r, Matrix* p,
                            RiccatiMethod method = RiccatiMethod::kAuto,
                            const DesignOptions& options = {});
bool SolveDiscreteRiccati(const Matrix& a, const Matrix& b, const Matrix& q,
                          const Matrix& r, Matrix* p,
                          RiccatiMethod method = RiccatiMethod::kAuto,
                          const DesignOptions& options = {});

// Observer gain by duality: place the poles of (A - L C) by running pole
// placement on (A', C'). Works for a single-output model; a multi-output model
// has the same family-of-solutions freedom as multi-input pole placement.
DesignResult PlaceObserverPoles(
    const LtiSystem& system,
    const std::vector<std::complex<double>>& desired_poles,
    const DesignOptions& options = {});

// Discrete steady-state Kalman gain, for the same model the estimation library
// uses (x+ = A x + B u + w, y = C x + v). Provided here because an LQG design
// is two halves and this is the half that belongs with the controller gains;
// the recursive filter itself lives in the sibling kalman-filter project.
// Returns the gain and the prior covariance by the same iterative route as the
// DARE solver, stopping when the gain stops moving.
bool SolveDiscreteSteadyStateGain(const Matrix& a, const Matrix& c,
                                  const Matrix& process_noise,
                                  const Matrix& measurement_noise, Matrix* gain,
                                  Matrix* prior_covariance,
                                  const DesignOptions& options = {});

// Adds an integrator on the tracking error: the augmented model has state
// [x; z] with z' = r - y (continuous) or z+ = z + (r - y) (discrete), so a
// constant reference is tracked with zero steady-state error. The returned gain
// acts on the augmented state.
//
// `output_to_integrate` selects which output is integrated (0 for a SISO
// model); an integral channel on an output that no input can influence is
// refused.
struct AugmentedDesign {
    LtiSystem augmented;  // the model the gain acts on
    DesignResult design;  // gain on [x; z]
    int integral_channels = 1;
};
AugmentedDesign AddIntegralAction(
    const LtiSystem& system, int output_to_integrate,
    const std::vector<std::complex<double>>& poles,
    const DesignOptions& options = {});
AugmentedDesign AddIntegralActionLqr(const LtiSystem& system,
                                     int output_to_integrate,
                                     const Matrix& state_weight,
                                     const Matrix& integral_weight,
                                     const Matrix& input_weight,
                                     const DesignOptions& options = {});

// Feedforward gain that makes the output reach the reference at steady state:
// solve the equilibrium equations for [x_ss; u_ss] and take N = u_ss. Returns
// false when the equilibrium does not exist (a singular plant at DC), which is
// the case an integrator is for.
bool ReferenceFeedforward(const LtiSystem& system, double reference,
                          Matrix* gain, Vector* state_at_reference = nullptr);

}  // namespace control

#endif  // CONTROL_DESIGN_H_
