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

#ifndef CONTROL_LTI_H_
#define CONTROL_LTI_H_

// Properties of a state-space model, without designing anything: poles, zeros,
// stability, DC gain, frequency response, and the two Lyapunov equations (which
// also give the controllability/observability Gramians).
//
// The split from control/analysis.h is deliberate: everything here answers
// "what is this system", while analysis.h answers "what can be done to it"
// (controllability, observability, canonical forms).

#include <Eigen/Core>
#include <complex>
#include <vector>

#include "control/types.h"

namespace control {

// Eigenvalues of A, sorted by descending real part (continuous) or descending
// magnitude (discrete) so that the dominant mode comes first.
ComplexVector Poles(const LtiSystem& system);

// Transmission zeros of a strictly proper system, from the generalized
// eigenvalue problem on the system pencil
//
//   [ A - sI   B ] [x]     [0]
//   [   C      0 ] [u]  =  [0]
//
// i.e. the finite values of s where the pencil loses rank. A system with more
// outputs than inputs generally has no zeros; those infinite zeros are not
// reported. This is the definition that matters for control (a zero in the
// right half plane limits achievable bandwidth and forbids exact inversion).
PoleZeroResult PolesAndZeros(const LtiSystem& system);

// Stability of A (continuous: all poles strictly in the left half plane;
// discrete: all poles strictly inside the unit circle). A pole exactly on the
// boundary counts as not stable, which is the useful convention for a design
// tool: a marginally stable pole will not decay.
bool IsStable(const LtiSystem& system);

// Signed distance of the dominant pole from the stability boundary: the decay
// rate -max Re(p) for continuous time, or 1 - max|p| for discrete time.
// Positive means stable, and its size is how much margin there is.
double StabilityMargin(const LtiSystem& system);

// Damping ratios and natural frequencies of the poles, most damped first.
// Continuous time treats a pole as s = -zeta wn +/- j wn sqrt(1 - zeta^2);
// discrete time reports the equivalent continuous-time values through
// s = log(z) / Ts. Meant for design reporting, not for control logic.
struct PoleCharacteristics {
    ComplexVector poles;
    Vector damping_ratios;       // zeta, in [0, 1] for a complex pair
    Vector natural_frequencies;  // wn [rad/s], continuous-time equivalent
    bool is_continuous = true;
};
PoleCharacteristics DescribePoles(const LtiSystem& system);

// Steady-state gain. Continuous: -C A^-1 B + D (infinite when A is singular).
// Discrete: C (I - A)^-1 B + D (infinite when I - A is singular).
Matrix DcGain(const LtiSystem& system);

// Frequency response at one frequency. Continuous: C (sI - A)^-1 B + D with
// s = j w. Discrete: C (zI - A)^-1 B + D with z = exp(j w Ts), where w is in
// rad/s and Ts comes from the system. Returns a p x m complex matrix.
Eigen::MatrixXcd FrequencyResponse(const LtiSystem& system, double omega);

// Frequency response on a grid, for plotting or for margin computation.
struct FrequencyResponseData {
    Vector omega;  // rad/s
    std::vector<Eigen::MatrixXcd> response;
};
FrequencyResponseData FrequencyResponseGrid(const LtiSystem& system, int points,
                                            double omega_max);
FrequencyResponseData FrequencyResponseGrid(const LtiSystem& system,
                                            const Vector& omega);

// Singular values of the frequency response (sigma plot), one row per
// frequency, one column per singular value.
Matrix FrequencyResponseSingularValues(const LtiSystem& system, int points,
                                       double omega_max);

// Solves the continuous Lyapunov equation  A X + X A^T + Q = 0.
// Returns false when A has an eigenvalue pair summing to zero (no unique
// solution), which is the case a design tool has to refuse rather than answer
// with noise.
bool SolveLyapunov(const Matrix& a, const Matrix& q, Matrix* x,
                   RiccatiMethod method = RiccatiMethod::kAuto);

// Solves the discrete Lyapunov equation  A X A^T - X + Q = 0.
bool SolveDiscreteLyapunov(const Matrix& a, const Matrix& q, Matrix* x,
                           RiccatiMethod method = RiccatiMethod::kAuto);

// Controllability and observability Gramians, from the Lyapunov equations
//   A Wc + Wc A^T + B B^T = 0        (continuous)
//   A Wc A^T - Wc + B B^T = 0        (discrete)
// and the dual for Wo. Their eigenvalues are the "energy" of each state: a
// nearly singular Gramian is the quantitative version of "controllable, but
// only just", which the rank test alone hides.
Matrix ControllabilityGramian(const LtiSystem& system);
Matrix ObservabilityGramian(const LtiSystem& system);

// Hankel singular values: sqrt(eig(Wc Wo)), sorted descending. They are the
// input-output importance of each state and the basis of balanced truncation;
// a large ratio between the first and last one is what makes model reduction
// safe.
Vector HankelSingularValues(const LtiSystem& system);

// Balanced truncation: change coordinates so that Wc = Wo = diag(hsv), keep the
// `order` most important states, and return the reduced model. Throws when the
// system is not stable (the Gramians do not exist) or `order` is out of range.
LtiSystem BalancedTruncation(const LtiSystem& system, int order);

}  // namespace control

#endif  // CONTROL_LTI_H_
