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

#ifndef CONTROL_TYPES_H_
#define CONTROL_TYPES_H_

// Shared vocabulary: the state-space container that every design function
// takes, and the enumerations that select a numerical method.
//
// The types here are dynamic-size Eigen objects on purpose. This library runs
// on the host, at design time, where matrices are a few dozen rows at most and
// clarity beats static sizing. The *output* of a design (a gain vector, a
// compensator) is what gets deployed, not these types - see control/codegen.h.

#include <Eigen/Core>
#include <complex>
#include <iosfwd>
#include <string>
#include <vector>

namespace control {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using ComplexVector = Eigen::VectorXcd;

// State-space model, continuous or discrete:
//
//   continuous:  x' = A x + B u,   y = C x + D u
//   discrete:    x+ = A x + B u,   y = C x + D u
//
// A is n x n, B is n x m, C is p x n, D is p x m.
class LtiSystem {
 public:
    LtiSystem() = default;
    LtiSystem(Matrix a, Matrix b, Matrix c, Matrix d, double sample_time = 0.0);

    // Continuous-time model with D = 0 (strictly proper), the usual case.
    static LtiSystem Continuous(Matrix a, Matrix b, Matrix c);
    // Discrete-time model sampled at `sample_time` seconds, D = 0.
    static LtiSystem Discrete(Matrix a, Matrix b, Matrix c, double sample_time);

    const Matrix& a() const { return a_; }
    const Matrix& b() const { return b_; }
    const Matrix& c() const { return c_; }
    const Matrix& d() const { return d_; }

    int state_dimension() const { return static_cast<int>(a_.rows()); }
    int input_dimension() const { return static_cast<int>(b_.cols()); }
    int output_dimension() const { return static_cast<int>(c_.rows()); }

    // A sample time of zero marks a continuous-time model.
    bool is_continuous() const { return sample_time_ <= 0.0; }
    bool is_discrete() const { return !is_continuous(); }
    double sample_time() const { return sample_time_; }

    // Rejects malformed systems (wrong shapes, non-finite entries) here, rather
    // than letting them surface later inside a solver with an obscure message.
    bool IsValid() const;
    // What IsValid() would reject, one line each; empty when valid.
    std::string DescribeProblem() const;
    // Throws std::invalid_argument carrying DescribeProblem() when invalid, so
    // a caller mistake names itself at the call site.
    void Validate(const char* context) const;

 private:
    Matrix a_;
    Matrix b_;
    Matrix c_;
    Matrix d_;
    double sample_time_ = 0.0;  // 0 marks continuous time, see is_continuous()
};

std::ostream& operator<<(std::ostream& out, const LtiSystem& system);

// Returned by the analysis functions instead of throwing, because "this system
// is not controllable" is a result, not an error.
struct RankResult {
    int rank = 0;
    int dimension = 0;
    double smallest_singular_value = 0.0;
    double largest_singular_value = 0.0;

    // True when the rank is full. The tolerance is relative to the largest
    // singular value, so it does not depend on the scaling of the model.
    bool IsFullRank() const { return rank == dimension; }
    // Reasonable numerical measure of "how far from singular": near zero means
    // the property holds only marginally, which matters more in practice than
    // the boolean.
    double condition_number() const;
};

// How a continuous system is turned into a discrete one. The choice changes the
// answer, not merely its accuracy, so it is always explicit in the API.
// Note on first-order hold: it is deliberately absent. An exact FOH model needs
// the *previous* input as part of the state, so it cannot be expressed as a
// plain single-rate state-space model - a version that pretends otherwise is
// simply wrong. Zero-order hold at a fast enough sample time covers the same
// ground with one parameter fewer, and the sample-time rule in discretize.h is
// what makes "fast enough" a number instead of an opinion.
enum class DiscretizationMethod {
    kZeroOrderHold,  // exact for a piecewise-constant input (the default)
    kTustin,         // bilinear: stable poles stay inside the unit circle
    kForwardEuler,   // cheap; can destabilize a stable continuous system
    kBackwardEuler,  // cheap; always maps a stable pole inside the unit circle
    kMatchedPoleZero,  // pole/zero matching plus a DC or Nyquist gain match
};

const char* ToString(DiscretizationMethod method);

// Which algebraic Riccati equation solver to use.
enum class RiccatiMethod {
    kAuto,       // choose from the problem shape (the default)
    kSchur,      // structured Schur on the Hamiltonian / symplectic pencil
    kKleinman,   // Newton iteration from a stabilizing initial gain
    kIteration,  // fixed-point iteration (slow, hard to break)
};

const char* ToString(RiccatiMethod method);

// Pole/zero locations, with the multiplicities kept in the vector.
struct PoleZeroResult {
    ComplexVector poles;
    ComplexVector zeros;  // transmission zeros; empty when there are none
    bool has_zeros = false;
};

}  // namespace control

#endif  // CONTROL_TYPES_H_
