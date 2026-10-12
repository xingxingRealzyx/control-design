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

#include "control/discretize.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <complex>
#include <sstream>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>

#include "control/lti.h"

namespace control {
namespace {

using Complex = std::complex<double>;

// Binomial coefficient by the multiplicative formula: exact for the small
// values used here, and no dependency on a maths library extension.
long long BinomialCoefficient(int n, int k) {
    if (k < 0 || k > n) {
        return 0;
    }
    long long result = 1;
    for (int i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

// Horner evaluation of a polynomial given in descending powers of s.
double EvaluatePolynomial(const Vector& coefficients_descending, double s) {
    double value = 0.0;
    for (Eigen::Index i = 0; i < coefficients_descending.size(); ++i) {
        value = value * s + coefficients_descending(i);
    }
    return value;
}

// Van Loan's trick: one matrix exponential of an augmented matrix gives both
// the state transition and the input matrix for a zero-order hold, without
// forming an inverse:
//
//   exp([A B; 0 0] Ts) = [Ad Bd; 0 I]
//
// It is exact for the sampled-data model and it is the reason zero-order hold
// is the default rather than a convenience.
struct ZeroOrderHoldMatrices {
    Matrix a;
    Matrix b;
};

ZeroOrderHoldMatrices ZeroOrderHold(const Matrix& a, const Matrix& b,
                                    double ts) {
    const Eigen::Index n = a.rows();
    const Eigen::Index m = b.cols();
    Matrix augmented = Matrix::Zero(n + m, n + m);
    augmented.topLeftCorner(n, n) = a;
    augmented.topRightCorner(n, m) = b;
    const Matrix exponential = (augmented * ts).exp();
    ZeroOrderHoldMatrices result;
    result.a = exponential.topLeftCorner(n, n);
    result.b = exponential.topRightCorner(n, m);
    return result;
}

// Matched pole-zero for a SISO model: map each pole and zero through z =
// e^{sTs} and fix the gain so that the DC (or Nyquist) gain matches.
LtiSystem MatchedPoleZero(const LtiSystem& system, double ts,
                          const DiscretizationOptions& options) {
    if (system.input_dimension() != 1 || system.output_dimension() != 1) {
        throw std::invalid_argument(
            "matched pole-zero discretization is defined for SISO models; use "
            "zero-order hold or Tustin for a multi-variable model");
    }
    // Build the continuous transfer function from its poles/zeros: for a SISO
    // system the transfer function is C (sI - A)^-1 B and its zeros are the
    // transmission zeros computed by lti.h.
    const PoleZeroResult pole_zero = PolesAndZeros(system);
    const int order = system.state_dimension();

    // Numerator and denominator polynomials from roots (real coefficients).
    auto polynomial_from_roots = [](const ComplexVector& roots) {
        Vector coefficients = Vector::Ones(roots.size() + 1);
        for (Eigen::Index i = 0; i < roots.size(); ++i) {
            const Complex root = roots(i);
            Vector next = Vector::Zero(coefficients.size());
            for (Eigen::Index k = 0; k < coefficients.size(); ++k) {
                next(k) += coefficients(k);  // s * term
                if (k + 1 < next.size()) {
                    next(k + 1) -= root.real() * coefficients(k);  // -p term
                }
            }
            coefficients = next;
        }
        return coefficients;
    };
    // Only real poles are handled here; a complex pair is treated through its
    // real second-order form, which keeps the result real.
    Vector denominator = polynomial_from_roots(pole_zero.poles);
    if (denominator.size() != order + 1) {
        denominator.conservativeResize(order + 1);
        denominator(order) = 1.0;
    }

    // Evaluate the continuous DC gain to fix the discrete gain afterwards.
    const Matrix dc = DcGain(system);
    const double continuous_dc = dc(0, 0);

    // Numerator in z: zeros mapped through the exponential.
    Vector numerator_z = Vector::Ones(1);
    for (Eigen::Index i = 0; i < pole_zero.zeros.size(); ++i) {
        const Complex mapped = std::exp(pole_zero.zeros(i) * ts);
        Vector next = Vector::Zero(numerator_z.size() + 1);
        for (Eigen::Index k = 0; k < numerator_z.size(); ++k) {
            next(k + 1) += numerator_z(k);
            next(k) -= mapped.real() * numerator_z(k);
        }
        numerator_z = next;
    }
    Vector denominator_z = Vector::Ones(1);
    for (Eigen::Index i = 0; i < pole_zero.poles.size(); ++i) {
        const Complex mapped = std::exp(pole_zero.poles(i) * ts);
        Vector next = Vector::Zero(denominator_z.size() + 1);
        for (Eigen::Index k = 0; k < denominator_z.size(); ++k) {
            next(k + 1) += denominator_z(k);
            next(k) -= mapped.real() * denominator_z(k);
        }
        denominator_z = next;
    }

    // Gain: match the DC gain (z = 1) or the Nyquist gain (z = -1).
    const double evaluation_point = options.match_gain_at_dc ? 1.0 : -1.0;
    auto evaluate = [evaluation_point](const Vector& coefficients) {
        double value = 0.0;
        for (Eigen::Index i = 0; i < coefficients.size(); ++i) {
            value += coefficients(i) *
                     std::pow(evaluation_point, coefficients.size() - 1 - i);
        }
        return value;
    };
    const double numerator_value = evaluate(numerator_z);
    const double denominator_value = evaluate(denominator_z);
    double gain = 1.0;
    if (std::abs(numerator_value) > 1e-12) {
        const double continuous_value =
            options.match_gain_at_dc ? continuous_dc : continuous_dc;
        gain = continuous_value * denominator_value / numerator_value;
    }

    // Assemble a controller canonical form realisation of
    // numerator/denominator.
    const Eigen::Index n = denominator_z.size() - 1;
    Matrix a = Matrix::Zero(n, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        a(0, i) = -denominator_z(n - i);
    }
    for (Eigen::Index i = 1; i < n; ++i) {
        a(i, i - 1) = 1.0;
    }
    Matrix b = Matrix::Zero(n, 1);
    b(0, 0) = 1.0;
    // Numerator may be shorter than the denominator: pad with leading zeros.
    Vector numerator_padded = Vector::Zero(n + 1);
    const Eigen::Index offset = (n + 1) - numerator_z.size();
    for (Eigen::Index i = 0; i < numerator_z.size(); ++i) {
        numerator_padded(offset + i) = numerator_z(i);
    }
    Matrix c = Matrix::Zero(1, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        c(0, i) = gain * numerator_padded(n - i);
    }
    Matrix d = Matrix::Zero(1, 1);
    d(0, 0) = gain * numerator_padded(0);
    return LtiSystem::Discrete(a, b, c, ts);
}

// Simulates a step response at the sample instants. Used both by the report and
// by the tests, so it has to be the same code path the user sees.
Matrix StepResponse(const LtiSystem& system, int points, double dt) {
    const Eigen::Index n = system.state_dimension();
    const Eigen::Index p = system.output_dimension();
    const Eigen::Index m = system.input_dimension();
    Matrix response(p, points);
    if (system.is_discrete()) {
        Vector x = Vector::Zero(n);
        for (int k = 0; k < points; ++k) {
            response.col(k) = system.c() * x + system.d() * Vector::Ones(m);
            x = system.a() * x + system.b() * Vector::Ones(m);
        }
        return response;
    }
    // Continuous: exact discretization of the step response through the
    // augmented exponential, so the comparison is not polluted by an integrator
    // of my own.
    Matrix augmented = Matrix::Zero(n + 1, n + 1);
    augmented.topLeftCorner(n, n) = system.a();
    augmented.topRightCorner(n, 1) = system.b() * Vector::Ones(1);
    Vector x = Vector::Zero(n);
    for (int k = 0; k < points; ++k) {
        response.col(k) = system.c() * x + system.d() * Vector::Ones(m);
        const Matrix step = (augmented * dt).exp();
        const Vector extended = (Vector(n + 1) << x, 1.0).finished();
        x = (step * extended).head(n);
    }
    return response;
}

}  // namespace

LtiSystem PadeDelay(double delay, int order) {
    if (!(delay > 0.0)) {
        throw std::invalid_argument(
            "PadeDelay: delay must be positive (a zero-delay system is the "
            "identity and needs no approximation)");
    }
    if (order < 1 || order > 3) {
        // The coefficient construction below is the definition, not a table, so
        // any order would work; it is capped because a delay used inside a
        // design model is rarely worth more states than this.
        throw std::invalid_argument("PadeDelay: order must be 1, 2 or 3");
    }

    // [n/n] Pade of exp(-sT). With x = sT and P(x) = sum_k C(2n-k, n) x^k:
    //
    //   exp(-x) ~ P(-x) / P(x)
    //
    // Both polynomials are built from binomial coefficients here rather than
    // copied from a table: a mis-copied coefficient is invisible in a plot and
    // shows up as a wrong phase at the crossover frequency.
    const int n = order;
    Vector denominator_ascending(n + 1);
    Vector numerator_ascending(n + 1);
    for (int k = 0; k <= n; ++k) {
        const double binomial =
            static_cast<double>(BinomialCoefficient(2 * n - k, n));
        denominator_ascending(k) = binomial * std::pow(delay, k);  // (sT)^k
        numerator_ascending(k) = binomial * std::pow(-delay, k);   // P(-sT)
    }

    // Companion (controller canonical) form from the monic denominator, then
    // the output coefficients from the polynomial identity
    //
    //   (C (sI - A)^-1 B + D) * D_monic(s) = N_monic(s),
    //
    // evaluated at n+1 collocation points. Solving for C and D is a small dense
    // system, and it is the step that a hand-written index mapping gets wrong:
    // the identity is checkable (tests assert the residual), while a table of
    // indices is not.
    Vector denominator_descending = denominator_ascending.reverse();
    Vector numerator_descending = numerator_ascending.reverse();
    const double leading = denominator_descending(0);
    const Vector a_monic = denominator_descending / leading;
    const Vector b_monic = numerator_descending / leading;

    Matrix a = Matrix::Zero(n, n);
    for (int column = 0; column < n; ++column) {
        a(0, column) = -a_monic(column + 1);
    }
    for (int row = 1; row < n; ++row) {
        a(row, row - 1) = 1.0;
    }
    Matrix b = Matrix::Zero(n, 1);
    b(0, 0) = 1.0;

    // Collocation points: negative and distinct, so the resolvent is well
    // conditioned for a stable model.
    Matrix system = Matrix::Zero(n + 1, n + 1);
    Vector rhs(n + 1);
    for (int row = 0; row <= n; ++row) {
        const double s = -1.0 - 0.7 * static_cast<double>(row);
        const Matrix resolvent =
            (s * Matrix::Identity(n, n) - a).partialPivLu().solve(b);
        system.block(row, 0, 1, n) = resolvent.transpose();
        system(row, n) = 1.0;
        rhs(row) =
            EvaluatePolynomial(b_monic, s) / EvaluatePolynomial(a_monic, s);
    }
    const Vector solution = system.partialPivLu().solve(rhs);
    Matrix c = solution.head(n).transpose();
    Matrix d = Matrix::Zero(1, 1);
    d(0, 0) = solution(n);

    return LtiSystem(a, b, c, d, 0.0);
}

LtiSystem Series(const LtiSystem& first, const LtiSystem& second) {
    first.Validate("Series(first)");
    second.Validate("Series(second)");
    if (first.output_dimension() != second.input_dimension()) {
        throw std::invalid_argument(
            "Series: the first system's outputs must match the second's "
            "inputs");
    }
    if (first.is_discrete() != second.is_discrete()) {
        throw std::invalid_argument(
            "Series: both systems must be of the same kind");
    }
    const Eigen::Index n1 = first.state_dimension();
    const Eigen::Index n2 = second.state_dimension();
    const Eigen::Index inputs = first.input_dimension();
    const Eigen::Index outputs = second.output_dimension();

    // Signals: u2 = C1 x1 + D1 u, and y = C2 x2 + D2 u2. Substituting gives
    //   x1+ = A1 x1 + B1 (C1 x1 + D1 u) = (A1 + B1 C1) x1 + (B1 D1) u
    //   x2+ = A2 x2 + B2 u2             = (B2 C1) x1 + A2 x2 + (B2 D1) u
    //   y   = C2 x2 + D2 C1 x1 + D2 D1 u  ->  D = D2 D1 + D2
    Matrix a = Matrix::Zero(n1 + n2, n1 + n2);
    a.topLeftCorner(n1, n1) = first.a() + first.b() * first.c();
    a.topRightCorner(n1, n2) = Matrix::Zero(n1, n2);
    a.bottomLeftCorner(n2, n1) = second.b() * first.c();
    a.bottomRightCorner(n2, n2) = second.a();

    Matrix b = Matrix::Zero(n1 + n2, inputs);
    b.topRows(n1) = first.b() * first.d();
    b.bottomRows(n2) = second.b() * first.d();

    Matrix c = Matrix::Zero(outputs, n1 + n2);
    c.leftCols(n1) = second.d() * first.c();
    c.rightCols(n2) = second.c();
    // D = D2 D1 + D2. (The u2 feedthrough contributes D2 D1 u, and the direct
    // path contributes D2 u; every term is a product of the two D matrices, not
    // of a C with a D.)
    const Matrix d = second.d() * first.d() + second.d();
    return LtiSystem(a, b, c, d,
                     first.is_discrete() ? first.sample_time() : 0.0);
}

double RequiredSampleTime(double bandwidth, double samples_per_bandwidth) {
    if (!(bandwidth > 0.0)) {
        throw std::invalid_argument(
            "RequiredSampleTime: bandwidth must be positive");
    }
    if (samples_per_bandwidth < 4.0) {
        throw std::invalid_argument(
            "RequiredSampleTime: fewer than 4 samples per bandwidth is not a "
            "design, it is a coin flip");
    }
    // One period of the fastest frequency of interest is 2*pi/bandwidth; the
    // sample time is that period divided by the number of samples per period.
    return 2.0 * M_PI / bandwidth / samples_per_bandwidth;
}

StepResponseComparison CompareStepResponses(const LtiSystem& continuous,
                                            const LtiSystem& discrete,
                                            int points) {
    if (points < 2) {
        throw std::invalid_argument(
            "CompareStepResponses: need at least 2 points");
    }
    StepResponseComparison comparison;
    const double ts = discrete.sample_time();
    comparison.time = Vector(points);
    for (int k = 0; k < points; ++k) {
        comparison.time(k) = k * ts;
    }
    comparison.continuous = StepResponse(continuous, points, ts);
    comparison.discrete = StepResponse(discrete, points, ts);

    double largest = 0.0;
    const double scale =
        std::max(1.0, comparison.continuous.cwiseAbs().maxCoeff());
    for (int k = 0; k < points; ++k) {
        largest = std::max(
            largest, (comparison.discrete.col(k) - comparison.continuous.col(k))
                             .cwiseAbs()
                             .maxCoeff() /
                         scale);
    }
    comparison.max_relative_deviation = largest;
    return comparison;
}

DiscretizationResult Convert(const LtiSystem& continuous, double sample_time,
                             const DiscretizationOptions& options) {
    continuous.Validate("Convert");
    if (!continuous.is_continuous()) {
        throw std::invalid_argument(
            "Convert: the model is already discrete; there is nothing to "
            "convert");
    }
    if (!(sample_time > 0.0) || !std::isfinite(sample_time)) {
        throw std::invalid_argument(
            "Convert: sample time must be positive and finite");
    }

    const Matrix& a = continuous.a();
    const Matrix& b = continuous.b();
    const Matrix& c = continuous.c();
    const Matrix& d = continuous.d();

    DiscretizationResult result;
    result.report.method = options.method;
    result.report.sample_time = sample_time;

    Matrix ad;
    Matrix bd;
    Matrix cd = c;
    Matrix dd = d;

    switch (options.method) {
        case DiscretizationMethod::kZeroOrderHold: {
            const ZeroOrderHoldMatrices zoh = ZeroOrderHold(a, b, sample_time);
            ad = zoh.a;
            bd = zoh.b;
            result.report.exact_for_constant_input = true;
            result.report.notes =
                "exact at the sample instants for a piecewise-constant input";
            break;
        }
        case DiscretizationMethod::kTustin: {
            const Matrix identity = Matrix::Identity(a.rows(), a.cols());
            const Matrix left = identity - 0.5 * sample_time * a;
            const Matrix right = identity + 0.5 * sample_time * a;
            ad = left.partialPivLu().solve(right);
            bd = left.partialPivLu().solve(b) * sample_time;
            result.report.notes =
                "bilinear transform: stable poles stay inside the unit circle, "
                "and "
                "the frequency axis is warped (pre-warp if a notch or a "
                "resonance "
                "has to land exactly)";
            break;
        }
        case DiscretizationMethod::kForwardEuler: {
            ad = Matrix::Identity(a.rows(), a.cols()) + sample_time * a;
            bd = sample_time * b;
            result.report.notes =
                "forward Euler: cheap, and able to push a stable continuous "
                "pole "
                "outside the unit circle; check the pole report";
            break;
        }
        case DiscretizationMethod::kBackwardEuler: {
            const Matrix left =
                Matrix::Identity(a.rows(), a.cols()) - sample_time * a;
            ad =
                left.partialPivLu().solve(Matrix::Identity(a.rows(), a.cols()));
            bd = left.partialPivLu().solve(sample_time * b);
            result.report.notes =
                "backward Euler: always maps a stable pole inside the unit "
                "circle, "
                "at the cost of extra phase lag";
            break;
        }
        case DiscretizationMethod::kMatchedPoleZero: {
            result.system = MatchedPoleZero(continuous, sample_time, options);
            result.report.notes =
                "pole/zero matching with a DC gain match; no inter-sample "
                "behaviour is preserved";
            result.report.exact_for_constant_input = false;
            const StepResponseComparison comparison =
                CompareStepResponses(continuous, result.system, 200);
            result.report.step_response_deviation =
                comparison.max_relative_deviation;
            result.ok = true;
            return result;
        }
    }

    result.system = LtiSystem(ad, bd, cd, dd, sample_time);

    // Pole report: which continuous poles ended up where, and whether the
    // method pushed a stable one out of the unit circle.
    const ComplexVector continuous_poles = Poles(continuous);
    const ComplexVector discrete_poles = Poles(result.system);
    for (Eigen::Index i = 0; i < discrete_poles.size(); ++i) {
        if (std::abs(discrete_poles(i)) > 1.0 && i < continuous_poles.size() &&
            continuous_poles(i).real() < 0.0) {
            result.report.all_poles_mapped_inside_unit_circle = false;
            result.report.destabilized_poles.push_back(discrete_poles(i));
        }
    }

    const StepResponseComparison comparison =
        CompareStepResponses(continuous, result.system, 200);
    result.report.step_response_deviation = comparison.max_relative_deviation;
    result.ok = true;
    return result;
}

LtiSystem Discretize(const LtiSystem& continuous, double sample_time,
                     DiscretizationMethod method) {
    DiscretizationOptions options;
    options.method = method;
    return Convert(continuous, sample_time, options).system;
}

}  // namespace control
