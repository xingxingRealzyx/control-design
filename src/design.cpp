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

#include "control/design.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "control/lti.h"

namespace control {
namespace {

void AddWarning(DesignResult* result, const std::string& warning) {
    result->warnings.push_back(warning);
}

// Fills in the poles/damping/gain-magnitude part of a result and applies the
// magnitude warning. Every design path ends here, so a design is never returned
// without its numbers.
void Finalise(DesignResult* result, const LtiSystem& closed_loop,
              const DesignOptions& options) {
    const PoleCharacteristics characteristics = DescribePoles(closed_loop);
    result->closed_loop_poles = characteristics.poles;
    result->damping_ratios = characteristics.damping_ratios;
    result->gain_magnitude =
        result->gain.size() > 0 ? result->gain.cwiseAbs().maxCoeff() : 0.0;
    if (result->gain_magnitude > options.gain_warning_threshold) {
        AddWarning(result,
                   "the largest gain is " +
                       std::to_string(result->gain_magnitude) +
                       ", which will be limited by actuator saturation or "
                       "by the resolution of the DAC it is written to");
    }
    for (Eigen::Index i = 0; i < characteristics.poles.size(); ++i) {
        if (characteristics.damping_ratios(i) < 0.0) {
            AddWarning(result, "closed-loop pole " + std::to_string(i) +
                                   " has negative damping (it is unstable)");
            break;
        }
    }
    result->ok = true;
}

Vector BuildPolePolynomial(const std::vector<std::complex<double>>& poles) {
    // Coefficients of prod (s - p_i), highest power first.
    //
    // The product is carried out in complex arithmetic and only the real part
    // is returned. Taking `pole.real()` at each step instead - which is what an
    // earlier version did - turns (s - (-1 + 2j)) into (s + 1) and silently
    // destroys the imaginary part: poles -1 +/- 2j then produce the polynomial
    // (s+1)^2 instead of (s+1)^2 + 4, and the gain comes out as [1, 2] where
    // [5, 2] was asked for. The closed loop is still stable, which is why only
    // an exact-value check catches it.
    Eigen::VectorXcd coefficients = Eigen::VectorXcd::Ones(1);
    for (const std::complex<double>& pole : poles) {
        Eigen::VectorXcd next = Eigen::VectorXcd::Zero(coefficients.size() + 1);
        for (Eigen::Index i = 0; i < coefficients.size(); ++i) {
            next(i) += coefficients(i);             // s * term
            next(i + 1) -= pole * coefficients(i);  // -p * term
        }
        coefficients = next;
    }
    Vector result(coefficients.size());
    for (Eigen::Index i = 0; i < coefficients.size(); ++i) {
        if (std::abs(coefficients(i).imag()) >
            1e-9 * std::max(1.0, std::abs(coefficients(i)))) {
            throw std::invalid_argument(
                "PlacePoles: the requested poles do not form conjugate pairs, "
                "so no "
                "real gain places them. Complex poles must be supplied in "
                "conjugate "
                "pairs");
        }
        result(i) = coefficients(i).real();
    }
    return result;
}

std::string DescribeModes(const ComplexVector& poles) {
    std::ostringstream text;
    text.precision(4);
    for (Eigen::Index i = 0; i < poles.size(); ++i) {
        if (i > 0) {
            text << ", ";
        }
        text << std::fixed << poles(i).real();
        if (std::abs(poles(i).imag()) > 1e-12) {
            text << (poles(i).imag() >= 0 ? "+" : "-")
                 << std::abs(poles(i).imag()) << "j";
        }
    }
    return text.str();
}

}  // namespace

bool SolveContinuousRiccati(const Matrix& a, const Matrix& b, const Matrix& q,
                            const Matrix& r, Matrix* p, RiccatiMethod method,
                            const DesignOptions& options) {
    static_cast<void>(
        method);  // the Hamiltonian route is used for both methods
    if (p == nullptr || a.rows() != a.cols() || b.rows() != a.rows() ||
        q.rows() != a.rows() || q.cols() != a.cols() || r.rows() != b.cols() ||
        r.cols() != b.cols()) {
        return false;
    }
    if (!a.allFinite() || !b.allFinite() || !q.allFinite() || !r.allFinite()) {
        return false;
    }

    // Hamiltonian matrix. The stabilising solution comes from its stable
    // invariant subspace: with [U1; U2] spanning that subspace, P = U2 U1^-1.
    //
    //   H = [ A   -B R^-1 B' ]
    //       [ -Q   -A'       ]
    const Eigen::Index n = a.rows();
    const Matrix r_inverse =
        r.partialPivLu().solve(Matrix::Identity(r.rows(), r.cols()));
    Matrix hamiltonian = Matrix::Zero(2 * n, 2 * n);
    hamiltonian.topLeftCorner(n, n) = a;
    hamiltonian.topRightCorner(n, n) = -b * r_inverse * b.transpose();
    hamiltonian.bottomLeftCorner(n, n) = -q;
    hamiltonian.bottomRightCorner(n, n) = -a.transpose();

    // A Schur form would be the textbook route; the eigenvectors are used here
    // because Eigen's real Schur does not order the spectrum, and the stable
    // subspace has to be selected explicitly either way.
    Eigen::EigenSolver<Matrix> solver(hamiltonian);
    if (solver.info() != Eigen::Success) {
        return false;
    }
    const ComplexVector eigenvalues = solver.eigenvalues();
    const Eigen::MatrixXcd eigenvectors = solver.eigenvectors();

    // Select the n eigenvalues with negative real part, keeping conjugate pairs
    // together so the resulting basis stays real.
    std::vector<int> stable;
    std::vector<int> unstable;
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i).real() < 0.0) {
            stable.push_back(static_cast<int>(i));
        } else {
            unstable.push_back(static_cast<int>(i));
        }
    }
    if (static_cast<int>(stable.size()) != n) {
        // A Hamiltonian with an eigenvalue on the imaginary axis has no
        // stabilising solution; this is the case R not positive definite (or a
        // mode that cannot be stabilised) produces.
        return false;
    }

    Eigen::MatrixXcd basis(2 * n, n);
    for (int i = 0; i < n; ++i) {
        basis.col(i) = eigenvectors.col(stable[static_cast<std::size_t>(i)]);
    }
    const Eigen::MatrixXcd u1 = basis.topRows(n);
    const Eigen::MatrixXcd u2 = basis.bottomRows(n);
    const Eigen::MatrixXcd solution =
        u2 * u1.partialPivLu().solve(Eigen::MatrixXcd::Identity(n, n));
    if (!solution.allFinite()) {
        return false;
    }
    *p = solution.real();
    *p = 0.5 * (*p + p->transpose());

    // Verify rather than trust: the residual of the Riccati equation is the one
    // statement this function makes, so it is checked here instead of being
    // assumed. A design built on a P that does not solve the equation is a
    // design built on nothing.
    const Matrix residual = a.transpose() * (*p) + (*p) * a -
                            (*p) * b * r_inverse * b.transpose() * (*p) + q;
    const double scale =
        std::max(1.0, (*p).cwiseAbs().maxCoeff() * a.cwiseAbs().maxCoeff());
    if (residual.cwiseAbs().maxCoeff() > 1e-6 * scale) {
        return false;
    }
    static_cast<void>(options);
    return true;
}

bool SolveDiscreteRiccati(const Matrix& a, const Matrix& b, const Matrix& q,
                          const Matrix& r, Matrix* p, RiccatiMethod method,
                          const DesignOptions& options) {
    static_cast<void>(method);
    if (p == nullptr || a.rows() != a.cols() || b.rows() != a.rows() ||
        q.rows() != a.rows() || q.cols() != a.cols() || r.rows() != b.cols() ||
        r.cols() != b.cols()) {
        return false;
    }
    if (!a.allFinite() || !b.allFinite() || !q.allFinite() || !r.allFinite()) {
        return false;
    }

    // Fixed-point iteration of the Riccati recursion, which converges to the
    // stabilising solution from P = Q for a stabilizable/detectable pair.
    //
    // The textbook alternative is a symplectic pencil; the iteration is used
    // because it needs no ordered generalized Schur form (Eigen can compute the
    // decomposition but not order it), and because the DARE is small in every
    // design this library is for. Convergence is checked, and failure is
    // reported rather than returned as a half-solved matrix.
    Matrix current = q;
    bool converged = false;
    for (int iteration = 0; iteration < options.riccati_max_iterations;
         ++iteration) {
        const Matrix s = r + b.transpose() * current * b;
        const Matrix next =
            a.transpose() * current * a -
            a.transpose() * current * b *
                s.partialPivLu().solve(b.transpose() * current * a) +
            q;
        if (!next.allFinite()) {
            return false;
        }
        const double change = (next - current).cwiseAbs().maxCoeff();
        const double scale = std::max(1.0, next.cwiseAbs().maxCoeff());
        current = 0.5 * (next + next.transpose());
        if (change <= options.riccati_tolerance * scale) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        return false;
    }

    const Matrix s = r + b.transpose() * current * b;
    const Matrix residual =
        a.transpose() * current * a - current -
        a.transpose() * current * b *
            s.partialPivLu().solve(b.transpose() * current * a) +
        q;
    const double scale = std::max(1.0, current.cwiseAbs().maxCoeff());
    if (residual.cwiseAbs().maxCoeff() > 1e-6 * scale) {
        return false;
    }
    *p = current;
    return true;
}

DesignResult LinearQuadraticRegulator(const LtiSystem& system,
                                      const Matrix& state_weight,
                                      const Matrix& input_weight,
                                      RiccatiMethod method,
                                      const DesignOptions& options) {
    system.Validate("LinearQuadraticRegulator");
    if (system.is_discrete()) {
        throw std::invalid_argument(
            "LinearQuadraticRegulator: the model is discrete; use "
            "DiscreteLinearQuadraticRegulator (the two Riccati equations are "
            "not "
            "the same, and using the continuous one on sampled data gives a "
            "controller that is wrong by the sampling period)");
    }
    const int n = system.state_dimension();
    const int m = system.input_dimension();
    if (state_weight.rows() != n || state_weight.cols() != n) {
        throw std::invalid_argument(
            "LinearQuadraticRegulator: Q must be n x n");
    }
    if (input_weight.rows() != m || input_weight.cols() != m) {
        throw std::invalid_argument(
            "LinearQuadraticRegulator: R must be m x m");
    }
    // R must be positive definite for the problem to be well posed: R = 0 means
    // control is free, and the optimal gain is infinite.
    Eigen::LLT<Matrix> r_cholesky(input_weight);
    if (r_cholesky.info() != Eigen::Success) {
        throw std::invalid_argument(
            "LinearQuadraticRegulator: R must be symmetric positive definite");
    }
    Eigen::SelfAdjointEigenSolver<Matrix> q_solver(
        0.5 * (state_weight + state_weight.transpose()));
    if (q_solver.info() != Eigen::Success ||
        q_solver.eigenvalues().minCoeff() < -1e-9) {
        throw std::invalid_argument(
            "LinearQuadraticRegulator: Q must be symmetric positive "
            "semidefinite");
    }

    const ControllabilityResult controllability =
        AnalyseControllability(system, options.analysis);
    if (!controllability.stabilizable) {
        throw std::invalid_argument(
            "LinearQuadraticRegulator: the model is not stabilizable "
            "(unreachable "
            "unstable modes: " +
            DescribeModes(controllability.uncontrollable_modes) +
            "), so no state feedback can stabilise it");
    }

    DesignResult result;
    Matrix p;
    if (!SolveContinuousRiccati(system.a(), system.b(), state_weight,
                                input_weight, &p, method, options)) {
        throw std::runtime_error(
            "LinearQuadraticRegulator: the Riccati equation could not be "
            "solved "
            "(no stabilising solution - check that Q is positive semidefinite, "
            "R "
            "positive definite, and that (A, B) has no uncontrollable mode on "
            "the "
            "imaginary axis)");
    }
    result.riccati_solution = p;
    result.gain = input_weight.partialPivLu().solve(system.b().transpose() * p);
    result.notes =
        "LQR: u = -K x, K = R^-1 B' P with P the stabilising CARE solution";

    const LtiSystem closed_loop = LtiSystem::Continuous(
        system.a() - system.b() * result.gain, system.b(), system.c());
    Finalise(&result, closed_loop, options);
    return result;
}

DesignResult DiscreteLinearQuadraticRegulator(const LtiSystem& system,
                                              const Matrix& state_weight,
                                              const Matrix& input_weight,
                                              RiccatiMethod method,
                                              const DesignOptions& options) {
    system.Validate("DiscreteLinearQuadraticRegulator");
    if (!system.is_discrete()) {
        throw std::invalid_argument(
            "DiscreteLinearQuadraticRegulator: the model is continuous; "
            "discretize "
            "it first (the sample time is part of the design, not an "
            "afterthought)");
    }
    const int n = system.state_dimension();
    const int m = system.input_dimension();
    if (state_weight.rows() != n || state_weight.cols() != n) {
        throw std::invalid_argument(
            "DiscreteLinearQuadraticRegulator: Q must be n x n");
    }
    if (input_weight.rows() != m || input_weight.cols() != m) {
        throw std::invalid_argument(
            "DiscreteLinearQuadraticRegulator: R must be m x m");
    }
    Eigen::LLT<Matrix> r_cholesky(input_weight);
    if (r_cholesky.info() != Eigen::Success) {
        throw std::invalid_argument(
            "DiscreteLinearQuadraticRegulator: R must be symmetric positive "
            "definite");
    }

    const ControllabilityResult controllability =
        AnalyseControllability(system, options.analysis);
    if (!controllability.stabilizable) {
        throw std::invalid_argument(
            "DiscreteLinearQuadraticRegulator: the model is not stabilizable "
            "(unreachable unstable modes: " +
            DescribeModes(controllability.uncontrollable_modes) + ")");
    }

    DesignResult result;
    Matrix p;
    if (!SolveDiscreteRiccati(system.a(), system.b(), state_weight,
                              input_weight, &p, method, options)) {
        throw std::runtime_error(
            "DiscreteLinearQuadraticRegulator: the discrete Riccati equation "
            "did not "
            "converge to a stabilising solution");
    }
    result.riccati_solution = p;
    const Matrix s = input_weight + system.b().transpose() * p * system.b();
    result.gain =
        s.partialPivLu().solve(system.b().transpose() * p * system.a());
    result.notes =
        "DLQR: u = -K x, K = (R + B' P B)^-1 B' P A with P the stabilising "
        "DARE "
        "solution";

    const LtiSystem closed_loop =
        LtiSystem::Discrete(system.a() - system.b() * result.gain, system.b(),
                            system.c(), system.sample_time());
    Finalise(&result, closed_loop, options);
    return result;
}

DesignResult PlacePoles(const LtiSystem& system,
                        const std::vector<std::complex<double>>& desired_poles,
                        const DesignOptions& options) {
    system.Validate("PlacePoles");
    const int n = system.state_dimension();
    if (static_cast<int>(desired_poles.size()) != n) {
        throw std::invalid_argument(
            "PlacePoles: the number of desired poles must equal the number of "
            "states (" +
            std::to_string(desired_poles.size()) + " given, " +
            std::to_string(n) + " expected)");
    }
    if (system.input_dimension() != 1) {
        throw std::invalid_argument(
            "PlacePoles: only single-input models are supported. A multi-input "
            "model "
            "has a family of gains placing the same poles, and choosing within "
            "that "
            "family is what the LQR weights are for; use "
            "LinearQuadraticRegulator instead of an arbitrary member of the "
            "family");
    }

    const ControllabilityResult controllability =
        AnalyseControllability(system, options.analysis);
    if (!controllability.fully_controllable) {
        throw std::invalid_argument(
            "PlacePoles: the model is not fully controllable (unreachable "
            "modes: " +
            DescribeModes(controllability.uncontrollable_modes) +
            "); those modes keep their own dynamics whatever gain is chosen");
    }

    // Ackermann: K = e_n' * C^-1 * phi(A), with phi the desired characteristic
    // polynomial evaluated at A and C the controllability matrix.
    // phi(A) = sum_{j=0}^{n} c_j A^j, where c_j is the coefficient of s^j.
    // `coefficients` is highest-power-first, so the coefficient of s^j sits at
    // index (n - j) - and the power must ascend with j. An earlier version
    // walked i downwards while multiplying the leading coefficient by A^0,
    // which evaluates the polynomial backwards: it produced K = [-11, -13]
    // instead of [4, 2] for poles -2, -3, with the closed loop unstable. The
    // achieved-pole comparison below is what caught it.
    const Vector coefficients = BuildPolePolynomial(desired_poles);
    Matrix phi = Matrix::Zero(n, n);
    Matrix power = Matrix::Identity(n, n);  // A^0
    for (int j = 0; j <= n; ++j) {
        phi += coefficients(n - j) * power;
        power = power * system.a();
    }

    const Matrix controllability_matrix = ControllabilityMatrix(system);
    Matrix selector = Matrix::Zero(1, n);
    selector(0, n - 1) = 1.0;
    const Matrix k =
        selector * controllability_matrix.partialPivLu().solve(phi);
    if (!k.allFinite()) {
        throw std::runtime_error(
            "PlacePoles: the Ackermann formula produced a non-finite gain (the "
            "controllability matrix is numerically singular)");
    }

    DesignResult result;
    result.gain = k;
    result.notes =
        "Ackermann pole placement on a single-input model; the achieved poles "
        "are "
        "reported below and should match the request";
    const LtiSystem closed_loop = LtiSystem::Continuous(
        system.a() - system.b() * k, system.b(), system.c());
    Finalise(&result, closed_loop, options);

    // The achieved poles are the honest statement: compare them with the
    // request and say so when they differ.
    Vector requested_real(static_cast<Eigen::Index>(desired_poles.size()));
    for (std::size_t i = 0; i < desired_poles.size(); ++i) {
        requested_real(static_cast<Eigen::Index>(i)) = desired_poles[i].real();
    }
    double worst = 0.0;
    for (Eigen::Index i = 0; i < result.closed_loop_poles.size(); ++i) {
        worst = std::max(worst, std::abs(result.closed_loop_poles(i).real() -
                                         requested_real(i)));
    }
    if (worst > 1e-6 * std::max(1.0, requested_real.cwiseAbs().maxCoeff())) {
        AddWarning(&result,
                   "the achieved poles differ from the request by up to " +
                       std::to_string(worst) +
                       " (the model is nearly uncontrollable, so the "
                       "placement is ill-conditioned)");
    }
    return result;
}

DesignResult PlaceObserverPoles(
    const LtiSystem& system,
    const std::vector<std::complex<double>>& desired_poles,
    const DesignOptions& options) {
    system.Validate("PlaceObserverPoles");
    if (system.output_dimension() != 1) {
        throw std::invalid_argument(
            "PlaceObserverPoles: only single-output models are supported, by "
            "duality with PlacePoles. A multi-output model has a family of "
            "observer gains; use the Kalman gain "
            "(SolveDiscreteSteadyStateGain) to "
            "choose within it by noise rather than arbitrarily");
    }
    const ObservabilityResult observability =
        AnalyseObservability(system, options.analysis);
    if (!observability.fully_observable) {
        throw std::invalid_argument(
            "PlaceObserverPoles: the model is not fully observable (invisible "
            "modes: " +
            DescribeModes(observability.unobservable_modes) +
            "); an observer cannot estimate what no sensor sees");
    }

    // Duality: placing the poles of (A - L C) is placing the poles of
    // (A' - C' L') on the transposed problem.
    const LtiSystem dual =
        system.is_continuous()
            ? LtiSystem::Continuous(system.a().transpose(),
                                    system.c().transpose(),
                                    system.b().transpose())
            : LtiSystem::Discrete(system.a().transpose(),
                                  system.c().transpose(),
                                  system.b().transpose(), system.sample_time());
    const DesignResult dual_design = PlacePoles(dual, desired_poles, options);

    DesignResult result;
    result.gain = dual_design.gain.transpose();  // L = K'
    result.notes =
        "observer gain by duality: the poles of (A - L C) are the requested "
        "ones";
    result.warnings = dual_design.warnings;
    const LtiSystem error_dynamics =
        system.is_continuous()
            ? LtiSystem::Continuous(system.a() - result.gain * system.c(),
                                    system.b(), system.c())
            : LtiSystem::Discrete(system.a() - result.gain * system.c(),
                                  system.b(), system.c(), system.sample_time());
    Finalise(&result, error_dynamics, options);
    return result;
}

bool SolveDiscreteSteadyStateGain(const Matrix& a, const Matrix& c,
                                  const Matrix& process_noise,
                                  const Matrix& measurement_noise, Matrix* gain,
                                  Matrix* prior_covariance,
                                  const DesignOptions& options) {
    if (gain == nullptr || a.rows() != a.cols() || c.cols() != a.rows() ||
        process_noise.rows() != a.rows() || process_noise.cols() != a.rows() ||
        measurement_noise.rows() != c.rows() ||
        measurement_noise.cols() != c.rows()) {
        return false;
    }
    // Iterate the Kalman covariance recursion to its fixed point. The gain
    // stops moving long before the covariance does, which is why the
    // convergence test is on the gain.
    const Eigen::Index n = a.rows();
    Matrix prior = Matrix::Identity(n, n);
    Matrix previous_gain = Matrix::Zero(n, c.rows());
    for (int iteration = 0; iteration < options.riccati_max_iterations;
         ++iteration) {
        const Matrix s = c * prior * c.transpose() + measurement_noise;
        const Matrix current_gain =
            prior * c.transpose() *
            s.partialPivLu().solve(Matrix::Identity(s.rows(), s.cols()));
        const Matrix posterior =
            (Matrix::Identity(n, n) - current_gain * c) * prior;
        prior = a * posterior * a.transpose() + process_noise;
        prior = 0.5 * (prior + prior.transpose());
        if (!prior.allFinite()) {
            return false;
        }
        const double change =
            (current_gain - previous_gain).cwiseAbs().maxCoeff();
        if (iteration > 0 &&
            change <=
                1e-12 * std::max(1.0, current_gain.cwiseAbs().maxCoeff())) {
            *gain = current_gain;
            if (prior_covariance != nullptr) {
                *prior_covariance = prior;
            }
            return true;
        }
        previous_gain = current_gain;
    }
    return false;
}

AugmentedDesign AddIntegralAction(
    const LtiSystem& system, int output_to_integrate,
    const std::vector<std::complex<double>>& poles,
    const DesignOptions& options) {
    system.Validate("AddIntegralAction");
    const int n = system.state_dimension();
    const int m = system.input_dimension();
    if (output_to_integrate < 0 ||
        output_to_integrate >= system.output_dimension()) {
        throw std::invalid_argument(
            "AddIntegralAction: output_to_integrate is out of range");
    }
    if (static_cast<int>(poles.size()) != n + 1) {
        throw std::invalid_argument(
            "AddIntegralAction: one extra pole is needed for the integrator "
            "state (" +
            std::to_string(n + 1) + " expected, " +
            std::to_string(poles.size()) + " given)");
    }

    // Augmented model with z' = r - y (continuous) or z+ = z + r - y
    // (discrete).
    Matrix augmented_a = Matrix::Zero(n + 1, n + 1);
    augmented_a.topLeftCorner(n, n) = system.a();
    // The integrator state is z = integral of (r - y) dt, so a discrete design
    // updates z+ = z + Ts (r - y): z carries the unit of output*second, and the
    // generated controller integrates with the same convention (kSampleTime
    // multiplies the error there). Getting this factor wrong scales the
    // integral gain by the sample period - a 100x error at 10 ms that looks
    // like a badly tuned weight.
    const Matrix integrator_row =
        system.is_discrete() ? Matrix(-system.sample_time() *
                                      system.c().row(output_to_integrate))
                             : Matrix(-system.c().row(output_to_integrate));
    augmented_a.bottomLeftCorner(1, n) = integrator_row;
    if (system.is_discrete()) {
        augmented_a(n, n) = 1.0;
    }
    Matrix augmented_b = Matrix::Zero(n + 1, m);
    augmented_b.topRows(n) = system.b();
    // The augmented system has n+1 states, so C gains a column: the output is
    // still C x, with no direct path from the integrator state.
    Matrix augmented_c = Matrix::Zero(system.output_dimension(), n + 1);
    augmented_c.leftCols(n) = system.c();
    const LtiSystem augmented =
        system.is_continuous()
            ? LtiSystem::Continuous(augmented_a, augmented_b, augmented_c)
            : LtiSystem::Discrete(augmented_a, augmented_b, augmented_c,
                                  system.sample_time());

    AugmentedDesign design;
    design.augmented = augmented;
    design.integral_channels = 1;
    design.design = PlacePoles(augmented, poles, options);
    design.design.notes =
        "integral action on output " + std::to_string(output_to_integrate) +
        ": the gain acts on [x; z], with z the integral of the tracking error";
    return design;
}

AugmentedDesign AddIntegralActionLqr(const LtiSystem& system,
                                     int output_to_integrate,
                                     const Matrix& state_weight,
                                     const Matrix& integral_weight,
                                     const Matrix& input_weight,
                                     const DesignOptions& options) {
    system.Validate("AddIntegralActionLqr");
    const int n = system.state_dimension();
    const int m = system.input_dimension();
    if (output_to_integrate < 0 ||
        output_to_integrate >= system.output_dimension()) {
        throw std::invalid_argument(
            "AddIntegralActionLqr: output_to_integrate is out of range");
    }
    if (state_weight.rows() != n || state_weight.cols() != n) {
        throw std::invalid_argument("AddIntegralActionLqr: Q must be n x n");
    }
    if (integral_weight.rows() != 1 || integral_weight.cols() != 1) {
        throw std::invalid_argument(
            "AddIntegralActionLqr: the integral weight must be 1 x 1");
    }

    Matrix augmented_a = Matrix::Zero(n + 1, n + 1);
    augmented_a.topLeftCorner(n, n) = system.a();
    const Matrix integrator_row =
        system.is_discrete() ? Matrix(-system.sample_time() *
                                      system.c().row(output_to_integrate))
                             : Matrix(-system.c().row(output_to_integrate));
    augmented_a.bottomLeftCorner(1, n) = integrator_row;
    if (system.is_discrete()) {
        augmented_a(n, n) = 1.0;
    }
    Matrix augmented_b = Matrix::Zero(n + 1, m);
    augmented_b.topRows(n) = system.b();
    Matrix augmented_c = Matrix::Zero(system.output_dimension(), n + 1);
    augmented_c.leftCols(n) = system.c();
    const LtiSystem augmented =
        system.is_continuous()
            ? LtiSystem::Continuous(augmented_a, augmented_b, augmented_c)
            : LtiSystem::Discrete(augmented_a, augmented_b, augmented_c,
                                  system.sample_time());

    // Augmented weights: Q on the states, the integral weight on z. A small
    // integral weight gives slow offset removal, a large one gives overshoot.
    Matrix augmented_q = Matrix::Zero(n + 1, n + 1);
    augmented_q.topLeftCorner(n, n) = state_weight;
    augmented_q(n, n) = integral_weight(0, 0);

    AugmentedDesign design;
    design.augmented = augmented;
    design.integral_channels = 1;
    design.design =
        system.is_continuous()
            ? LinearQuadraticRegulator(augmented, augmented_q, input_weight,
                                       RiccatiMethod::kAuto, options)
            : DiscreteLinearQuadraticRegulator(augmented, augmented_q,
                                               input_weight,
                                               RiccatiMethod::kAuto, options);
    design.design.notes = "LQR with integral action on output " +
                          std::to_string(output_to_integrate) +
                          "; the integral weight sets how fast the offset is "
                          "removed and how much "
                          "overshoot is accepted";
    return design;
}

bool ReferenceFeedforward(const LtiSystem& system, double reference,
                          Matrix* gain, Vector* state_at_reference) {
    system.Validate("ReferenceFeedforward");
    if (gain == nullptr) {
        return false;
    }
    const int n = system.state_dimension();
    // Solve for the equilibrium that produces the requested output:
    //   continuous:  [A  B] [x; u] = [0; reference]
    //   discrete:    [A-I B] [x; u] = [0; reference]
    // the second row being C x = reference.
    Matrix left = Matrix::Zero(n + 1, n + system.input_dimension());
    left.topLeftCorner(n, n) = system.is_continuous()
                                   ? system.a()
                                   : system.a() - Matrix::Identity(n, n);
    left.topRightCorner(n, system.input_dimension()) = system.b();
    left.bottomLeftCorner(1, n) = system.c().row(0);

    Vector right = Vector::Zero(n + 1);
    right(n) = reference;

    // Least squares rather than a square solve: the system is (n+1) x (n+m) and
    // overdetermined for a single input, which is the normal case.
    const Vector solution = left.colPivHouseholderQr().solve(right);
    if (!solution.allFinite()) {
        return false;
    }
    const Vector residual = left * solution - right;
    if (residual.cwiseAbs().maxCoeff() >
        1e-9 * std::max(1.0, std::abs(reference))) {
        // No equilibrium produces this output: an integrator is the answer, and
        // saying so is better than returning a feedforward that misses.
        return false;
    }
    *gain = solution.tail(system.input_dimension()).transpose();
    if (state_at_reference != nullptr) {
        *state_at_reference = solution.head(n);
    }
    return true;
}

}  // namespace control
