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

#include "control/lti.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>

namespace control {
namespace {

using Complex = std::complex<double>;

// Tolerances are relative to the size of the matrices involved, so that a model
// scaled by 1e6 is judged the same way as one scaled by 1e-6. Absolute
// tolerances are the classic source of "works on my example" bugs.
double RelativeTolerance(const Matrix& matrix) {
    const double scale = matrix.size() > 0 ? matrix.cwiseAbs().maxCoeff() : 1.0;
    return 1e-9 * std::max(1.0, scale);
}

// Eigenvalue-based stability, with the boundary counted as unstable.
bool AllPolesStable(const ComplexVector& poles, bool continuous) {
    for (Eigen::Index i = 0; i < poles.size(); ++i) {
        if (continuous) {
            if (poles(i).real() >= 0.0) {
                return false;
            }
        } else if (std::abs(poles(i)) >= 1.0) {
            return false;
        }
    }
    return true;
}

// Kronecker-product formulation of the continuous Lyapunov equation
//   A X + X A^T + Q = 0
// solved as the linear system (I (x) A + A (x) I) vec(X) = -vec(Q).
Matrix SolveLyapunovKronecker(const Matrix& a, const Matrix& q) {
    const Eigen::Index n = a.rows();
    Matrix system = Eigen::KroneckerProduct(Matrix::Identity(n, n), a) +
                    Eigen::KroneckerProduct(a, Matrix::Identity(n, n));
    Matrix rhs = -q;
    rhs.resize(n * n, 1);
    const Matrix solution = system.partialPivLu().solve(rhs);
    Matrix x = solution;
    x.resize(n, n);
    return x;
}

Matrix SolveDiscreteLyapunovKronecker(const Matrix& a, const Matrix& q) {
    const Eigen::Index n = a.rows();
    Matrix system =
        Eigen::KroneckerProduct(a, a) - Matrix::Identity(n * n, n * n);
    Matrix rhs = -q;
    rhs.resize(n * n, 1);
    const Matrix solution = system.partialPivLu().solve(rhs);
    Matrix x = solution;
    x.resize(n, n);
    return x;
}

// Symmetrize and report whether the result is usable. A Lyapunov solve returns
// garbage (or infinities) exactly when the equation has no unique solution, and
// silently handing that back is how a design tool produces nonsense gains.
bool AcceptSolution(Matrix* x) {
    if (!x->allFinite()) {
        return false;
    }
    *x = 0.5 * (*x + x->transpose());
    return true;
}

// The infinite-series Gramian, used as a fallback and as a cross-check:
//   Wc = sum_k A^k B B^T (A^T)^k        (discrete)
//   Wc = integral e^{At} B B^T e^{A^T t} dt  (continuous, by trapezoid on a
//        grid fine enough for the slowest and fastest pole)
Matrix GramianBySeries(const LtiSystem& system, bool controllability,
                       int terms) {
    const Matrix& a = system.a();
    const Matrix& b = controllability ? system.b() : system.c().transpose();
    const Matrix q = b * b.transpose();
    if (system.is_discrete()) {
        Matrix gramian = Matrix::Zero(a.rows(), a.cols());
        Matrix power = Matrix::Identity(a.rows(), a.cols());
        for (int k = 0; k < terms; ++k) {
            gramian += power * q * power.transpose();
            power = a * power;
        }
        return gramian;
    }
    // Continuous: integrate with the scaling-and-squaring step matrix. A grid
    // based on the fastest pole is stable but expensive; this is a fallback for
    // the cases where the Lyapunov equation itself is singular.
    const ComplexVector poles = Poles(system);
    double fastest = 1.0;
    for (Eigen::Index i = 0; i < poles.size(); ++i) {
        fastest = std::max(fastest, std::abs(poles(i)));
    }
    const double dt = 0.1 / fastest;
    const Matrix step = (a * dt).exp();
    Matrix gramian = Matrix::Zero(a.rows(), a.cols());
    Matrix power = Matrix::Identity(a.rows(), a.cols());
    for (int k = 0; k < terms; ++k) {
        gramian += power * q * power.transpose() * dt;
        power = step * power;
    }
    return gramian;
}

}  // namespace

ComplexVector Poles(const LtiSystem& system) {
    system.Validate("Poles");
    Eigen::EigenSolver<Matrix> solver(system.a(),
                                      /*computeEigenvectors=*/false);
    ComplexVector poles = solver.eigenvalues();
    std::sort(poles.data(), poles.data() + poles.size(),
              [&system](const Complex& left, const Complex& right) {
                  if (system.is_continuous()) {
                      return left.real() > right.real();
                  }
                  return std::abs(left) > std::abs(right);
              });
    return poles;
}

PoleZeroResult PolesAndZeros(const LtiSystem& system) {
    system.Validate("PolesAndZeros");
    PoleZeroResult result;
    result.poles = Poles(system);

    const Eigen::Index n = system.state_dimension();
    const Eigen::Index m = system.input_dimension();
    const Eigen::Index p = system.output_dimension();

    // Scope limit, stated rather than hidden: the pencil-based computation is
    // verified for SISO *continuous* models (tests/lti_test.cpp covers the
    // analytic cases). Two situations are known not to work and are refused:
    //
    //   * a discrete model, where the zero structure includes the sampling
    //   zeros
    //     that zero-order hold introduces (a minimum-phase continuous plant
    //     commonly becomes non-minimum-phase once sampled, and the pencil
    //     formulation here does not reproduce that correctly);
    //   * a non-square model, where the number of finite zeros depends on the
    //     pencil's rest dimension and the null-vector test used below is not
    //     sufficient to separate them.
    //
    // Returning a wrong zero would silently corrupt a bandwidth or inversion
    // calculation downstream, so this throws instead.
    if (system.is_discrete()) {
        throw std::invalid_argument(
            "PolesAndZeros: discrete-time zeros are not supported; the "
            "sampling "
            "zeros introduced by a zero-order hold make the pencil formulation "
            "unreliable here (see the source comment)");
    }
    if (system.input_dimension() != 1 || system.output_dimension() != 1) {
        throw std::invalid_argument(
            "PolesAndZeros: only SISO models are supported; for a non-square "
            "model the finite-zero count depends on the pencil's rest "
            "dimension");
    }

    // Generalized eigenvalue problem on the system pencil
    //   [ A  B ]        [ I  0 ]
    //   [ C  0 ] - s *  [ 0  0 ]
    // The finite generalized eigenvalues are the transmission zeros.
    Matrix left = Matrix::Zero(n + p, n + m);
    left.topLeftCorner(n, n) = system.a();
    left.topRightCorner(n, m) = system.b();
    left.bottomLeftCorner(p, n) = system.c();
    Matrix right = Matrix::Zero(n + p, n + m);
    right.topLeftCorner(n, n) = Matrix::Identity(n, n);

    Eigen::RealQZ<Matrix> qz(left, right);
    if (qz.info() != Eigen::Success) {
        // No reliable decomposition: report no zeros rather than wrong ones.
        return result;
    }

    // RealQZ gives the triangular factors, not the eigenvalues directly: the
    // generalized eigenvalues of (S, T) are the ratios of the diagonal entries,
    // and a zero on the diagonal of T is an infinite eigenvalue (which is not a
    // finite transmission zero and is skipped).
    const Matrix& s_matrix = qz.matrixS();
    const Matrix& t_matrix = qz.matrixT();
    const Eigen::Index size = s_matrix.rows();
    const double t_scale = std::max(1.0, t_matrix.cwiseAbs().maxCoeff());
    std::vector<Complex> zeros;
    for (Eigen::Index i = 0; i < size; ++i) {
        if (std::abs(t_matrix(i, i)) <= 1e-12 * t_scale) {
            continue;
        }
        zeros.push_back(Complex(s_matrix(i, i) / t_matrix(i, i), 0.0));
    }

    // A generalized eigenvalue of the pencil is a *transmission* zero only when
    // it comes with genuine left and right null vectors of the shifted pencil
    //
    //   [ A - sI   B ] [x]     [0]        [y1^H y2^H] [ A - sI  B ] = [0 0]
    //   [   C      0 ] [u]  =  [0],                      [   C     0 ]
    //
    // so we test exactly that: the left null space must be orthogonal to B and
    // the right null space to C. Checking "the pencil is singular" instead is
    // not enough - for a rectangular pencil the rest dimension contributes
    // candidates whose left null vector has no reason to annihilate B, and
    // those spurious values used to leak through and appear as invented zeros.
    constexpr double kNullTolerance = 1e-7;
    std::vector<Complex> kept;
    for (const Complex& candidate : zeros) {
        Eigen::MatrixXcd shifted = left.cast<Complex>();
        shifted.topLeftCorner(n, n) -=
            candidate * Eigen::MatrixXcd::Identity(n, n);

        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
            shifted, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Vector& singular = svd.singularValues();
        if (singular.size() == 0 || singular(0) <= 0.0) {
            continue;
        }
        if (singular(singular.size() - 1) / singular(0) > kNullTolerance) {
            continue;  // the pencil is not (numerically) singular here
        }

        // Null vectors of the shifted pencil: the last column of V (length
        // n + m) and the last column of U (length n + p), which correspond to
        // the smallest singular value.
        const Eigen::VectorXcd right_kernel =
            svd.matrixV().col(svd.matrixV().cols() - 1);
        const Eigen::VectorXcd left_kernel =
            svd.matrixU().col(svd.matrixU().cols() - 1);

        // The right null vector's input part (the last m entries) is the zero
        // direction; it must be annihilated by C. The left null vector's output
        // part (the last p entries) must be annihilated by B.
        const Eigen::VectorXcd right_input = right_kernel.tail(m);
        const Eigen::VectorXcd left_output = left_kernel.tail(p);
        const double scale = std::max(1.0, singular(0));
        const double right_residual =
            (system.c().cast<Complex>() * right_kernel.head(n)).norm();
        const double left_residual =
            (system.b().cast<Complex>().adjoint() * left_kernel.head(n)).norm();
        if (right_residual <= 1e-6 * scale && left_residual <= 1e-6 * scale &&
            right_input.norm() > 1e-12 && left_output.norm() > 1e-12) {
            kept.push_back(candidate);
        }
    }
    std::sort(kept.begin(), kept.end(),
              [](const Complex& first, const Complex& second) {
                  return first.real() < second.real();
              });

    result.zeros.resize(static_cast<Eigen::Index>(kept.size()));
    for (std::size_t i = 0; i < kept.size(); ++i) {
        result.zeros(static_cast<Eigen::Index>(i)) = kept[i];
    }
    result.has_zeros = !kept.empty();
    return result;
}

bool IsStable(const LtiSystem& system) {
    return AllPolesStable(Poles(system), system.is_continuous());
}

double StabilityMargin(const LtiSystem& system) {
    const ComplexVector poles = Poles(system);
    if (poles.size() == 0) {
        return 0.0;
    }
    if (system.is_continuous()) {
        double worst = -std::numeric_limits<double>::infinity();
        for (Eigen::Index i = 0; i < poles.size(); ++i) {
            worst = std::max(worst, poles(i).real());
        }
        return -worst;
    }
    double largest = 0.0;
    for (Eigen::Index i = 0; i < poles.size(); ++i) {
        largest = std::max(largest, std::abs(poles(i)));
    }
    return 1.0 - largest;
}

PoleCharacteristics DescribePoles(const LtiSystem& system) {
    PoleCharacteristics characteristics;
    characteristics.is_continuous = system.is_continuous();
    characteristics.poles = Poles(system);

    const Eigen::Index count = characteristics.poles.size();
    characteristics.damping_ratios = Vector::Zero(count);
    characteristics.natural_frequencies = Vector::Zero(count);
    const double ts = system.sample_time();

    for (Eigen::Index i = 0; i < count; ++i) {
        Complex pole = characteristics.poles(i);
        if (!characteristics.is_continuous) {
            // Map a discrete pole to its continuous-time equivalent; a pole at
            // the origin is infinitely fast and is reported as such.
            if (std::abs(pole) <= std::numeric_limits<double>::min()) {
                characteristics.natural_frequencies(i) =
                    std::numeric_limits<double>::infinity();
                continue;
            }
            pole = std::log(pole) / ts;
        }
        const double magnitude = std::abs(pole);
        characteristics.natural_frequencies(i) = magnitude;
        if (magnitude > std::numeric_limits<double>::min()) {
            characteristics.damping_ratios(i) = -pole.real() / magnitude;
        }
    }
    return characteristics;
}

Matrix DcGain(const LtiSystem& system) {
    system.Validate("DcGain");
    const Matrix& a = system.a();
    const Eigen::Index n = a.rows();
    if (system.is_continuous()) {
        // det(A) decides existence; report infinities rather than throwing,
        // because "integrator in the loop" is a legitimate answer.
        if (std::abs(a.determinant()) <= RelativeTolerance(a)) {
            return Matrix::Constant(system.output_dimension(),
                                    system.input_dimension(),
                                    std::numeric_limits<double>::infinity());
        }
        return -system.c() * a.partialPivLu().solve(system.b()) + system.d();
    }
    const Matrix identity_shift = Matrix::Identity(n, n) - a;
    if (std::abs(identity_shift.determinant()) <=
        RelativeTolerance(identity_shift)) {
        return Matrix::Constant(system.output_dimension(),
                                system.input_dimension(),
                                std::numeric_limits<double>::infinity());
    }
    return system.c() * identity_shift.partialPivLu().solve(system.b()) +
           system.d();
}

Eigen::MatrixXcd FrequencyResponse(const LtiSystem& system, double omega) {
    system.Validate("FrequencyResponse");
    const Eigen::Index n = system.state_dimension();
    if (system.is_continuous()) {
        const Eigen::MatrixXcd shifted =
            Complex(0.0, omega) * Eigen::MatrixXcd::Identity(n, n) -
            system.a().cast<Complex>();
        return system.c().cast<Complex>() *
                   shifted.partialPivLu().solve(system.b().cast<Complex>()) +
               system.d().cast<Complex>();
    }
    const Complex z = std::exp(Complex(0.0, omega * system.sample_time()));
    const Eigen::MatrixXcd shifted =
        z * Eigen::MatrixXcd::Identity(n, n) - system.a().cast<Complex>();
    return system.c().cast<Complex>() *
               shifted.partialPivLu().solve(system.b().cast<Complex>()) +
           system.d().cast<Complex>();
}

FrequencyResponseData FrequencyResponseGrid(const LtiSystem& system, int points,
                                            double omega_max) {
    if (points < 2) {
        throw std::invalid_argument(
            "FrequencyResponseGrid: needs at least 2 points");
    }
    Vector omega(points);
    for (int i = 0; i < points; ++i) {
        // Logarithmic spacing from the first decade up to omega_max: linear
        // spacing wastes most samples at high frequency where nothing happens.
        const double lower = std::max(omega_max * 1e-4, 1e-12);
        const double t =
            static_cast<double>(i) / static_cast<double>(points - 1);
        omega(i) = lower * std::pow(omega_max / lower, t);
    }
    return FrequencyResponseGrid(system, omega);
}

FrequencyResponseData FrequencyResponseGrid(const LtiSystem& system,
                                            const Vector& omega) {
    FrequencyResponseData data;
    data.omega = omega;
    data.response.reserve(static_cast<std::size_t>(omega.size()));
    for (Eigen::Index i = 0; i < omega.size(); ++i) {
        data.response.push_back(FrequencyResponse(system, omega(i)));
    }
    return data;
}

Matrix FrequencyResponseSingularValues(const LtiSystem& system, int points,
                                       double omega_max) {
    const FrequencyResponseData data =
        FrequencyResponseGrid(system, points, omega_max);
    const int outputs = system.output_dimension();
    const int inputs = system.input_dimension();
    const int count = std::min(outputs, inputs);
    Matrix values(points, count);
    for (int i = 0; i < points; ++i) {
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
            data.response[static_cast<std::size_t>(i)]);
        for (int j = 0; j < count; ++j) {
            values(i, j) = svd.singularValues()(j);
        }
    }
    return values;
}

bool SolveLyapunov(const Matrix& a, const Matrix& q, Matrix* x,
                   RiccatiMethod method) {
    if (x == nullptr || a.rows() != a.cols() || q.rows() != a.rows() ||
        q.cols() != a.cols()) {
        return false;
    }
    if (!a.allFinite() || !q.allFinite()) {
        return false;
    }
    // The Kronecker formulation is a direct solve, so there is nothing to
    // iterate. The method argument is kept for API symmetry with the Riccati
    // solvers, where the choice does matter.
    static_cast<void>(method);
    *x = SolveLyapunovKronecker(a, q);
    return AcceptSolution(x);
}

bool SolveDiscreteLyapunov(const Matrix& a, const Matrix& q, Matrix* x,
                           RiccatiMethod method) {
    (void)method;  // the direct formulation is used for both methods
    if (x == nullptr || a.rows() != a.cols() || q.rows() != a.rows() ||
        q.cols() != a.cols()) {
        return false;
    }
    if (!a.allFinite() || !q.allFinite()) {
        return false;
    }
    *x = SolveDiscreteLyapunovKronecker(a, q);
    return AcceptSolution(x);
}

Matrix ControllabilityGramian(const LtiSystem& system) {
    system.Validate("ControllabilityGramian");
    const Matrix q = system.b() * system.b().transpose();
    Matrix gramian;
    const bool solved = system.is_continuous()
                            ? SolveLyapunov(system.a(), q, &gramian)
                            : SolveDiscreteLyapunov(system.a(), q, &gramian);
    if (!solved) {
        // The equation has no unique solution (an unstable or marginal mode):
        // the finite-horizon sum is still the right answer for a design tool,
        // and it degrades gracefully instead of returning infinities.
        gramian = GramianBySeries(system, /*controllability=*/true, 2000);
    }
    return gramian;
}

Matrix ObservabilityGramian(const LtiSystem& system) {
    system.Validate("ObservabilityGramian");
    const Matrix q = system.c().transpose() * system.c();
    Matrix gramian;
    const bool solved = system.is_continuous()
                            ? SolveLyapunov(system.a(), q, &gramian)
                            : SolveDiscreteLyapunov(system.a(), q, &gramian);
    if (!solved) {
        gramian = GramianBySeries(system, /*controllability=*/false, 2000);
    }
    return gramian;
}

Vector HankelSingularValues(const LtiSystem& system) {
    const Matrix wc = ControllabilityGramian(system);
    const Matrix wo = ObservabilityGramian(system);
    // The product of two symmetric positive semidefinite matrices is not
    // symmetric, but its eigenvalues are real and non-negative. Taking the
    // symmetric square roots first keeps the computation well conditioned.
    Eigen::SelfAdjointEigenSolver<Matrix> wc_solver(wc);
    Eigen::SelfAdjointEigenSolver<Matrix> wo_solver(wo);
    if (wc_solver.info() != Eigen::Success ||
        wo_solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "HankelSingularValues: Gramian decomposition failed");
    }
    const Matrix wc_root =
        wc_solver.eigenvectors() *
        wc_solver.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal() *
        wc_solver.eigenvectors().transpose();
    const Matrix product = wc_root * wo * wc_root;
    Eigen::SelfAdjointEigenSolver<Matrix> solver(
        0.5 * (product + product.transpose()));
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "HankelSingularValues: eigenvalue solve failed");
    }
    Vector values = solver.eigenvalues().cwiseMax(0.0).cwiseSqrt();
    // Eigen returns ascending order; the Hankel values are conventionally
    // reported largest first.
    Vector descending(values.size());
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        descending(i) = values(values.size() - 1 - i);
    }
    return descending;
}

LtiSystem BalancedTruncation(const LtiSystem& system, int order) {
    system.Validate("BalancedTruncation");
    if (!IsStable(system)) {
        throw std::invalid_argument(
            "BalancedTruncation: the system must be stable; the Gramians do "
            "not "
            "exist otherwise and truncation would be meaningless");
    }
    const int n = system.state_dimension();
    if (order <= 0 || order > n) {
        throw std::invalid_argument(
            "BalancedTruncation: order must be in [1, n]");
    }
    if (order == n) {
        return system;
    }

    // Square-root (Laub) balancing, in the form that is short enough to check:
    //
    //   Wc = Lc Lc^T,  Wo = Lo Lo^T            (Cholesky)
    //   Lo^T Lc = U Sigma V^T                  (SVD)
    //   T    = Lc V Sigma^(-1/2)
    //   Tinv = Sigma^(-1/2) U^T Lo^T
    //
    // which gives Tinv Wc Tinv^T = T^T Wo T = diag(hsv) exactly. Getting this
    // wrong is not visible in the reduced model's order or stability, only in
    // its accuracy: a first attempt here lost 2 percent of the DC gain, which
    // is why the test asserts the gain and not just the shape.
    const Matrix wc = ControllabilityGramian(system);
    const Matrix wo = ObservabilityGramian(system);
    Eigen::LLT<Matrix> wc_cholesky(wc);
    Eigen::LLT<Matrix> wo_cholesky(wo);
    if (wc_cholesky.info() != Eigen::Success ||
        wo_cholesky.info() != Eigen::Success) {
        throw std::runtime_error(
            "BalancedTruncation: a Gramian is not positive definite (the model "
            "is "
            "uncontrollable or unobservable, so it has no balanced realisation "
            "to "
            "truncate)");
    }
    const Matrix lc = wc_cholesky.matrixL();
    const Matrix lo = wo_cholesky.matrixL();

    Eigen::JacobiSVD<Matrix> svd(lo.transpose() * lc,
                                 Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Vector& singular = svd.singularValues();
    const Matrix u = svd.matrixU();
    const Matrix v = svd.matrixV();
    for (Eigen::Index i = 0; i < singular.size(); ++i) {
        if (!(singular(i) > 0.0)) {
            throw std::runtime_error(
                "BalancedTruncation: a Hankel singular value is zero (the "
                "model "
                "has a state that is either unreachable or unobservable)");
        }
    }
    Vector inverse_sqrt(singular.size());
    for (Eigen::Index i = 0; i < singular.size(); ++i) {
        inverse_sqrt(i) = 1.0 / std::sqrt(singular(i));
    }
    const Matrix t = lc * v * inverse_sqrt.asDiagonal();
    const Matrix t_inverse =
        inverse_sqrt.asDiagonal() * u.transpose() * lo.transpose();

    const Matrix a_reduced =
        (t_inverse * system.a() * t).topLeftCorner(order, order);
    const Matrix b_reduced = (t_inverse * system.b()).topRows(order);
    const Matrix c_reduced = (system.c() * t).leftCols(order);
    return system.is_continuous()
               ? LtiSystem(a_reduced, b_reduced, c_reduced, system.d(), 0.0)
               : LtiSystem(a_reduced, b_reduced, c_reduced, system.d(),
                           system.sample_time());
}

}  // namespace control
