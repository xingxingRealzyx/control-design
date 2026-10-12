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

#include <Eigen/Core>
#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

#include "tests/testing.h"

namespace {

// G(s) = (s + 3) / (s^2 + 3s + 2): poles -1, -2; zero -3; DC gain 1.5.
control::LtiSystem WithZero() {
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -2.0, -3.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 3.0, 1.0;
    return control::LtiSystem::Continuous(a, b, c);
}

// G(s) = 1 / (s^2 + 3s + 2): the same poles, no zeros; DC gain 0.5.
control::LtiSystem WithoutZero() {
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -2.0, -3.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    return control::LtiSystem::Continuous(a, b, c);
}

bool HasPoleAt(const control::ComplexVector& poles, double real, double imag,
               double tolerance) {
    for (Eigen::Index i = 0; i < poles.size(); ++i) {
        if (std::abs(poles(i).real() - real) < tolerance &&
            std::abs(poles(i).imag() - imag) < tolerance) {
            return true;
        }
    }
    return false;
}

}  // namespace

TEST(Lti, PolesAreTheAnalyticValuesAndSortedByDominance) {
    const control::ComplexVector poles = control::Poles(WithZero());
    CHECK_EQ(poles.size(), 2);
    // Descending real part: the dominant (slowest) pole comes first.
    CHECK_NEAR(poles(0).real(), -1.0, 1e-12);
    CHECK_NEAR(poles(1).real(), -2.0, 1e-12);
    CHECK_NEAR(poles(0).imag(), 0.0, 1e-12);
    CHECK_TRUE(HasPoleAt(poles, -2.0, 0.0, 1e-12));
}

TEST(Lti, ZerosMatchTheAnalyticTransferFunction) {
    // The zero of (s+3)/(s^2+3s+2) is at -3. This is the case that caught a
    // rank test which kept spurious pencil eigenvalues.
    const control::PoleZeroResult with_zero =
        control::PolesAndZeros(WithZero());
    CHECK_TRUE(with_zero.has_zeros);
    CHECK_EQ(with_zero.zeros.size(), 1);
    CHECK_NEAR(with_zero.zeros(0).real(), -3.0, 1e-9);
    CHECK_NEAR(with_zero.zeros(0).imag(), 0.0, 1e-9);

    // A strictly proper second-order system with a constant numerator has no
    // finite zeros, and inventing one is worse than reporting none.
    const control::PoleZeroResult without_zero =
        control::PolesAndZeros(WithoutZero());
    CHECK_FALSE(without_zero.has_zeros);
    CHECK_EQ(without_zero.zeros.size(), 0);
}

TEST(Lti, ZeroComputationRefusesWhatItCannotDoReliably) {
    // A discrete model: zero-order hold introduces sampling zeros (a minimum
    // phase continuous plant can become non-minimum-phase when sampled), and
    // the pencil formulation does not reproduce them. Refusing is the honest
    // answer.
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -2.0, -3.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    const control::LtiSystem discrete =
        control::LtiSystem::Discrete(a, b, c, 0.1);
    CHECK_THROWS(control::PolesAndZeros(discrete));

    // A non-square model: the finite-zero count depends on the pencil's rest
    // dimension, which this implementation does not resolve.
    control::Matrix two_outputs(2, 2);
    two_outputs << 1.0, 0.0, 0.0, 1.0;
    CHECK_THROWS(control::PolesAndZeros(
        control::LtiSystem::Continuous(a, b, two_outputs)));
}

TEST(Lti, StabilityAndMargin) {
    const control::LtiSystem stable = WithZero();
    CHECK_TRUE(control::IsStable(stable));
    // The dominant pole is at -1, so the continuous margin is 1.
    CHECK_NEAR(control::StabilityMargin(stable), 1.0, 1e-12);

    // An integrator is not stable: a marginally stable pole does not decay, and
    // a design tool that calls it stable is lying to the user.
    control::Matrix integrator(1, 1);
    integrator << 0.0;
    control::Matrix b(1, 1);
    b << 1.0;
    control::Matrix c(1, 1);
    c << 1.0;
    const control::LtiSystem marginal =
        control::LtiSystem::Continuous(integrator, b, c);
    CHECK_FALSE(control::IsStable(marginal));
    CHECK_NEAR(control::StabilityMargin(marginal), 0.0, 1e-12);

    // Discrete: a pole at 0.5 is stable with margin 0.5; one at 1.0 is not.
    control::Matrix half(1, 1);
    half << 0.5;
    const control::LtiSystem discrete_stable =
        control::LtiSystem::Discrete(half, b, c, 0.1);
    CHECK_TRUE(control::IsStable(discrete_stable));
    CHECK_NEAR(control::StabilityMargin(discrete_stable), 0.5, 1e-12);
}

TEST(Lti, PoleCharacteristicsReportDamping) {
    // A well-known second-order system: wn = 2 rad/s, zeta = 0.5 gives poles at
    // -1 +/- j sqrt(3).
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -4.0, -2.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    const control::PoleCharacteristics characteristics =
        control::DescribePoles(control::LtiSystem::Continuous(a, b, c));
    CHECK_EQ(characteristics.poles.size(), 2);
    for (Eigen::Index i = 0; i < 2; ++i) {
        CHECK_NEAR(characteristics.natural_frequencies(i), 2.0, 1e-9);
        CHECK_NEAR(characteristics.damping_ratios(i), 0.5, 1e-9);
    }
}

TEST(Lti, DcGainMatchesTheTransferFunction) {
    CHECK_NEAR(control::DcGain(WithZero())(0, 0), 1.5, 1e-12);
    CHECK_NEAR(control::DcGain(WithoutZero())(0, 0), 0.5, 1e-12);
}

TEST(Lti, FrequencyResponseMatchesTheAnalyticMagnitudeAndPhase) {
    // |G(j w)| and phase of (s+3)/(s^2+3s+2) at w = 1, straight from the
    // definition: |G| = |j+3| / |(j)^2 + 3j + 2|.
    const double omega = 1.0;
    const std::complex<double> numerator(3.0, omega);
    const std::complex<double> denominator =
        std::complex<double>(-1.0, 0.0) +
        std::complex<double>(0.0, 3.0 * omega) + 2.0;
    const std::complex<double> expected = numerator / denominator;

    const Eigen::MatrixXcd response =
        control::FrequencyResponse(WithZero(), omega);
    CHECK_EQ(response.rows(), 1);
    CHECK_EQ(response.cols(), 1);
    CHECK_NEAR(std::abs(response(0, 0)), std::abs(expected), 1e-12);
    CHECK_NEAR(std::arg(response(0, 0)), std::arg(expected), 1e-12);
}

TEST(Lti, LyapunovSolversSatisfyTheirEquations) {
    // Continuous: A X + X A^T + Q = 0. The residual is the assertion; a solver
    // that returns a plausible-looking matrix without satisfying the equation
    // is the failure mode worth catching.
    const control::Matrix a = WithZero().a();
    control::Matrix q(2, 2);
    q << 1.0, 0.5, 0.5, 2.0;
    control::Matrix x;
    CHECK_TRUE(control::SolveLyapunov(a, q, &x));
    const control::Matrix residual = a * x + x * a.transpose() + q;
    CHECK_NEAR(residual.cwiseAbs().maxCoeff(), 0.0, 1e-10);
    // The solution of a symmetric problem is symmetric.
    CHECK_NEAR((x - x.transpose()).cwiseAbs().maxCoeff(), 0.0, 1e-12);

    // Discrete: A X A^T - X + Q = 0.
    control::Matrix ad(2, 2);
    ad << 0.9, 0.1, 0.0, 0.8;
    control::Matrix xd;
    CHECK_TRUE(control::SolveDiscreteLyapunov(ad, q, &xd));
    const control::Matrix discrete_residual = ad * xd * ad.transpose() - xd + q;
    CHECK_NEAR(discrete_residual.cwiseAbs().maxCoeff(), 0.0, 1e-10);
}

TEST(Lti, GramiansAndHankelValuesAreOrderedAndPositive) {
    const control::Vector hsv = control::HankelSingularValues(WithoutZero());
    CHECK_EQ(hsv.size(), 2);
    CHECK_TRUE(hsv(0) > 0.0);
    CHECK_TRUE(hsv(1) > 0.0);
    // Descending order is the convention, and it is what makes "the first k
    // values capture most of the energy" a statement someone can act on.
    CHECK_TRUE(hsv(0) >= hsv(1));

    // Cross-check against the Gramians directly: hsv^2 are the eigenvalues of
    // Wc Wo, so their product must match det(Wc) det(Wo).
    const control::Matrix wc = control::ControllabilityGramian(WithoutZero());
    const control::Matrix wo = control::ObservabilityGramian(WithoutZero());
    CHECK_NEAR(wc.determinant() * wo.determinant(),
               hsv(0) * hsv(0) * hsv(1) * hsv(1), 1e-10);
}

TEST(Lti, BalancedTruncationKeepsTheDcGainOfTheKeptDynamics) {
    // A stable third-order system where one state is nearly irrelevant: the
    // truncated model must keep the DC gain and stay stable.
    control::Matrix a = control::Matrix::Zero(3, 3);
    a << -1.0, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0, -100.0;
    control::Matrix b(3, 1);
    b << 1.0, 0.1, 0.001;
    control::Matrix c(1, 3);
    c << 1.0, 1.0, 1.0;
    const control::LtiSystem full = control::LtiSystem::Continuous(a, b, c);
    const control::LtiSystem reduced = control::BalancedTruncation(full, 2);
    CHECK_EQ(reduced.state_dimension(), 2);
    CHECK_TRUE(control::IsStable(reduced));
    // The reduced model is not exact, but with Hankel values spanning five
    // decades the third state contributes almost nothing: the DC gain must be
    // preserved to better than 1e-5 (a wrong balancing transformation loses
    // percent-level accuracy instead, which is how the first version failed).
    CHECK_NEAR(control::DcGain(reduced)(0, 0), control::DcGain(full)(0, 0),
               1e-5);
    // Truncating a stable system to an unstable one would be a bug worth
    // catching, so the poles are checked as well.
    const control::ComplexVector poles = control::Poles(reduced);
    for (Eigen::Index i = 0; i < poles.size(); ++i) {
        CHECK_TRUE(poles(i).real() < 0.0);
    }
}
