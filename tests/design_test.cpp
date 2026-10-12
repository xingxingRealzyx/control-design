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

#include <Eigen/Core>
#include <Eigen/LU>
#include <cmath>
#include <stdexcept>

#include "control/discretize.h"
#include "control/lti.h"
#include "tests/testing.h"

namespace {

control::LtiSystem DoubleIntegrator() {
    control::Matrix a(2, 2);
    a << 0.0, 1.0, 0.0, 0.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    return control::LtiSystem::Continuous(a, b, c);
}

control::LtiSystem SecondOrder() {
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -2.0, -3.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    return control::LtiSystem::Continuous(a, b, c);
}

}  // namespace

TEST(Design, LqrOnTheDoubleIntegratorMatchesTheAnalyticSolution) {
    // The textbook case: Q = I, R = 1 gives P = [[sqrt(3), 1], [1, sqrt(3)]],
    // K = [1, sqrt(3)] and a closed loop at -sqrt(3)/2 +/- j/2. The reference
    // values were produced independently with
    // scipy.linalg.solve_continuous_are.
    const control::DesignResult result = control::LinearQuadraticRegulator(
        DoubleIntegrator(), control::Matrix::Identity(2, 2),
        control::Matrix::Identity(1, 1));
    CHECK_TRUE(result.ok);

    const double root_three = std::sqrt(3.0);
    CHECK_NEAR(result.riccati_solution(0, 0), root_three, 1e-12);
    CHECK_NEAR(result.riccati_solution(0, 1), 1.0, 1e-12);
    CHECK_NEAR(result.riccati_solution(1, 1), root_three, 1e-12);
    CHECK_NEAR(result.gain(0, 0), 1.0, 1e-12);
    CHECK_NEAR(result.gain(0, 1), root_three, 1e-12);
    for (Eigen::Index i = 0; i < result.closed_loop_poles.size(); ++i) {
        CHECK_NEAR(result.closed_loop_poles(i).real(), -root_three / 2.0,
                   1e-12);
    }
    // LQR guarantees at least 60 degrees of phase margin, which for this
    // problem shows up as a damping ratio of exactly 1/sqrt(2)... it is 0.5
    // here because the closed loop is -0.866 +/- 0.5j. What is checked is that
    // the design is stable and reported as such.
    CHECK_TRUE(control::IsStable(control::LtiSystem::Continuous(
        DoubleIntegrator().a() - DoubleIntegrator().b() * result.gain,
        DoubleIntegrator().b(), DoubleIntegrator().c())));
    CHECK_EQ(result.warnings.size(), 0u);
}

TEST(Design, LqrSatisfiesItsRiccatiEquation) {
    // The residual is the real assertion: a P that "looks plausible" but does
    // not solve the equation produces a controller that is not optimal, and
    // nothing about the gain reveals it.
    const control::LtiSystem system = SecondOrder();
    control::Matrix q(2, 2);
    q << 10.0, 0.0, 0.0, 1.0;
    control::Matrix r(1, 1);
    r << 0.5;
    const control::DesignResult result =
        control::LinearQuadraticRegulator(system, q, r);

    const control::Matrix p = result.riccati_solution;
    const control::Matrix residual =
        system.a().transpose() * p + p * system.a() -
        p * system.b() * r.inverse() * system.b().transpose() * p + q;
    CHECK_NEAR(residual.cwiseAbs().maxCoeff(), 0.0, 1e-10);
    // K = R^-1 B' P by definition of the LQR gain.
    CHECK_NEAR((result.gain - r.inverse() * system.b().transpose() * p)
                   .cwiseAbs()
                   .maxCoeff(),
               0.0, 1e-12);
    // scipy reference: K = [2.8989794856, 1.0985313188].
    CHECK_NEAR(result.gain(0, 0), 2.8989794856, 1e-9);
    CHECK_NEAR(result.gain(0, 1), 1.0985313188, 1e-9);
    CHECK_NEAR(result.riccati_solution(0, 0), 7.0393104259, 1e-8);
    CHECK_NEAR(result.riccati_solution(1, 1), 0.5492656594, 1e-8);
}

TEST(Design, DiscreteLqrSatisfiesTheDiscreteRiccatiEquation) {
    const control::LtiSystem continuous = SecondOrder();
    const control::LtiSystem discrete = control::Discretize(continuous, 0.1);
    control::Matrix q(2, 2);
    q << 10.0, 0.0, 0.0, 1.0;
    control::Matrix r(1, 1);
    r << 0.5;
    const control::DesignResult result =
        control::DiscreteLinearQuadraticRegulator(discrete, q, r);

    const control::Matrix p = result.riccati_solution;
    const control::Matrix s = r + discrete.b().transpose() * p * discrete.b();
    const control::Matrix residual =
        discrete.a().transpose() * p * discrete.a() - p -
        discrete.a().transpose() * p * discrete.b() * s.inverse() *
            discrete.b().transpose() * p * discrete.a() +
        q;
    // This implementation's solution has a residual of 5e-13; scipy's published
    // value for the same problem has 2e-9, so the check is on the equation
    // rather than on a copied matrix.
    CHECK_NEAR(residual.cwiseAbs().maxCoeff(), 0.0, 1e-10);
    CHECK_NEAR((result.gain -
                s.inverse() * discrete.b().transpose() * p * discrete.a())
                   .cwiseAbs()
                   .maxCoeff(),
               0.0, 1e-12);
    // scipy reference: K = [2.6383607501, 1.0165752846].
    CHECK_NEAR(result.gain(0, 0), 2.6383607501, 1e-8);
    CHECK_NEAR(result.gain(0, 1), 1.0165752846, 1e-8);
    // The closed loop must land inside the unit circle, and it does so at
    // |z| = 0.8149195372 for both poles.
    for (Eigen::Index i = 0; i < result.closed_loop_poles.size(); ++i) {
        CHECK_NEAR(std::abs(result.closed_loop_poles(i)), 0.8149195372, 1e-8);
    }
}

TEST(Design, PolePlacementAchievesTheRequestedPolesExactly) {
    // Ackermann on a controllable SISO model: K = [4, 2] for poles -2, -3.
    // This is the case that caught a reversed polynomial evaluation in phi(A),
    // which produced K = [-11, -13] and an unstable closed loop.
    const control::DesignResult result =
        control::PlacePoles(SecondOrder(), {{-2.0, 0.0}, {-3.0, 0.0}});
    CHECK_TRUE(result.ok);
    CHECK_NEAR(result.gain(0, 0), 4.0, 1e-12);
    CHECK_NEAR(result.gain(0, 1), 2.0, 1e-12);
    CHECK_EQ(result.closed_loop_poles.size(), 2);
    CHECK_NEAR(result.closed_loop_poles(0).real(), -2.0, 1e-12);
    CHECK_NEAR(result.closed_loop_poles(1).real(), -3.0, 1e-12);
    // The achieved poles match the request, so no warning is raised.
    CHECK_EQ(result.warnings.size(), 0u);
    CHECK_TRUE(control::IsStable(control::LtiSystem::Continuous(
        SecondOrder().a() - SecondOrder().b() * result.gain, SecondOrder().b(),
        SecondOrder().c())));
}

TEST(Design, PolePlacementHandlesComplexPairs) {
    // A complex conjugate pair must produce a real gain: placing -1 +/- 2j on
    // the double integrator gives the characteristic polynomial s^2 + 2s + 5,
    // so K = [5, 2].
    const control::DesignResult result =
        control::PlacePoles(DoubleIntegrator(), {{-1.0, 2.0}, {-1.0, -2.0}});
    CHECK_TRUE(result.ok);
    CHECK_NEAR(result.gain(0, 0), 5.0, 1e-12);
    CHECK_NEAR(result.gain(0, 1), 2.0, 1e-12);
    for (Eigen::Index i = 0; i < result.closed_loop_poles.size(); ++i) {
        CHECK_NEAR(result.closed_loop_poles(i).real(), -1.0, 1e-12);
        CHECK_NEAR(std::abs(result.closed_loop_poles(i).imag()), 2.0, 1e-12);
    }
}

TEST(Design, ObserverPolesComeFromDuality) {
    // Placing (A - L C) at -3, -4 for the double integrator gives L = [7; 12].
    const control::DesignResult result = control::PlaceObserverPoles(
        DoubleIntegrator(), {{-3.0, 0.0}, {-4.0, 0.0}});
    CHECK_TRUE(result.ok);
    CHECK_EQ(result.gain.rows(), 2);
    CHECK_EQ(result.gain.cols(), 1);
    CHECK_NEAR(result.gain(0, 0), 7.0, 1e-12);
    CHECK_NEAR(result.gain(1, 0), 12.0, 1e-12);
    const control::LtiSystem error_dynamics = control::LtiSystem::Continuous(
        DoubleIntegrator().a() - result.gain * DoubleIntegrator().c(),
        DoubleIntegrator().b(), DoubleIntegrator().c());
    const control::ComplexVector poles = control::Poles(error_dynamics);
    CHECK_NEAR(poles(0).real(), -3.0, 1e-12);
    CHECK_NEAR(poles(1).real(), -4.0, 1e-12);
}

TEST(Design, IntegralActionTracksAConstantReferenceWithZeroOffset) {
    // The augmented gain must make the closed loop stable and the integrator
    // must actually eliminate the offset: that is the whole point of the extra
    // state, and a design that merely places the poles does not guarantee it.
    const control::AugmentedDesign design = control::AddIntegralAction(
        SecondOrder(), 0, {{-2.0, 0.0}, {-3.0, 0.0}, {-5.0, 0.0}});
    CHECK_EQ(design.augmented.state_dimension(), 3);
    // scipy reference: Ka = [29, 7, -30].
    CHECK_NEAR(design.design.gain(0, 0), 29.0, 1e-9);
    CHECK_NEAR(design.design.gain(0, 1), 7.0, 1e-9);
    CHECK_NEAR(design.design.gain(0, 2), -30.0, 1e-9);

    // Simulate the augmented closed loop against a constant reference and check
    // that the output converges to it.
    control::Matrix augmented_a = design.augmented.a();
    const control::Matrix augmented_b = design.augmented.b();
    const control::Matrix gain = design.design.gain;
    control::Vector state = control::Vector::Zero(3);
    const double reference = 2.0;
    const double dt = 1e-4;
    for (int step = 0; step < 200000; ++step) {
        // z' = r - y for the continuous augmentation.
        const double output = (design.augmented.c() * state)(0, 0);
        const double u = -(gain * state)(0, 0);
        control::Vector derivative = augmented_a * state + augmented_b * u;
        derivative(2) = reference - output;
        state += dt * derivative;
    }
    const double output = (design.augmented.c() * state)(0, 0);
    CHECK_NEAR(output, reference, 1e-4);
}

TEST(Design, ReferenceFeedforwardSolvesTheEquilibrium) {
    // For G(s) = 1/(s^2+3s+2) the DC gain is 0.5, so a reference of 1 needs
    // u_ss = 2 in steady state.
    control::Matrix feedforward;
    control::Vector state_at_reference;
    CHECK_TRUE(control::ReferenceFeedforward(SecondOrder(), 1.0, &feedforward,
                                             &state_at_reference));
    CHECK_NEAR(feedforward(0, 0), 2.0, 1e-9);
    // The equilibrium state is A^-1 B u_ss, which for this plant is [1, 0].
    CHECK_NEAR(state_at_reference(0), 1.0, 1e-9);
    CHECK_NEAR(state_at_reference(1), 0.0, 1e-9);

    // A plant with an integrator is *not* a failure: x = 1 with u = 0 is a
    // perfectly good equilibrium for a reference of 1, which is why an
    // integrator alone is not a reason to add integral action.
    control::Matrix a(1, 1);
    a << 0.0;
    control::Matrix b(1, 1);
    b << 1.0;
    control::Matrix c(1, 1);
    c << 1.0;
    const control::LtiSystem integrator =
        control::LtiSystem::Continuous(a, b, c);
    CHECK_TRUE(control::ReferenceFeedforward(integrator, 1.0, &feedforward,
                                             &state_at_reference));
    CHECK_NEAR(feedforward(0, 0), 0.0, 1e-12);
    CHECK_NEAR(state_at_reference(0), 1.0, 1e-12);

    // What is refused is a reference the model cannot hold at any equilibrium:
    // an output that is identically zero cannot reach 1, whatever the input
    // does.
    control::Matrix c_zero(1, 1);
    c_zero << 0.0;
    const control::LtiSystem blind =
        control::LtiSystem::Continuous(a, b, c_zero);
    CHECK_FALSE(control::ReferenceFeedforward(blind, 1.0, &feedforward));

    // Poles that are not conjugate pairs cannot be placed by a real gain, and
    // pretending otherwise would produce a real part that places different
    // poles.
    CHECK_THROWS(
        control::PlacePoles(SecondOrder(), {{-1.0, 2.0}, {-3.0, 0.0}}));
}

TEST(Design, RefusesWhatItCannotDoAndSaysWhy) {
    // A discrete model cannot be given a continuous LQR.
    const control::LtiSystem discrete = control::Discretize(SecondOrder(), 0.1);
    CHECK_THROWS(control::LinearQuadraticRegulator(
        discrete, control::Matrix::Identity(2, 2),
        control::Matrix::Identity(1, 1)));
    // And a continuous model cannot be given a discrete one.
    CHECK_THROWS(control::DiscreteLinearQuadraticRegulator(
        SecondOrder(), control::Matrix::Identity(2, 2),
        control::Matrix::Identity(1, 1)));

    // R must be positive definite: R = 0 makes control free and the gain
    // infinite.
    control::Matrix zero_r = control::Matrix::Zero(1, 1);
    CHECK_THROWS(control::LinearQuadraticRegulator(
        SecondOrder(), control::Matrix::Identity(2, 2), zero_r));

    // An unstable mode that the input cannot reach makes the whole request
    // impossible, and the message has to name the mode.
    control::Matrix a(2, 2);
    a << -1.0, 0.0, 0.0, 1.0;
    control::Matrix b(2, 1);
    b << 1.0, 0.0;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    const control::LtiSystem unstabilizable =
        control::LtiSystem::Continuous(a, b, c);
    CHECK_THROWS(control::LinearQuadraticRegulator(
        unstabilizable, control::Matrix::Identity(2, 2),
        control::Matrix::Identity(1, 1)));
    CHECK_THROWS(
        control::PlacePoles(unstabilizable, {{-1.0, 0.0}, {-2.0, 0.0}}));

    // A multi-input model has a family of gains placing the same poles, and
    // this function does not pretend to pick one.
    control::Matrix two_inputs(2, 2);
    two_inputs << 0.0, 1.0, 1.0, 0.0;
    CHECK_THROWS(control::PlacePoles(
        control::LtiSystem::Continuous(SecondOrder().a(), two_inputs,
                                       SecondOrder().c()),
        {{-1.0, 0.0}, {-2.0, 0.0}}));

    // The number of requested poles has to match the model.
    CHECK_THROWS(control::PlacePoles(SecondOrder(), {{-1.0, 0.0}}));
    // An unobservable model has no observer.
    control::Matrix sees_nothing(1, 2);
    sees_nothing << 0.0, 0.0;
    CHECK_THROWS(control::PlaceObserverPoles(
        control::LtiSystem::Continuous(SecondOrder().a(), SecondOrder().b(),
                                       sees_nothing),
        {{-3.0, 0.0}, {-4.0, 0.0}}));
}

TEST(Design, GainMagnitudeIsReportedWithAWarning) {
    // Placing the poles very far out on a plant with a nearly unreachable mode
    // asks for enormous gains, which is the failure that looks fine in a pole
    // list and dies on a real actuator.
    control::Matrix a(2, 2);
    a << -1.0, 0.0, 0.0, -2.0;
    control::Matrix b(2, 1);
    b << 1.0, 1e-4;
    control::Matrix c(1, 2);
    c << 1.0, 1.0;
    const control::LtiSystem system = control::LtiSystem::Continuous(a, b, c);
    const control::DesignResult result =
        control::PlacePoles(system, {{-1000.0, 0.0}, {-1200.0, 0.0}});
    CHECK_TRUE(result.ok);
    CHECK_TRUE(result.gain_magnitude > 1e5);
    CHECK_TRUE(result.warnings.size() > 0);
}

TEST(Design, SteadyStateKalmanGainAgreesWithTheRicciatiSolution) {
    // The Kalman gain is the dual of the LQR gain, so it can be cross-checked
    // against the DARE solver rather than only against a simulation.
    control::Matrix a(2, 2);
    a << 0.9, 0.1, 0.0, 0.8;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    control::Matrix process(2, 2);
    process << 0.01, 0.0, 0.0, 0.01;
    control::Matrix measurement(1, 1);
    measurement << 0.1;

    control::Matrix gain;
    control::Matrix prior;
    CHECK_TRUE(control::SolveDiscreteSteadyStateGain(a, c, process, measurement,
                                                     &gain, &prior));
    CHECK_EQ(gain.rows(), 2);
    CHECK_EQ(gain.cols(), 1);
    // Prior covariance is symmetric positive definite at the fixed point.
    CHECK_NEAR((prior - prior.transpose()).cwiseAbs().maxCoeff(), 0.0, 1e-14);
    CHECK_TRUE(prior(0, 0) > 0.0);

    // The fixed point satisfies the covariance Riccati equation:
    //   P = A P A' - A P C' (C P C' + R)^-1 C P A' + Q
    const control::Matrix s = c * prior * c.transpose() + measurement;
    const control::Matrix residual =
        a * prior * a.transpose() -
        a * prior * c.transpose() * s.inverse() * c * prior * a.transpose() +
        process - prior;
    CHECK_NEAR(residual.cwiseAbs().maxCoeff(), 0.0, 1e-10);
}
