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

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "control/lti.h"
#include "tests/testing.h"

namespace {

// G(s) = 1 / (s^2 + 3s + 2), poles at -1 and -2, no zeros.
control::LtiSystem SecondOrder() {
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -2.0, -3.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    return control::LtiSystem::Continuous(a, b, c);
}

// Closed-form zero-order hold of 1/(s^2+3s+2):
//   Ad = [[2e^-t - e^-2t, e^-t - e^-2t], [-2e^-t + 2e^-2t, -e^-t + 2e^-2t]]
//   Bd = [1/2 - e^-t + e^-2t/2, e^-t - e^-2t]^T
control::Matrix AnalyticAd(double t) {
    control::Matrix m(2, 2);
    const double e1 = std::exp(-t);
    const double e2 = std::exp(-2.0 * t);
    m << 2.0 * e1 - e2, e1 - e2, -2.0 * e1 + 2.0 * e2, -e1 + 2.0 * e2;
    return m;
}

control::Matrix AnalyticBd(double t) {
    control::Matrix m(2, 1);
    const double e1 = std::exp(-t);
    const double e2 = std::exp(-2.0 * t);
    m << 0.5 - e1 + 0.5 * e2, e1 - e2;
    return m;
}

}  // namespace

TEST(Discretize, ZeroOrderHoldMatchesTheClosedFormSolution) {
    // The van Loan augmented exponential has to reproduce the textbook closed
    // form exactly, not approximately: anything else means the augmented matrix
    // is assembled wrong, and the error is invisible in a step-response plot.
    const double ts = 0.1;
    const control::LtiSystem discrete = control::Discretize(
        SecondOrder(), ts, control::DiscretizationMethod::kZeroOrderHold);
    CHECK_NEAR((discrete.a() - AnalyticAd(ts)).cwiseAbs().maxCoeff(), 0.0,
               1e-14);
    CHECK_NEAR((discrete.b() - AnalyticBd(ts)).cwiseAbs().maxCoeff(), 0.0,
               1e-14);
    CHECK_NEAR(discrete.sample_time(), ts, 0.0);
}

TEST(Discretize, ZeroOrderHoldMapsEachPoleThroughTheExponential) {
    const double ts = 0.1;
    const control::LtiSystem discrete = control::Discretize(SecondOrder(), ts);
    const control::ComplexVector poles = control::Poles(discrete);
    CHECK_EQ(poles.size(), 2);
    // Continuous poles -1 and -2 become exp(-Ts) and exp(-2 Ts). This is the
    // property that makes zero-order hold the default: sampling cannot move a
    // stable pole out of the unit circle.
    CHECK_NEAR(std::abs(poles(0)), std::exp(-ts), 1e-12);
    CHECK_NEAR(std::abs(poles(1)), std::exp(-2.0 * ts), 1e-12);
    CHECK_TRUE(control::IsStable(discrete));
}

TEST(Discretize, ZeroOrderHoldIsExactAtTheSampleInstants) {
    // For a piecewise-constant input the discrete model has to reproduce the
    // continuous response at the sample instants to machine precision. This is
    // the strongest statement available about a discretization, so it is the
    // one the report claims.
    const control::LtiSystem continuous = SecondOrder();
    const control::DiscretizationResult result =
        control::Convert(continuous, 0.05, {});
    CHECK_TRUE(result.ok);
    CHECK_TRUE(result.report.exact_for_constant_input);
    CHECK_NEAR(result.report.step_response_deviation, 0.0, 1e-10);
}

TEST(Discretize, TustinKeepsStabilityAndApproximatesTheResponse) {
    const control::LtiSystem continuous = SecondOrder();
    control::DiscretizationOptions options;
    options.method = control::DiscretizationMethod::kTustin;
    const control::DiscretizationResult result =
        control::Convert(continuous, 0.02, options);
    CHECK_TRUE(result.ok);
    CHECK_TRUE(control::IsStable(result.system));
    // Bilinear discretization is not exact, but at 50 samples per second on a
    // 1 rad/s pole the deviation must be small; a large number here means the
    // transform was applied with the wrong scaling.
    CHECK_TRUE(result.report.step_response_deviation < 1e-3);
    CHECK_FALSE(result.report.exact_for_constant_input);
}

TEST(Discretize, ForwardEulerCanDestabilizeAndSaysSo) {
    // With Ts = 1.5 s the pole at -2 maps to 1 - 3 = -2, outside the unit
    // circle. Cheap discretization methods fail this way, and the whole point
    // of returning a report is that the failure is announced rather than
    // discovered on hardware.
    control::DiscretizationOptions options;
    options.method = control::DiscretizationMethod::kForwardEuler;
    const control::DiscretizationResult result =
        control::Convert(SecondOrder(), 1.5, options);
    CHECK_TRUE(result.ok);
    CHECK_FALSE(result.report.all_poles_mapped_inside_unit_circle);
    CHECK_EQ(result.report.destabilized_poles.size(), std::size_t{1});
    CHECK_FALSE(control::IsStable(result.system));

    // Backward Euler, at the same absurd step, cannot do that.
    control::DiscretizationOptions backward;
    backward.method = control::DiscretizationMethod::kBackwardEuler;
    const control::DiscretizationResult backward_result =
        control::Convert(SecondOrder(), 1.5, backward);
    CHECK_TRUE(backward_result.report.all_poles_mapped_inside_unit_circle);
    CHECK_TRUE(control::IsStable(backward_result.system));
}

TEST(Discretize, RequiredSampleTimeFollowsTheTwentySamplesRule) {
    // 20 samples per period of the fastest frequency of interest.
    CHECK_NEAR(control::RequiredSampleTime(10.0, 20.0),
               2.0 * M_PI / 10.0 / 20.0, 1e-15);
    CHECK_NEAR(control::RequiredSampleTime(10.0, 20.0), 0.0314159265, 1e-9);
    // Refusing an absurd request is part of the contract: four samples per
    // bandwidth is already marginal, fewer is not a design.
    CHECK_THROWS(control::RequiredSampleTime(10.0, 2.0));
    CHECK_THROWS(control::RequiredSampleTime(0.0, 20.0));
}

TEST(Discretize, PadeDelayMatchesItsVerifiedMagnitudeAndPhase) {
    // Values verified against a scipy reference: DC gain exactly 1, and the
    // phase at 1 rad/s for a 0.1 s delay. Order 1 is the classic
    // (1 - sT/2)/(1 + sT/2); orders 2 and 3 tighten the phase.
    const double delay = 0.1;
    const double expected_phase[4] = {0.0, -5.725, -5.734, -5.731};
    for (int order = 1; order <= 3; ++order) {
        const control::LtiSystem approximation =
            control::PadeDelay(delay, order);
        CHECK_EQ(approximation.state_dimension(), order);
        CHECK_TRUE(approximation.is_continuous());
        CHECK_NEAR(control::DcGain(approximation)(0, 0), 1.0, 1e-12);

        const Eigen::MatrixXcd response =
            control::FrequencyResponse(approximation, 1.0);
        CHECK_NEAR(std::abs(response(0, 0)), 1.0, 1e-9);
        CHECK_NEAR(std::arg(response(0, 0)) * 180.0 / M_PI,
                   expected_phase[order], 2e-3);
    }

    // The approximation is stable, which is a requirement: an unstable delay
    // approximation inside a design model would be worse than no delay model.
    CHECK_TRUE(control::IsStable(control::PadeDelay(delay, 2)));

    // Poles of the order-2 approximation are verified values, not guesses.
    const control::ComplexVector poles =
        control::Poles(control::PadeDelay(delay, 2));
    CHECK_NEAR(poles(0).real(), -15.0, 1e-9);
    CHECK_NEAR(std::abs(poles(0).imag()), 19.3649167, 1e-6);

    CHECK_THROWS(control::PadeDelay(0.0, 1));
    CHECK_THROWS(control::PadeDelay(delay, 4));
}

TEST(Discretize, SeriesComposesTwoSystemsWithTheRightGain) {
    // A delay in series with a plant: the DC gain is the product, and the order
    // is the sum.
    const control::LtiSystem plant = SecondOrder();
    const control::LtiSystem delay = control::PadeDelay(0.05, 1);
    const control::LtiSystem combined = control::Series(delay, plant);
    CHECK_EQ(combined.state_dimension(),
             plant.state_dimension() + delay.state_dimension());
    CHECK_NEAR(control::DcGain(combined)(0, 0),
               control::DcGain(plant)(0, 0) * control::DcGain(delay)(0, 0),
               1e-10);

    // A mismatched connection is a mistake, not something to guess about: a
    // two-output plant cannot feed a single-input delay.
    const control::Matrix two_outputs = control::Matrix::Identity(2, 2);
    const control::LtiSystem two_output_plant =
        control::LtiSystem::Continuous(plant.a(), plant.b(), two_outputs);
    CHECK_THROWS(
        control::Series(two_output_plant, control::PadeDelay(0.05, 1)));
}

TEST(Discretize, RefusesNonsenseInput) {
    // A discrete model cannot be converted again.
    CHECK_THROWS(
        control::Convert(control::Discretize(SecondOrder(), 0.1), 0.1, {}));
    // A non-positive or non-finite sample time is not a design decision.
    CHECK_THROWS(control::Convert(SecondOrder(), 0.0, {}));
    CHECK_THROWS(control::Convert(SecondOrder(), -0.1, {}));
    CHECK_THROWS(control::Convert(SecondOrder(),
                                  std::numeric_limits<double>::infinity(), {}));
}
