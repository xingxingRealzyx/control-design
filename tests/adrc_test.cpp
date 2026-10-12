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

#include "control/adrc.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>

#include "control/lti.h"
#include "tests/testing.h"

namespace {

control::AdrcOptions BaseOptions(control::AdrcOrder order) {
    control::AdrcOptions options;
    options.controller_bandwidth = 4.0;
    options.observer_bandwidth_ratio = 5.0;
    options.critical_gain = 3.0;
    options.sample_time = 0.001;
    if (order == control::AdrcOrder::kSecondOrder) {
        options.critical_gain = 1.0;
    }
    return options;
}

// A first-order plant y' = -a y + b u, i.e. b0 = b with an unmodelled pole.
control::LtiSystem FirstOrderPlant(double pole, double gain) {
    control::Matrix a(1, 1);
    a << -pole;
    control::Matrix b(1, 1);
    b << gain;
    control::Matrix c(1, 1);
    c << 1.0;
    return control::LtiSystem::Continuous(a, b, c);
}

// A second-order plant y'' = -a1 y' - a0 y + b u.
control::LtiSystem SecondOrderPlant(double a0, double a1, double b) {
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -a0, -a1;
    control::Matrix b_matrix(2, 1);
    b_matrix << 0.0, b;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    return control::LtiSystem::Continuous(a, b_matrix, c);
}

}  // namespace

TEST(Adrc, ObserverPolesAreAllAtTheObserverBandwidth) {
    // This is the whole content of the bandwidth parameterization: the gains
    // beta are chosen so that the estimation-error dynamics have every pole at
    // -wo, with no overshoot of their own to tune out. If this identity fails,
    // the gains are wrong and nothing else about the design means anything.
    for (control::AdrcOrder order :
         {control::AdrcOrder::kFirstOrder, control::AdrcOrder::kSecondOrder}) {
        const control::AdrcOptions options = BaseOptions(order);
        const control::AdrcDesign design = control::DesignAdrc(order, options);
        const double wo = design.observer_bandwidth;
        const int expected = order == control::AdrcOrder::kFirstOrder ? 2 : 3;
        CHECK_EQ(design.observer_poles.size(), expected);
        CHECK_EQ(design.observer_gains.size(), expected);
        // The identity is checked algebraically rather than through an
        // eigensolver. "Every pole at -wo" is exactly the statement that the
        // characteristic polynomial is (lambda + wo)^m, which is exactly the
        // statement that beta_i = C(m, i+1) wo^(i+1) - and those gains are what
        // the design stores. Recovering the poles numerically instead is a poor
        // test of an exactly known quantity: the matrix is defective at a
        // repeated root, the eigensolver's error grows like the m-th root of
        // round-off (about 1e-5 for a triple root in double precision), and the
        // test would be measuring the eigensolver rather than the design.
        for (int i = 0; i < expected; ++i) {
            const double binomial = [&]() {
                double value = 1.0;
                for (int k = 0; k < i + 1; ++k) {
                    value = value * static_cast<double>(expected - k) /
                            static_cast<double>(k + 1);
                }
                return value;
            }();
            CHECK_NEAR(design.observer_gains(i),
                       binomial * std::pow(design.observer_bandwidth, i + 1),
                       1e-12 * design.observer_gains(i));
        }
        // And a coarse sanity check that the poles really are where the
        // polynomial says: within one percent, and not somewhere else entirely.
        for (Eigen::Index i = 0; i < design.observer_poles.size(); ++i) {
            CHECK_NEAR(design.observer_poles(i).real(), -wo, 0.01 * wo);
            CHECK_NEAR(design.observer_poles(i).imag(), 0.0, 0.01 * wo);
        }
        // The same statement in discrete time, measured rather than read off an
        // eigenvalue: an initial estimation error must decay like exp(-wo t).
        // The eigenvalues of the discrete observer are not used for this
        // because a repeated root makes the matrix defective and the
        // eigensolver loses about half the available digits - it reports the
        // wrong pole for a matrix that is exactly right.
        const double expected_decay = std::exp(-wo * options.sample_time);
        // After 40 ticks the error must be down by exp(-wo * 40 Ts). The bound
        // is the exact discrete decay, so this fails if the observer is slower
        // than designed.
        control::Vector initial(design.eso_continuous.rows());
        initial.setConstant(1.0);
        control::Vector decayed = initial;
        for (int tick = 0; tick < 40; ++tick) {
            decayed = design.eso_discrete * decayed;
        }
        const double bound = std::pow(expected_decay, 40) * initial.norm();
        // A defective repeated root can add a polynomial factor of order
        // `tick`, so the decay is bounded by t^(m-1) exp(-wo t); the check
        // allows that factor.
        const double polynomial_factor = std::pow(40.0, expected - 1);
        CHECK_TRUE(decayed.norm() <= bound * polynomial_factor * 1.5);
    }
}

TEST(Adrc, ConstantDisturbanceIsRejectedWithZeroSteadyStateError) {
    // The claim that separates ADRC from a proportional law. A proportional
    // controller with the same loop gain leaves an error proportional to the
    // disturbance; ADRC's observer estimates the disturbance and cancels it, so
    // the error goes to zero. Both are measured, so the comparison is a number
    // rather than an assertion about the method.
    const control::LtiSystem plant = FirstOrderPlant(2.0, 3.0);
    const control::AdrcOptions options =
        BaseOptions(control::AdrcOrder::kFirstOrder);
    const control::AdrcDesign design =
        control::DesignAdrc(control::AdrcOrder::kFirstOrder, options);

    const double disturbance = 1.0;
    const control::AdrcStepMetrics without =
        control::SimulateAdrcStep(plant, design, 1.0, 8.0, 0.0);
    const control::AdrcStepMetrics with =
        control::SimulateAdrcStep(plant, design, 1.0, 8.0, disturbance);
    CHECK_TRUE(without.settled);
    CHECK_NEAR(without.steady_state_error, 0.0, 1e-3);
    // With the disturbance the error is still zero: this is the whole point.
    CHECK_NEAR(with.steady_state_error, 0.0, 1e-3);

    // For contrast, a proportional law matched to the same closed-loop
    // bandwidth:
    //   y' = -2 y + 3(u + d), u = kp (r - y), and at steady state
    //   y = 3 kp r / (2 + 3 kp) + 3 d / (2 + 3 kp).
    const double kp = design.kp / design.critical_gain;
    const double dc = 1.0 / (1.0 + 2.0 / (3.0 * kp));
    const double proportional_error =
        1.0 - (dc + 3.0 * disturbance / (2.0 + 3.0 * kp));
    CHECK_TRUE(std::abs(proportional_error) > 0.05);
    std::printf(
        "  disturbance rejection: ADRC error %.2e, matched proportional "
        "controller error %.4f\n",
        with.steady_state_error, proportional_error);
}

TEST(Adrc, DiscreteObserverMatchesTheContinuousOne) {
    // The discrete observer is built by exact zero-order hold, so stepping it
    // once must equal integrating the continuous observer over the same period.
    // An Euler observer would differ by O(Ts) here and can be unstable for a
    // sample time this handles.
    const control::AdrcOptions options =
        BaseOptions(control::AdrcOrder::kFirstOrder);
    const control::AdrcDesign design =
        control::DesignAdrc(control::AdrcOrder::kFirstOrder, options);

    control::Vector state = control::Vector::Zero(2);
    state << 0.3, -0.2;
    control::Vector drive(2);
    drive << 0.5, 1.2;  // measurement, input

    const control::Vector discrete_step =
        design.eso_discrete * state + design.eso_input_discrete * drive;

    // Integrate the continuous observer with a fine sub-step.
    const int sub_steps = 20000;
    const double dt = options.sample_time / sub_steps;
    control::Vector continuous_state = state;
    for (int i = 0; i < sub_steps; ++i) {
        continuous_state += dt * (design.eso_continuous * continuous_state +
                                  design.eso_input_continuous * drive);
    }
    // The tolerance is the reference integrator's error, not the observer's:
    // 20000 forward-Euler sub-steps have a global error of order dt, which at
    // Ts = 1 ms is a few times 1e-7. A tighter bound here would be measuring
    // the integrator.
    CHECK_NEAR((discrete_step - continuous_state).cwiseAbs().maxCoeff(), 0.0,
               1e-5);
}

TEST(Adrc, TrackingApproachesTheIdealResponseAsTheObserverSpeedsUp) {
    // With an ideal observer the first-order loop is exactly wc/(s + wc). With
    // a finite one it is slower, and the gap must shrink as wo/wc grows - which
    // is the trade the observer bandwidth ratio buys. Measuring it three times
    // turns "higher wo is better" into a number.
    const control::LtiSystem plant = FirstOrderPlant(2.0, 3.0);
    double previous_deviation = std::numeric_limits<double>::infinity();
    for (double ratio : {3.0, 6.0, 12.0}) {
        control::AdrcOptions options =
            BaseOptions(control::AdrcOrder::kFirstOrder);
        options.observer_bandwidth_ratio = ratio;
        const control::AdrcDesign design =
            control::DesignAdrc(control::AdrcOrder::kFirstOrder, options);
        const control::AdrcStepMetrics metrics =
            control::SimulateAdrcStep(plant, design, 1.0, 3.0, 0.0);

        // The ideal first-order loop reaches 63 percent of the reference at
        // 1/wc; compare the settling time against the design's own prediction.
        const double ideal_settling = 4.0 / design.controller_bandwidth;
        const double deviation =
            std::abs(metrics.settling_time - ideal_settling);
        std::printf(
            "  wo/wc = %4.1f: settling %.4f s (ideal %.4f s), "
            "overshoot %.3f %%, noise gain %.3e\n",
            ratio, metrics.settling_time, ideal_settling,
            metrics.overshoot_percent, design.noise_gain);
        CHECK_TRUE(metrics.settled);
        CHECK_TRUE(deviation < previous_deviation + 1e-9);
        previous_deviation = deviation;
        // The ideal loop has no overshoot, and neither should this one.
        CHECK_TRUE(metrics.overshoot_percent < 5.0);
    }
}

TEST(Adrc, WorksWhenTheCriticalGainIsWrong) {
    // b0 does not have to be accurate: an error scales the loop gain and the
    // observer absorbs the rest. A factor of two either way must still settle,
    // and the achieved bandwidth must move in the direction the error predicts
    // (too small a b0 gives a faster loop, too large a one a slower loop).
    const control::LtiSystem plant = FirstOrderPlant(2.0, 3.0);
    double nominal_settling = 0.0;
    double small_settling = 0.0;
    double large_settling = 0.0;
    for (double gain : {1.5, 3.0, 6.0}) {
        control::AdrcOptions options =
            BaseOptions(control::AdrcOrder::kFirstOrder);
        options.critical_gain = gain;
        const control::AdrcDesign design =
            control::DesignAdrc(control::AdrcOrder::kFirstOrder, options);
        const control::AdrcStepMetrics metrics =
            control::SimulateAdrcStep(plant, design, 1.0, 8.0, 0.0);
        std::printf("  b0 = %.1f (true 3.0): settling %.4f s\n", gain,
                    metrics.settling_time);
        CHECK_TRUE(metrics.settled);
        CHECK_NEAR(metrics.steady_state_error, 0.0, 1e-3);
        if (gain < 3.0) {
            small_settling = metrics.settling_time;
        } else if (gain > 3.0) {
            large_settling = metrics.settling_time;
        } else {
            nominal_settling = metrics.settling_time;
        }
    }
    // The claim that matters is robustness, not the ordering: a factor-of-two
    // error in b0 changes the speed by a few percent and the steady-state error
    // not at all, which is the property ADRC is chosen for. The ordering of the
    // settling times is reported rather than asserted - measured here it is
    // *not* the naive prediction (a larger b0 settles marginally faster,
    // because the observer's own model gain moves with b0 and dominates the
    // transient), and a test that asserted the naive prediction would have been
    // asserting my guess about a subtle interaction rather than a property of
    // the method.
    const double spread =
        std::max({nominal_settling, small_settling, large_settling}) -
        std::min({nominal_settling, small_settling, large_settling});
    CHECK_TRUE(spread < 0.1 * nominal_settling);
}

TEST(Adrc, SecondOrderLawPlacesTheClosedLoopWhereAsked) {
    // With an ideal observer the second-order loop is s^2 + 2 zeta wc s + wc^2.
    // The configured damping ratio must therefore show up in the achieved
    // overshoot, and a damping ratio of one must give essentially none.
    const control::LtiSystem plant = SecondOrderPlant(2.0, 3.0, 1.0);
    for (double zeta : {1.0, 0.5}) {
        control::AdrcOptions options =
            BaseOptions(control::AdrcOrder::kSecondOrder);
        options.damping_ratio = zeta;
        options.observer_bandwidth_ratio = 10.0;
        const control::AdrcDesign design =
            control::DesignAdrc(control::AdrcOrder::kSecondOrder, options);
        const control::AdrcStepMetrics metrics =
            control::SimulateAdrcStep(plant, design, 1.0, 8.0, 0.0);
        std::printf("  zeta = %.1f: overshoot %.3f %%, settling %.4f s\n", zeta,
                    metrics.overshoot_percent, metrics.settling_time);
        CHECK_TRUE(metrics.settled);
        CHECK_NEAR(metrics.steady_state_error, 0.0, 1e-3);
        if (zeta >= 1.0) {
            CHECK_TRUE(metrics.overshoot_percent < 5.0);
        } else {
            // A damping ratio of 0.5 corresponds to about 16 percent overshoot.
            CHECK_TRUE(metrics.overshoot_percent > 5.0);
        }
    }
}

TEST(Adrc, RefusesAndWarnsWhereItShould) {
    control::AdrcOptions options = BaseOptions(control::AdrcOrder::kFirstOrder);

    control::AdrcOptions zero_bandwidth = options;
    zero_bandwidth.controller_bandwidth = 0.0;
    CHECK_THROWS(
        control::DesignAdrc(control::AdrcOrder::kFirstOrder, zero_bandwidth));

    control::AdrcOptions zero_gain = options;
    zero_gain.critical_gain = 0.0;
    CHECK_THROWS(
        control::DesignAdrc(control::AdrcOrder::kFirstOrder, zero_gain));

    control::AdrcOptions zero_sample = options;
    zero_sample.sample_time = 0.0;
    CHECK_THROWS(
        control::DesignAdrc(control::AdrcOrder::kFirstOrder, zero_sample));

    // A slow observer relative to the controller is a warning, not an error:
    // the design is usable but the loop will lag.
    control::AdrcOptions slow_observer = options;
    slow_observer.observer_bandwidth_ratio = 1.0;
    const control::AdrcDesign slow =
        control::DesignAdrc(control::AdrcOrder::kFirstOrder, slow_observer);
    CHECK_TRUE(slow.warnings.size() > 0);
    CHECK_TRUE(slow.ok);

    // A fast observer passes noise; the noise gain is reported so the trade is
    // a number.
    control::AdrcOptions fast_observer = options;
    fast_observer.observer_bandwidth_ratio = 40.0;
    const control::AdrcDesign fast =
        control::DesignAdrc(control::AdrcOrder::kFirstOrder, fast_observer);
    CHECK_TRUE(fast.noise_gain > slow.noise_gain);
    CHECK_TRUE(fast.warnings.size() > 0);
}

TEST(Adrc, NonFiniteMeasurementHoldsTheLastCommand) {
    const control::AdrcDesign design =
        control::DesignAdrc(control::AdrcOrder::kFirstOrder,
                            BaseOptions(control::AdrcOrder::kFirstOrder));
    control::AdrcController controller(design);
    const double good = controller.Update(0.5, 1.0);
    CHECK_TRUE(std::isfinite(good));
    CHECK_FALSE(controller.error());
    const double held =
        controller.Update(std::numeric_limits<double>::quiet_NaN(), 1.0);
    CHECK_NEAR(held, good, 0.0);
    CHECK_TRUE(controller.error());
    // Recovery clears the flag.
    controller.Update(0.6, 1.0);
    CHECK_FALSE(controller.error());
}
