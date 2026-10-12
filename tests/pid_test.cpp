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

#include "control/pid.h"

#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "control/lti.h"
#include "tests/testing.h"

namespace {

// G(s) = b / (s^2 + a1 s + a0) in controller canonical form.
control::LtiSystem SecondOrder(double a0, double a1, double b) {
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -a0, -a1;
    control::Matrix b_matrix(2, 1);
    b_matrix << 0.0, b;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    return control::LtiSystem::Continuous(a, b_matrix, c);
}

control::LtiSystem FirstOrder(double a0, double b) {
    control::Matrix a(1, 1);
    a << -a0;
    control::Matrix b_matrix(1, 1);
    b_matrix << b;
    control::Matrix c(1, 1);
    c << 1.0;
    return control::LtiSystem::Continuous(a, b_matrix, c);
}

}  // namespace

TEST(Pid, GainsConvertBetweenTheTwoParametrisations) {
    // Ti = Kp/Ki and Td = Kd/Kp, the definitions every industrial datasheet
    // uses.
    const control::PidGains gains =
        control::PidGains::FromIntegralAndDerivativeTimes(2.0, 4.0, 0.5);
    CHECK_NEAR(gains.kp, 2.0, 0.0);
    CHECK_NEAR(gains.ki, 0.5, 1e-15);  // Kp/Ti
    CHECK_NEAR(gains.kd, 1.0, 0.0);    // Kp*Td
    CHECK_NEAR(gains.integral_time(), 4.0, 1e-12);
    CHECK_NEAR(gains.derivative_time(), 0.5, 1e-12);

    // A pure P controller has infinite integral time, which is the honest
    // answer rather than a division by zero.
    control::PidGains proportional;
    proportional.kp = 3.0;
    CHECK_TRUE(std::isinf(proportional.integral_time()));
    CHECK_NEAR(proportional.derivative_time(), 0.0, 0.0);
}

TEST(Pid, SpecificationsMapToPolesByTheStandardFormulas) {
    // 10 percent overshoot corresponds to zeta = 0.5912 (from
    // Mp = exp(-pi zeta / sqrt(1-zeta^2))), and ts = 1 s to wn = 4/(zeta ts).
    control::PidSpecifications specifications;
    specifications.overshoot_percent = 10.0;
    specifications.settling_time = 1.0;
    double zeta = 0.0;
    double wn = 0.0;
    control::SpecificationsToPoles(specifications, &zeta, &wn);
    CHECK_NEAR(zeta, 0.5912, 1e-3);
    CHECK_NEAR(wn, 4.0 / (zeta * 1.0), 1e-9);

    // A zero overshoot request is clamped rather than dividing by zero.
    specifications.overshoot_percent = 0.0;
    control::SpecificationsToPoles(specifications, &zeta, &wn);
    CHECK_TRUE(zeta > 0.0 && zeta <= 1.0);
}

TEST(Pid, ZieglerNicholsTableIsExact) {
    // The classic table, checked against its own definition.
    const control::PidDesign pid =
        control::TuneZieglerNichols(10.0, 2.0, control::PidForm::kPid);
    CHECK_NEAR(pid.gains.kp, 6.0, 1e-12);                  // 0.6 Ku
    CHECK_NEAR(pid.gains.integral_time(), 1.0, 1e-12);     // Pu/2
    CHECK_NEAR(pid.gains.derivative_time(), 0.25, 1e-12);  // Pu/8
    CHECK_TRUE(pid.has_ultimate_values);
    // The table is aggressive and the design says so instead of letting the
    // commissioning session find out.
    CHECK_TRUE(pid.notes.find("25 percent overshoot") != std::string::npos);

    const control::PidDesign pi =
        control::TuneZieglerNichols(10.0, 2.0, control::PidForm::kPi);
    CHECK_NEAR(pi.gains.kp, 4.5, 1e-12);  // 0.45 Ku
    CHECK_NEAR(pi.gains.ki, 4.5, 1e-12);  // Kp/Ti with Ti = 1

    const control::PidDesign p =
        control::TuneZieglerNichols(10.0, 2.0, control::PidForm::kP);
    CHECK_NEAR(p.gains.kp, 5.0, 1e-12);  // 0.5 Ku
    CHECK_NEAR(p.gains.ki, 0.0, 0.0);

    // Derating scales the proportional gain and therefore the integral gain
    // too.
    const control::PidDesign derated =
        control::TuneZieglerNichols(10.0, 2.0, control::PidForm::kPid, 0.5);
    CHECK_NEAR(derated.gains.kp, 3.0, 1e-12);
    CHECK_TRUE(derated.notes.find("derated") != std::string::npos);
}

TEST(Pid, UltimateGainIsFoundFromTheModelAtThePhaseCrossover) {
    // For G(s) = 1/(s+1)^3 the phase reaches -180 degrees at w = sqrt(3), where
    // |G| = 1/8, so Ku = 8 and Pu = 2 pi / sqrt(3). This is the textbook
    // example for the critical-proportion method, and it is checkable by hand.
    control::Matrix a = control::Matrix::Zero(3, 3);
    a << -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0;
    control::Matrix b(3, 1);
    b << 0.0, 0.0, 1.0;
    control::Matrix c(1, 3);
    c << 1.0, 0.0, 0.0;
    const control::LtiSystem system = control::LtiSystem::Continuous(a, b, c);

    const control::PidDesign design =
        control::TuneZieglerNicholsFromModel(system, control::PidForm::kPid);
    CHECK_TRUE(design.has_ultimate_values);
    CHECK_NEAR(design.ultimate_gain, 8.0, 1e-3);
    CHECK_NEAR(design.ultimate_period, 2.0 * M_PI / std::sqrt(3.0), 1e-3);
    // With the plant known, the poles and margins are reported rather than left
    // to be discovered.
    CHECK_EQ(design.closed_loop_poles.size(),
             4);  // 3 plant poles + the integrator
    CHECK_TRUE(design.bandwidth > 0.0);
    CHECK_TRUE(design.phase_margin_degrees > 0.0);

    // A first-order lag has no ultimate gain: its phase never reaches -180
    // degrees. The method must refuse rather than invent a gain.
    CHECK_THROWS(control::TuneZieglerNicholsFromModel(FirstOrder(1.0, 1.0),
                                                      control::PidForm::kPid));
}

TEST(Pid, PolePlacementPlacesTheClosedLoopExactly) {
    // G(s) = 1/(s^2 + 3s + 2) with PID: the closed-loop polynomial is
    //   s(s^2 + 3s + 2) + (Kd s^2 + Kp s + Ki)
    // and it is matched against (s^2 + 2 zeta wn s + wn^2)(s + p0).
    const control::LtiSystem system = SecondOrder(2.0, 3.0, 1.0);
    const double zeta = 0.7;
    const double wn = 2.0;
    control::PidTuningOptions options;
    options.integrator_pole_ratio = 4.0;  // p0 = 8
    const control::PidDesign design = control::TunePolePlacement(
        system, control::PidForm::kPid, zeta, wn, options);
    CHECK_TRUE(design.ok);

    // Verify by the coefficients themselves: the closed-loop polynomial must be
    // (s^2 + 2*0.7*2 s + 4)(s + 8) = s^3 + 10.8 s^2 + 26.4 s + 32.
    const double p0 = 4.0 * wn;
    const double expected_s2 = 2.0 * zeta * wn + p0;
    const double expected_s1 = wn * wn + 2.0 * zeta * wn * p0;
    const double expected_s0 = wn * wn * p0;
    // From the design: a1 + b Kd, a0 + b Kp, b Ki with a1 = 3, a0 = 2, b = 1.
    CHECK_NEAR(3.0 + design.gains.kd, expected_s2, 1e-9);
    CHECK_NEAR(2.0 + design.gains.kp, expected_s1, 1e-9);
    CHECK_NEAR(design.gains.ki, expected_s0, 1e-9);

    // The reported poles are the roots of that polynomial, so two of them sit
    // at the requested second-order pair.
    int matching = 0;
    for (Eigen::Index i = 0; i < design.closed_loop_poles.size(); ++i) {
        const std::complex<double> pole = design.closed_loop_poles(i);
        if (std::abs(pole.real() + zeta * wn) < 1e-6 &&
            std::abs(std::abs(pole.imag()) -
                     wn * std::sqrt(1.0 - zeta * zeta)) < 1e-6) {
            ++matching;
        }
        // Every pole must be stable for the design to be usable.
        CHECK_TRUE(pole.real() < 0.0);
    }
    CHECK_EQ(matching, 2);

    // PI on a first-order plant: G = 2/(s+1), the closed loop is
    // s^2 + (1 + 2 Kp) s + 2 Ki, matched to s^2 + 2 zeta wn s + wn^2.
    const control::PidDesign pi_design = control::TunePolePlacement(
        FirstOrder(1.0, 2.0), control::PidForm::kPi, 0.8, 3.0);
    CHECK_NEAR(pi_design.gains.kp, (2.0 * 0.8 * 3.0 - 1.0) / 2.0, 1e-9);
    CHECK_NEAR(pi_design.gains.ki, 9.0 / 2.0, 1e-9);
    CHECK_EQ(pi_design.closed_loop_poles.size(), 2);
    CHECK_NEAR(pi_design.closed_loop_poles(0).real(), -0.8 * 3.0, 1e-6);
}

TEST(Pid, PolePlacementRefusesShapesItCannotMatch) {
    // A first-order plant with PID: the derivative gain is redundant and the
    // matching is underdetermined. Refusing beats returning one arbitrary
    // member of the family.
    CHECK_THROWS(control::TunePolePlacement(FirstOrder(1.0, 1.0),
                                            control::PidForm::kPid, 0.7, 2.0));
    // A third-order plant is beyond coefficient matching; the message points at
    // the functions that handle it properly.
    control::Matrix a = control::Matrix::Zero(3, 3);
    a << -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0;
    control::Matrix b(3, 1);
    b << 0.0, 0.0, 1.0;
    control::Matrix c(1, 3);
    c << 1.0, 0.0, 0.0;
    CHECK_THROWS(
        control::TunePolePlacement(control::LtiSystem::Continuous(a, b, c),
                                   control::PidForm::kPid, 0.7, 2.0));
    // Nonsense specifications are rejected before any arithmetic.
    CHECK_THROWS(control::TunePolePlacement(SecondOrder(2.0, 3.0, 1.0),
                                            control::PidForm::kPid, 1.5, 2.0));
    CHECK_THROWS(control::TunePolePlacement(SecondOrder(2.0, 3.0, 1.0),
                                            control::PidForm::kPid, 0.7, -1.0));
}

TEST(Pid, MarginsAreReportedForAKnownLoop) {
    // G(s) = 1/(s+1)^3: its phase crosses -180 degrees at w = sqrt(3) with
    // |G| = 1/8, so the gain margin is finite and checkable by hand. (With two
    // poles the phase only approaches -180 degrees asymptotically, so there is
    // no gain margin at all - and a loop gain below 1 never crosses 0 dB
    // either, which is why the gains below start at 2.)
    control::Matrix a = control::Matrix::Zero(3, 3);
    a << -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0;
    control::Matrix b(3, 1);
    b << 0.0, 0.0, 1.0;
    control::Matrix c(1, 3);
    c << 1.0, 0.0, 0.0;
    const control::LtiSystem system = control::LtiSystem::Continuous(a, b, c);

    control::PidGains low;
    low.kp = 2.0;
    const control::OpenLoopMargins low_margins =
        control::ComputeMargins(system, low);
    CHECK_TRUE(low_margins.has_crossover);
    CHECK_TRUE(low_margins.has_phase_crossover);
    CHECK_TRUE(low_margins.phase_margin_degrees > 0.0);
    // At the phase crossover |L| = kp/8, so the gain margin is 20 log10(8/kp):
    // 12.04 dB at kp = 2, and 4.08 dB at kp = 5.
    CHECK_NEAR(low_margins.gain_margin_db, 20.0 * std::log10(8.0 / 2.0), 1e-2);
    CHECK_NEAR(low_margins.phase_crossover_frequency, std::sqrt(3.0), 1e-2);

    control::PidGains high;
    high.kp = 5.0;
    const control::OpenLoopMargins high_margins =
        control::ComputeMargins(system, high);
    CHECK_NEAR(high_margins.gain_margin_db, 20.0 * std::log10(8.0 / 5.0), 1e-2);
    // More gain means more bandwidth, less phase margin and less gain margin.
    CHECK_TRUE(high_margins.crossover_frequency >
               low_margins.crossover_frequency);
    CHECK_TRUE(high_margins.phase_margin_degrees <
               low_margins.phase_margin_degrees);
    CHECK_TRUE(high_margins.gain_margin_db < low_margins.gain_margin_db);

    // What "no gain margin" looks like: a two-pole plant never reaches -180
    // degrees, and the margin is reported as absent rather than as a number.
    control::Matrix a2(2, 2);
    a2 << 0.0, 1.0, -1.0, -2.0;
    control::Matrix b2(2, 1);
    b2 << 0.0, 1.0;
    control::Matrix c2(1, 2);
    c2 << 1.0, 0.0;
    control::PidGains tiny;
    tiny.kp = 0.5;  // DC loop gain 0.5: never crosses 0 dB either
    const control::OpenLoopMargins absent = control::ComputeMargins(
        control::LtiSystem::Continuous(a2, b2, c2), tiny);
    CHECK_FALSE(absent.has_crossover);
    CHECK_FALSE(absent.has_phase_crossover);
}

TEST(Pid, ControllerDoesNotKickTheDerivativeOnASetpointStep) {
    // The derivative acts on the measurement, so a setpoint step must not
    // produce a derivative spike. With Kd large and the plant at rest, the
    // first command is Kp*e and nothing else: the "derivative kick" would show
    // up immediately.
    control::PidConfig config;
    config.gains.kp = 1.0;
    config.gains.ki = 0.0;
    config.gains.kd = 100.0;  // a deliberately huge derivative gain
    config.sample_time = 0.01;
    control::PidController controller(config);
    const double command = controller.Update(0.0, 1.0);
    CHECK_NEAR(command, 1.0, 1e-12);  // Kp*e only, no derivative term

    // A measurement that moves *does* produce derivative action, and with Kd =
    // 100 it is large: that is precisely why the derivative is filtered and why
    // it acts on the measurement. The claim being tested is that the setpoint
    // step produced none, not that the derivative is small.
    const double next = controller.Update(1.0, 1.0);
    CHECK_TRUE(next < 0.0);  // the measurement rose, so the derivative opposes
    CHECK_TRUE(std::abs(next) > 1.0);
}

TEST(Pid, IntegratorIsHeldWhileTheOutputIsSaturated) {
    control::PidConfig config;
    config.gains.kp = 1.0;
    config.gains.ki = 10.0;
    config.sample_time = 0.01;
    config.output_min = -1.0;
    config.output_max = 1.0;
    config.integral_min = -1.0;
    config.integral_max = 1.0;
    control::PidController controller(config);

    // Hold the measurement at zero with a large setpoint: the output saturates
    // immediately and the integrator must stop growing.
    for (int tick = 0; tick < 500; ++tick) {
        controller.Update(0.0, 100.0);
    }
    CHECK_TRUE(controller.saturated());
    // The integral term is bounded by the integral limit, which is the second
    // line of defence against wind-up.
    CHECK_TRUE(std::abs(controller.integral_term()) <= 1.0 + 1e-9);
    // Without the hold the integral term would be 10*0.01*100*500 = 5000.
    CHECK_TRUE(std::abs(controller.integral_term()) < 2.0);

    // Reversing the error must unwind it rather than being blocked.
    controller.Update(0.0, -100.0);
    CHECK_TRUE(controller.integral_term() < 1.0);
}

TEST(Pid, RateLimitAndBumplessTransfer) {
    control::PidConfig config;
    config.gains.kp = 10.0;
    config.gains.ki = 0.0;
    config.sample_time = 0.01;
    config.rate_limit = true;
    config.rate_max = 10.0;  // units per second, so 0.1 per tick
    control::PidController controller(config);
    const double first = controller.Update(0.0, 100.0);  // Kp*e = 1000, clipped
    CHECK_TRUE(controller.rate_limited());
    CHECK_NEAR(first, 0.1, 1e-12);
    const double second = controller.Update(0.0, 100.0);
    CHECK_NEAR(second, 0.2, 1e-12);

    // Bumpless transfer: adopting the manual command means the next automatic
    // tick starts from where the actuator already is.
    control::PidController tracking(config);
    tracking.TrackManualOutput(5.0);
    CHECK_NEAR(tracking.integral_term(), 5.0, 0.0);
    CHECK_NEAR(tracking.Update(0.0, 0.0), 5.0, 1e-12);
}

TEST(Pid, NonFiniteMeasurementHoldsTheLastCommand) {
    control::PidConfig config;
    config.gains.kp = 2.0;
    config.gains.ki = 1.0;
    config.sample_time = 0.01;
    control::PidController controller(config);
    const double good = controller.Update(0.5, 1.0);
    CHECK_TRUE(std::isfinite(good));
    const double held =
        controller.Update(std::numeric_limits<double>::quiet_NaN(), 1.0);
    CHECK_NEAR(held, good, 0.0);
    CHECK_TRUE(std::isfinite(held));
}

TEST(Pid, ClosedLoopSimulationReportsTheNumbersAReviewAsksFor) {
    // The pole-placement design on a known plant must actually settle, and the
    // metrics must agree with the specification that produced it: 10 percent
    // overshoot and a settling time near the request.
    const control::LtiSystem system = SecondOrder(2.0, 3.0, 1.0);
    control::PidSpecifications specifications;
    specifications.overshoot_percent = 10.0;
    specifications.settling_time = 1.0;
    double zeta = 0.0;
    double wn = 0.0;
    control::SpecificationsToPoles(specifications, &zeta, &wn);
    const control::PidDesign design =
        control::TunePolePlacement(system, control::PidForm::kPid, zeta, wn);

    control::PidConfig config;
    config.gains = design.gains;
    config.sample_time = 0.001;
    config.output_min =
        -1e6;  // effectively unlimited, so the design is measured
    config.output_max = 1e6;
    const control::PidStepMetrics metrics =
        control::SimulatePidStep(system, config, 1.0, 5.0);

    CHECK_TRUE(metrics.settled);
    // The design predicts its own achieved response, and that prediction has to
    // match an independent simulation of the same design: this is the claim
    // that makes the reported overshoot trustworthy.
    CHECK_TRUE(design.has_predicted_step);
    CHECK_NEAR(design.predicted_step.overshoot_percent,
               metrics.overshoot_percent, 1e-6);
    CHECK_NEAR(design.predicted_step.settling_time, metrics.settling_time,
               1e-6);
    // And the achieved overshoot is well above the 10 percent the damping ratio
    // suggests, which is exactly why the design reports it and warns: pole
    // placement places poles, and the controller's zeros are not placed by it.
    CHECK_TRUE(metrics.overshoot_percent > 15.0);
    CHECK_TRUE(design.warnings.size() > 0);
    // The settling time, unlike the overshoot, does land near the request.
    CHECK_TRUE(metrics.settling_time < 1.5);
    CHECK_NEAR(metrics.steady_state_error, 0.0, 1e-6);
    CHECK_EQ(metrics.saturated_ticks, 0);
}

TEST(Pid, SaturationIsVisibleInTheSimulationAndSlowsTheResponse) {
    const control::LtiSystem system = SecondOrder(2.0, 3.0, 1.0);
    const control::PidDesign design =
        control::TunePolePlacement(system, control::PidForm::kPid, 0.7, 2.0);

    control::PidConfig limited;
    limited.gains = design.gains;
    limited.sample_time = 0.001;
    limited.output_min = -0.5;  // a tight actuator: the design will hit it
    limited.output_max = 0.5;
    const control::PidStepMetrics metrics =
        control::SimulatePidStep(system, limited, 1.0, 5.0);
    // A 1.0 reference on a plant with DC gain 0.5 needs u = 2 in steady state,
    // so an actuator limited to 0.5 makes the reference unreachable - and the
    // simulation says so through the saturation count rather than silently
    // reporting a large steady-state error with no explanation.
    CHECK_TRUE(metrics.saturated_ticks > 0);
    CHECK_TRUE(std::abs(metrics.steady_state_error) > 0.1);
}
