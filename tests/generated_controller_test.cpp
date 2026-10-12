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

// The cross-check gate: the *generated* controller is compiled and run against
// an independent implementation over the same input sequence.
//
// Why this exists: the emitted code is hand-unrolled loops over hand-indexed
// arrays. A transposed index, an inverted sign or a bound written as < instead
// of <= compiles cleanly and produces a controller that is almost right - which
// is the worst kind, because it survives review and fails on hardware.
// Comparing against a reference written with Eigen expressions (a different
// arithmetic order, a different implementation of the same equations) is what
// makes the generated file trustworthy.
//
// The reference below is written from the design's equations, not by copying
// the emitted code: both implement
//
//   x_hat = measurement                       (full state feedback)
//   y     = C x_hat
//   e     = setpoint - y
//   z    += Ts e                              (held while saturated,
//   anti-windup) u     = -K x_hat - Ki z u     = clamp(u, u_min, u_max)
//
// with the gains taken from the same design the generator used.

#include <Eigen/Core>
#include <Eigen/LU>
#include <cmath>
#include <cstdio>
#include <vector>

#include "PositioningController.h"  // the generated header, compiled as-is
#include "control/codegen.h"
#include "control/design.h"
#include "control/discretize.h"
#include "control/lti.h"
#include "positioning_design.h"
#include "tests/testing.h"

namespace {

// Independent reference: same equations, Eigen arithmetic, separately written.
class ReferenceController {
 public:
    ReferenceController(const control::LtiSystem& system,
                        const control::Matrix& gain)
        : a_(system.a()),
          b_(system.b()),
          c_(system.c()),
          k_(gain.leftCols(system.state_dimension())),
          ki_(gain.rightCols(1)),
          ts_(system.sample_time()) {
        x_hat_ = control::Vector::Zero(system.state_dimension());
        z_ = 0.0;
        u_prev_ = 0.0;
    }

    double Update(const control::Vector& measurement, double setpoint) {
        x_hat_ = measurement;
        const double output = (c_ * x_hat_)(0, 0);
        const double error = setpoint - output;

        const bool winding_up =
            saturated_ &&
            ((u_prev_ > positioning::ForceLimit() && error > 0.0) ||
             (u_prev_ < -positioning::ForceLimit() && error < 0.0));
        if (!winding_up) {
            z_ += ts_ * error;
        }

        double u = -(k_ * x_hat_)(0, 0) - (ki_ * z_)(0, 0);
        const double clipped =
            u < -positioning::ForceLimit()
                ? -positioning::ForceLimit()
                : (u > positioning::ForceLimit() ? positioning::ForceLimit()
                                                 : u);
        saturated_ = (clipped != u);
        u = clipped;
        u_prev_ = u;
        return u;
    }

    double integral_state() const { return z_; }

 private:
    control::Matrix a_;
    control::Matrix b_;
    control::Matrix c_;
    control::Matrix k_;
    control::Matrix ki_;
    double ts_;
    control::Vector x_hat_;
    double z_;
    double u_prev_;
    bool saturated_ = false;
};

// The plant, stepped with the same zero-order hold the design assumed.
control::Vector StepPlant(const control::LtiSystem& system,
                          const control::Vector& state, double input) {
    return system.a() * state + system.b() * input;
}

}  // namespace

TEST(GeneratedController, MatchesTheReferenceOverThreeThousandTicks) {
    const control::LtiSystem discrete = positioning::DiscretePlant();
    const control::AugmentedDesign design = positioning::Design();
    const control::Matrix gain = design.design.gain;

    positioning::PositioningController generated;
    generated.Reset();
    ReferenceController reference(discrete, gain);

    const int ticks = 3000;
    control::Vector state = control::Vector::Zero(discrete.state_dimension());
    double worst_absolute = 0.0;
    double scale = 0.0;
    int saturated_ticks = 0;

    for (int tick = 0; tick < ticks; ++tick) {
        // A sequence that covers what actually stresses a controller: a step, a
        // reversal, and a large command that drives the actuator into its
        // limit.
        double setpoint = 0.0;
        if (tick >= 100) {
            setpoint = 1.0;
        }
        if (tick >= 1000) {
            setpoint = -1.0;
        }
        if (tick >= 1500 && tick < 1530) {
            setpoint = 50.0;  // demands more force than the actuator has
        }
        if (tick >= 2000) {
            setpoint = 0.25;
        }

        const double u_generated = [&]() {
            const double measurement[2] = {state(0), state(1)};
            double command[1] = {0.0};
            generated.Update(measurement, setpoint, command);
            return command[0];
        }();
        const double u_reference = reference.Update(state, setpoint);

        worst_absolute =
            std::max(worst_absolute, std::abs(u_generated - u_reference));
        scale = std::max(scale, std::abs(u_reference));
        if (generated.saturated() ||
            std::abs(u_reference) >= positioning::ForceLimit()) {
            ++saturated_ticks;
        }
        CHECK_EQ(generated.error(), 0);

        state = StepPlant(discrete, state, u_generated);
    }

    // double, so the agreement has to be at round-off level. A tolerance of
    // 1e-9 relative is what the code-generation protocol asks for.
    std::printf(
        "  ticks=%d, saturation active on %d ticks, |u|max=%.6f, "
        "max deviation=%.3e (relative %.3e)\n",
        ticks, saturated_ticks, scale, worst_absolute,
        scale > 0.0 ? worst_absolute / scale : 0.0);
    CHECK_TRUE(saturated_ticks >
               0);  // the sequence has to actually exercise the limit
    CHECK_TRUE(worst_absolute <= 1e-12 + 1e-9 * scale);
}

TEST(GeneratedController, IntegratorStateMatchesTheReferenceToo) {
    // Comparing only the command would miss a divergence in the integrator that
    // has not yet shown up in the output; the internal state is compared as
    // well.
    const control::LtiSystem discrete = positioning::DiscretePlant();
    const control::AugmentedDesign design = positioning::Design();
    const control::Matrix gain = design.design.gain;

    positioning::PositioningController generated;
    generated.Reset();
    ReferenceController reference(discrete, gain);

    control::Vector state = control::Vector::Zero(discrete.state_dimension());
    for (int tick = 0; tick < 800; ++tick) {
        const double setpoint = tick < 50 ? 0.0 : 1.0;
        const double measurement[2] = {state(0), state(1)};
        double command[1] = {0.0};
        generated.Update(measurement, setpoint, command);
        reference.Update(state, setpoint);
        state = StepPlant(discrete, state, command[0]);
    }
    const control::Vector estimate =
        control::Vector::Map(generated.state_estimate().data(), 2);
    CHECK_NEAR(generated.integral_state(), reference.integral_state(), 1e-12);
    // Full state feedback: the estimate is the measurement, not a filtered
    // copy.
    CHECK_NEAR(estimate(0), state(0), 1e-15);
    CHECK_TRUE(std::abs(generated.integral_state()) >
               1e-6);  // it actually integrated
}

TEST(GeneratedController, HoldsTheLastCommandOnANonFiniteMeasurement) {
    // A sensor that returns NaN must not reach the actuator. The controller
    // raises its error flag and repeats the previous command.
    positioning::PositioningController generated;
    generated.Reset();
    const double good_measurement[2] = {0.5, 0.0};
    double command[1] = {0.0};
    generated.Update(good_measurement, 1.0, command);
    const double safe = command[0];
    CHECK_EQ(generated.error(), 0);
    CHECK_TRUE(std::abs(safe) > 0.0);

    const double bad_measurement[2] = {std::numeric_limits<double>::quiet_NaN(),
                                       0.0};
    double after[1] = {0.0};
    generated.Update(bad_measurement, 1.0, after);
    CHECK_EQ(generated.error(), 1);
    CHECK_NEAR(after[0], safe, 0.0);
    CHECK_TRUE(std::isfinite(after[0]));

    // A null pointer is a programming error, reported separately.
    generated.Update(nullptr, 1.0, after);
    CHECK_EQ(generated.error(), 2);

    // Recovery: a good measurement clears the flag.
    generated.Update(good_measurement, 1.0, after);
    CHECK_EQ(generated.error(), 0);
}
