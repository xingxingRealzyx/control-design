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

#include "control/mpc.h"

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>

#include "control/design.h"
#include "control/discretize.h"
#include "control/lti.h"
#include "tests/testing.h"

namespace {

control::LtiSystem DiscreteSecondOrder() {
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -2.0, -3.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    return control::Discretize(control::LtiSystem::Continuous(a, b, c), 0.1);
}

control::MpcOptions Options(int horizon) {
    control::MpcOptions options;
    options.horizon = horizon;
    options.output_weight = control::Matrix::Identity(1, 1);
    options.input_weight = control::Matrix::Constant(1, 1, 0.5);
    return options;
}

// Steps the discrete plant forward one tick.
control::Vector Advance(const control::LtiSystem& system,
                        const control::Vector& state, double command) {
    return system.a() * state + system.b() * command;
}

}  // namespace

TEST(Mpc, PredictionMatricesMatchAStepByStepSimulation) {
    // Phi and Gamma are built by matrix powers; simulating the augmented model
    // directly is an independent way to get the same prediction, so the two
    // must agree. A transposed or off-by-one prediction matrix still produces a
    // stable controller, which is why this is checked rather than assumed.
    const control::LtiSystem discrete = DiscreteSecondOrder();
    const control::MpcOptions options = Options(4);
    const control::MpcDesign design = control::BuildMpc(discrete, options);

    const int n = design.state_dimension;
    const int m = design.input_dimension;
    const int p = design.output_dimension;
    control::Vector augmented(n + m);
    augmented << 1.0, -0.5, 0.2;

    control::Vector increments(options.horizon * m);
    for (int i = 0; i < increments.size(); ++i) {
        increments(i) = 0.1 * (i + 1);
    }
    const control::Vector predicted =
        design.phi * augmented + design.gamma * increments;

    // Simulate: xi+ = Aa xi + Ba du, and record the output after each step.
    control::Vector xi = augmented;
    for (int step = 0; step < options.horizon; ++step) {
        xi = design.augmented.a() * xi +
             design.augmented.b() * increments.segment(step * m, m);
        const control::Vector y = design.output_matrix * xi;
        for (int channel = 0; channel < p; ++channel) {
            CHECK_NEAR(predicted(step * p + channel), y(channel), 1e-12);
        }
    }
}

TEST(Mpc, UnconstrainedOptimumIsExactlyTheLqrGain) {
    // The theorem that makes MPC trustworthy without a simulation: with the
    // terminal cost taken from the DARE solution of the *augmented* problem and
    // the stage cost covering the steps the terminal cost does not, the
    // unconstrained optimum equals the LQR gain for any horizon. Checked at
    // three horizons, because a horizon-independent answer is the whole point -
    // an implementation that forgot the terminal cost would still be stable and
    // would fail this.
    const control::LtiSystem discrete = DiscreteSecondOrder();
    for (int horizon : {1, 5, 20}) {
        const control::MpcOptions options = Options(horizon);
        const control::MpcDesign design = control::BuildMpc(discrete, options);

        // The equivalent LQR problem is on the augmented model, because the
        // cost penalises the input increment rather than the input.
        const control::Matrix augmented_q = design.output_matrix.transpose() *
                                            options.output_weight *
                                            design.output_matrix;
        const control::DesignResult lqr =
            control::DiscreteLinearQuadraticRegulator(
                design.augmented, augmented_q, options.input_weight);

        for (int trial = 0; trial < 6; ++trial) {
            control::Vector augmented(3);
            augmented << 1.0 + 0.7 * trial, -0.5 + 0.3 * trial, -0.3 * trial;
            control::Vector reference(1);
            reference << 0.0;
            const control::Vector increments =
                design.UnconstrainedSolution(augmented, {reference});
            const double expected = -(lqr.gain * augmented)(0, 0);
            CHECK_NEAR(increments(0), expected, 1e-7);
        }
    }
}

TEST(Mpc, TerminalCostIsWhatMakesTheEquivalenceHold) {
    // Without the terminal cost the controller is still stable, but it is no
    // longer the LQR solution: the difference is the reason the terminal cost
    // exists, and it is invisible in a step response.
    const control::LtiSystem discrete = DiscreteSecondOrder();
    control::MpcOptions options = Options(3);
    options.use_terminal_cost = false;
    const control::MpcDesign design = control::BuildMpc(discrete, options);

    control::Matrix augmented_q = design.output_matrix.transpose() *
                                  options.output_weight * design.output_matrix;
    const control::DesignResult lqr = control::DiscreteLinearQuadraticRegulator(
        design.augmented, augmented_q, options.input_weight);
    control::Vector augmented(3);
    augmented << 1.0, 0.5, 0.0;
    control::Vector reference(1);
    reference << 0.0;
    const control::Vector increments =
        design.UnconstrainedSolution(augmented, {reference});
    const double expected = -(lqr.gain * augmented)(0, 0);
    // Not equal: a short horizon without a terminal cost is myopic.
    CHECK_TRUE(std::abs(increments(0) - expected) > 1e-3);
}

TEST(Mpc, InputIncrementLimitIsEnforcedExactly) {
    // The slew-rate limit is enforced inside the QP, so it holds on every tick
    // including the first, when the demand is largest.
    const control::LtiSystem discrete = DiscreteSecondOrder();
    control::MpcOptions options = Options(8);
    options.delta_input_min = control::Matrix::Constant(1, 1, -0.5);
    options.delta_input_max = control::Matrix::Constant(1, 1, 0.5);
    control::MpcController controller(discrete, options);

    control::Vector state(2);
    state << 5.0, 0.0;  // far from the reference, so the demand is large
    control::Vector command(1);
    double previous = 0.0;
    double worst_step = 0.0;
    int worst_iterations = 0;
    bool all_converged = true;
    for (int tick = 0; tick < 200; ++tick) {
        const int iterations = controller.Update(state, 0.0, &command);
        worst_iterations = std::max(worst_iterations, iterations);
        all_converged = all_converged && controller.solver_converged();
        worst_step = std::max(worst_step, std::abs(command(0) - previous));
        previous = command(0);
        state = Advance(discrete, state, command(0));
    }
    CHECK_NEAR(worst_step, 0.5, 1e-9);
    CHECK_TRUE(all_converged);
    // The iteration count is bounded by the option, which is what makes the
    // per-tick worst case knowable rather than hoped for.
    CHECK_TRUE(worst_iterations <= options.max_iterations);
    // And the loop actually does its job.
    CHECK_NEAR(state(0), 0.0, 1e-6);
    CHECK_NEAR(state(1), 0.0, 1e-6);
}

TEST(Mpc, AbsoluteInputLimitIsNeverExceeded) {
    const control::LtiSystem discrete = DiscreteSecondOrder();
    control::MpcOptions options = Options(8);
    options.input_min = control::Matrix::Constant(1, 1, -1.0);
    options.input_max = control::Matrix::Constant(1, 1, 1.0);
    control::MpcController controller(discrete, options);

    control::Vector state(2);
    state << 5.0, 0.0;
    control::Vector command(1);
    double peak = 0.0;
    int active = 0;
    for (int tick = 0; tick < 200; ++tick) {
        controller.Update(state, 0.0, &command);
        peak = std::max(peak, std::abs(command(0)));
        if (controller.constraints_active()) {
            ++active;
        }
        state = Advance(discrete, state, command(0));
    }
    // The limit is a guarantee about the command that leaves the controller,
    // not a hope about the QP's optimum.
    CHECK_TRUE(peak <= 1.0 + 1e-12);
    CHECK_TRUE(active > 0);
    CHECK_NEAR(state(0), 0.0, 1e-6);
}

TEST(Mpc, UnconstrainedRunMatchesTheLqrClosedLoop) {
    // With nothing active, the whole closed loop must reproduce the LQR closed
    // loop tick by tick - not just the first gain, but the trajectory.
    const control::LtiSystem discrete = DiscreteSecondOrder();
    const control::MpcOptions options = Options(10);
    const control::MpcDesign design = control::BuildMpc(discrete, options);
    const control::Matrix augmented_q = design.output_matrix.transpose() *
                                        options.output_weight *
                                        design.output_matrix;
    const control::DesignResult lqr = control::DiscreteLinearQuadraticRegulator(
        design.augmented, augmented_q, options.input_weight);

    control::MpcController controller(discrete, options);
    control::Vector mpc_state(2);
    mpc_state << 1.0, 0.0;
    control::Vector lqr_state = mpc_state;
    double lqr_command = 0.0;
    control::Vector command(1);
    double worst = 0.0;
    // 300 ticks, not 100: the augmented closed loop's spectral radius is 0.873, so
    // a hundred ticks leaves about 1e-6 of the initial condition behind and the
    // final-value check below would be measuring the horizon rather than the law.
    for (int tick = 0; tick < 300; ++tick) {
        controller.Update(mpc_state, 0.0, &command);
        // The LQR law acts on the augmented state, which needs the previous
        // command the MPC applied.
        control::Vector augmented(3);
        augmented << lqr_state, lqr_command;
        lqr_command = -(lqr.gain * augmented)(0, 0) + lqr_command;
        mpc_state = Advance(discrete, mpc_state, command(0));
        lqr_state = Advance(discrete, lqr_state, lqr_command);
        worst = std::max(worst, std::abs(mpc_state(0) - lqr_state(0)));
    }
    // The per-step agreement is at the level the DARE solution is accurate to
    // (about 5e-9); over a hundred ticks of a stable closed loop it accumulates to
    // roughly that divided by one minus the spectral radius, which is a few times
    // 1e-8. The tolerance is set from the measured value rather than from a wish,
    // and the value is printed so a regression is visible rather than merely fatal.
    std::printf("  MPC vs LQR closed loop over 300 ticks: worst deviation %.3e\n", worst);
    CHECK_TRUE(worst < 1e-6);
    CHECK_NEAR(mpc_state(0), 0.0, 1e-9);
}

TEST(Mpc, ReferenceTrackingHasNoSteadyStateOffset) {
    // The velocity form integrates by construction: the increments stop when
    // the output reaches the reference, so a constant reference is tracked
    // exactly. This is the property the formulation was chosen for.
    const control::LtiSystem discrete = DiscreteSecondOrder();
    const control::MpcOptions options = Options(10);
    control::MpcController controller(discrete, options);
    control::Vector state = control::Vector::Zero(2);
    control::Vector command(1);
    for (int tick = 0; tick < 400; ++tick) {
        controller.Update(state, 2.0, &command);
        state = Advance(discrete, state, command(0));
    }
    // The plant's DC gain is 0.5, so the output reaches 2.0 exactly only
    // because the increment form removes the offset.
    CHECK_NEAR(state(0), 2.0, 1e-6);
    CHECK_NEAR(controller.last_command()(0), 4.0, 1e-3);
}

TEST(Mpc, NonFiniteStateHoldsTheLastCommand) {
    const control::LtiSystem discrete = DiscreteSecondOrder();
    control::MpcController controller(discrete, Options(5));
    control::Vector state(2);
    state << 0.5, 0.0;
    control::Vector command(1);
    controller.Update(state, 1.0, &command);
    const double safe = command(0);
    control::Vector bad(2);
    bad << std::numeric_limits<double>::quiet_NaN(), 0.0;
    controller.Update(bad, 1.0, &command);
    CHECK_NEAR(command(0), safe, 0.0);
    CHECK_TRUE(std::isfinite(command(0)));
}

TEST(Mpc, RefusesModelsAndWeightsItCannotUse) {
    // A continuous model cannot be used to predict: the prediction would be of
    // the wrong plant.
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -2.0, -3.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    CHECK_THROWS(
        control::BuildMpc(control::LtiSystem::Continuous(a, b, c), Options(5)));

    // Weight matrices of the wrong size.
    control::MpcOptions wrong_output = Options(5);
    wrong_output.output_weight = control::Matrix::Identity(2, 2);
    CHECK_THROWS(control::BuildMpc(DiscreteSecondOrder(), wrong_output));

    control::MpcOptions wrong_input = Options(5);
    wrong_input.input_weight = control::Matrix::Identity(2, 2);
    CHECK_THROWS(control::BuildMpc(DiscreteSecondOrder(), wrong_input));

    // A horizon of zero steps is not a controller.
    control::MpcOptions no_horizon = Options(0);
    CHECK_THROWS(control::BuildMpc(DiscreteSecondOrder(), no_horizon));

    // A mismatched state or reference size at run time.
    control::MpcController controller(DiscreteSecondOrder(), Options(5));
    control::Vector command(1);
    control::Vector wrong_state = control::Vector::Zero(3);
    CHECK_THROWS(controller.Update(wrong_state, 0.0, &command));
    control::Vector state = control::Vector::Zero(2);
    control::Vector wrong_reference = control::Vector::Zero(2);
    CHECK_THROWS(controller.Update(state, wrong_reference, &command));
}
