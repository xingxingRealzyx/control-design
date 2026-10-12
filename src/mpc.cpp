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
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace control {
namespace {

// Euclidean projection onto a box, given as (lower, upper) vectors that may
// have infinite entries.
Vector ProjectBox(const Vector& point, const Vector& lower,
                  const Vector& upper) {
    Vector projected = point;
    for (Eigen::Index i = 0; i < point.size(); ++i) {
        if (projected(i) < lower(i)) {
            projected(i) = lower(i);
        }
        if (projected(i) > upper(i)) {
            projected(i) = upper(i);
        }
    }
    return projected;
}

}  // namespace

Vector MpcDesign::UnconstrainedSolution(
    const Vector& augmented_state, const std::vector<Vector>& reference) const {
    if (augmented_state.size() != phi.cols()) {
        throw std::invalid_argument(
            "MpcDesign::UnconstrainedSolution: the augmented state has the "
            "wrong size "
            "(it is [x; u_previous], with n + m entries)");
    }
    const int steps = horizon;
    const int outputs = output_dimension;
    Vector stacked_reference(steps * outputs);
    for (int step = 0; step < steps; ++step) {
        if (reference.empty()) {
            stacked_reference.segment(step * outputs, outputs).setZero();
            continue;
        }
        const std::size_t index = std::min<std::size_t>(
            static_cast<std::size_t>(step), reference.size() - 1);
        if (reference[index].size() != outputs) {
            throw std::invalid_argument(
                "MpcDesign::UnconstrainedSolution: a reference has the wrong "
                "size");
        }
        stacked_reference.segment(step * outputs, outputs) = reference[index];
    }
    // f = Gamma' Qbar (Phi xi - R), with Qbar block diagonal. The weight itself
    // is not stored here, so it is recovered from the relationship between the
    // stacked problem and the per-step one: the caller passes a reference only,
    // and H already encodes Qbar through Gamma. Keeping a copy of Qbar would be
    // the alternative; recovering f from H is not possible, so the weight is
    // applied by the caller's own Qbar through this helper's return of the raw
    // prediction error weighted by nothing.
    const Vector prediction_error = phi * augmented_state - stacked_reference;
    // Solve H dU = -Gamma' W (prediction error). W is block diagonal with the
    // output weight, which the design keeps in `output_weight_for_shape`.
    const int weighted_steps =
        terminal_cost.size() > 0 && terminal_cost.rows() == phi.cols()
            ? steps - 1
            : steps;
    Vector weighted = Vector::Zero(prediction_error.size());
    for (int step = 0; step < weighted_steps; ++step) {
        weighted.segment(step * outputs, outputs) =
            output_weight * prediction_error.segment(step * outputs, outputs);
    }
    Vector f = gamma.transpose() * weighted;
    if (gamma_terminal.size() > 0 && terminal_cost.size() > 0) {
        Vector target(output_dimension);
        target.setZero();
        if (!reference.empty() && reference.back().size() == output_dimension) {
            target = reference.back();
        }
        f += gamma_terminal.transpose() * terminal_cost *
             (phi_terminal * augmented_state - EquilibriumFor(target));
    }
    return -hessian.partialPivLu().solve(f);
}

MpcDesign BuildMpc(const LtiSystem& discrete_system,
                   const MpcOptions& options) {
    discrete_system.Validate("BuildMpc");
    if (!discrete_system.is_discrete()) {
        throw std::invalid_argument(
            "BuildMpc: the model must be discrete with the sample time the "
            "controller will run at; a prediction built from a continuous "
            "model is "
            "a prediction of the wrong plant");
    }
    const int n = discrete_system.state_dimension();
    const int m = discrete_system.input_dimension();
    const int p = discrete_system.output_dimension();
    if (options.horizon < 1) {
        throw std::invalid_argument(
            "BuildMpc: the horizon must be at least one step");
    }
    if (options.output_weight.rows() != p ||
        options.output_weight.cols() != p) {
        throw std::invalid_argument(
            "BuildMpc: the output weight must be p x p");
    }
    if (options.input_weight.rows() != m || options.input_weight.cols() != m) {
        throw std::invalid_argument("BuildMpc: the input weight must be m x m");
    }

    const int steps = options.horizon;
    MpcDesign design;
    design.horizon = steps;
    design.state_dimension = n;
    design.input_dimension = m;
    design.output_dimension = p;

    // Augmented model: xi = [x; u], xi+1 = Aa xi + Ba du.
    Matrix augmented_a = Matrix::Zero(n + m, n + m);
    augmented_a.topLeftCorner(n, n) = discrete_system.a();
    augmented_a.topRightCorner(n, m) = discrete_system.b();
    augmented_a.bottomRightCorner(m, m) = Matrix::Identity(m, m);
    Matrix augmented_b = Matrix::Zero(n + m, m);
    augmented_b.topRows(n) = discrete_system.b();
    augmented_b.bottomRows(m) = Matrix::Identity(m, m);
    // The augmented output is the plant output: y = [C 0] xi. The augmented
    // system therefore has n + m states and needs the output matrix widened to
    // match - passing the plant's own C leaves a system whose C has the wrong
    // number of columns, which every downstream call then rejects.
    Matrix output_matrix = Matrix::Zero(p, n + m);
    output_matrix.leftCols(n) = discrete_system.c();
    design.output_matrix = output_matrix;
    design.output_weight = options.output_weight;

    design.augmented = LtiSystem::Discrete(
        augmented_a, augmented_b, output_matrix, discrete_system.sample_time());

    // Terminal cost: the DARE solution of the *augmented* problem, whose
    // weights are the output weight applied to y = Cbar xi and the input weight
    // applied to du. It has to be the augmented DARE, not the plant's: the cost
    // being minimised penalises du, so the equivalent LQR problem is the one on
    // (Aa, Ba). Using the plant's DARE here produces a controller that is
    // stable and nearly right, and provably not the optimum - which is exactly
    // the kind of error that a theorem-shaped test catches and a simulation
    // does not.
    const Matrix augmented_q =
        output_matrix.transpose() * options.output_weight * output_matrix;
    Matrix terminal = Matrix::Zero(n + m, n + m);
    if (options.use_terminal_cost) {
        Matrix p_solution;
        if (!SolveDiscreteRiccati(augmented_a, augmented_b, augmented_q,
                                  options.input_weight, &p_solution)) {
            throw std::runtime_error(
                "BuildMpc: the terminal cost could not be computed (the "
                "discrete "
                "Riccati equation on the augmented model did not converge); "
                "pass "
                "use_terminal_cost = false to run without it, accepting a "
                "shorter "
                "effective horizon");
        }
        terminal = p_solution;
    }
    design.terminal_cost = terminal;

    // Prediction matrices. Each block row is the output after that many steps.
    design.phi = Matrix::Zero(steps * p, n + m);
    design.gamma = Matrix::Zero(steps * p, steps * m);
    Matrix power = Matrix::Identity(n + m, n + m);
    std::vector<Matrix> output_powers(static_cast<std::size_t>(steps));
    for (int step = 0; step < steps; ++step) {
        power = power * augmented_a;
        output_powers[static_cast<std::size_t>(step)] = output_matrix * power;
        design.phi.block(step * p, 0, p, n + m) =
            output_powers[static_cast<std::size_t>(step)];
    }
    for (int row = 0; row < steps; ++row) {
        for (int column = 0; column <= row; ++column) {
            // Contribution of du[column] to y[row+1]: Cbar Aa^(row-column) Ba.
            Matrix transition = Matrix::Identity(n + m, n + m);
            for (int k = 0; k < row - column; ++k) {
                transition = transition * augmented_a;
            }
            design.gamma.block(row * p, column * m, p, m) =
                output_matrix * transition * augmented_b;
        }
    }

    // Terminal prediction of the augmented state after N steps:
    //   xi[k+N] = Aa^N xi[k] + sum_j Aa^(N-1-j) Ba du[j].
    Matrix augmented_power = Matrix::Identity(n + m, n + m);
    for (int step = 0; step < steps; ++step) {
        augmented_power = augmented_power * augmented_a;
    }
    design.phi_terminal = augmented_power;
    design.gamma_terminal = Matrix::Zero(n + m, steps * m);
    for (int column = 0; column < steps; ++column) {
        Matrix transition = Matrix::Identity(n + m, n + m);
        for (int k = 0; k < steps - 1 - column; ++k) {
            transition = transition * augmented_a;
        }
        design.gamma_terminal.block(0, column * m, n + m, m) =
            transition * augmented_b;
    }

    // Weighted blocks: Qbar is block diagonal with Q on the diagonal.
    // Which predicted outputs carry the stage cost. With a terminal cost the
    // penalty applies to steps 1..N-1 and step N is priced by P: that is the
    // standard finite-horizon problem whose optimal first move is the LQR gain.
    // Penalising step N as well would double-count it (Qbar *and* P) and the
    // resulting policy is stable but not optimal - a difference of a few
    // percent that no simulation would flag and the equivalence test catches.
    const int weighted_steps = options.use_terminal_cost ? steps - 1 : steps;
    const Matrix& q = options.output_weight;
    const Matrix& r = options.input_weight;
    Matrix hessian = design.gamma.transpose() * [&]() {
        Matrix block = Matrix::Zero(steps * p, steps * p);
        for (int step = 0; step < weighted_steps; ++step) {
            block.block(step * p, step * p, p, p) = q;
        }
        return block;
    }() * design.gamma;
    for (int step = 0; step < steps; ++step) {
        hessian.block(step * m, step * m, m, m) += r;
    }
    if (options.use_terminal_cost) {
        hessian += design.gamma_terminal.transpose() * terminal *
                   design.gamma_terminal;
    }
    hessian += options.regularization * Matrix::Identity(steps * m, steps * m);
    design.hessian = 0.5 * (hessian + hessian.transpose());
    return design;
}

Vector MpcDesign::EquilibriumFor(const Vector& reference) const {
    // Solve [(A - I) B; C 0] [x; u] = [0; r] in the least-squares sense: the
    // augmented state that holds the reference. The plant is not required to be
    // square, so this is a least-squares problem and its residual says whether
    // the reference is reachable at all.
    const int n = state_dimension;
    const int m = input_dimension;
    const int p = output_dimension;
    Matrix left = Matrix::Zero(n + p, n + m);
    left.topLeftCorner(n, n) =
        augmented.a().topLeftCorner(n, n) - Matrix::Identity(n, n);
    left.topRightCorner(n, m) = augmented.a().topRightCorner(n, m);
    left.bottomLeftCorner(p, n) = output_matrix.leftCols(n);
    Vector right = Vector::Zero(n + p);
    right.tail(p) = reference;
    const Vector solution = left.colPivHouseholderQr().solve(right);
    if (!solution.allFinite()) {
        return Vector::Zero(n + m);
    }
    return solution;
}

MpcController::MpcController(const LtiSystem& discrete_system,
                             const MpcOptions& options)
    : design_(BuildMpc(discrete_system, options)), options_(options) {
    Reset();
}

void MpcController::Reset() {
    previous_command_ = Vector::Zero(design_.input_dimension);
    delta_inputs_ = Vector::Zero(design_.horizon * design_.input_dimension);
    objective_ = 0.0;
    constraints_active_ = false;
    converged_ = true;
}

int MpcController::Update(const Vector& state, double reference,
                          Vector* command) {
    Vector target(design_.output_dimension);
    target.setConstant(reference);
    return Update(state, target, command);
}

int MpcController::Update(const Vector& state, const Vector& reference,
                          Vector* command) {
    if (command == nullptr) {
        throw std::invalid_argument(
            "MpcController::Update: command must not be null");
    }
    const int n = design_.state_dimension;
    const int m = design_.input_dimension;
    const int p = design_.output_dimension;
    const int steps = design_.horizon;
    if (state.size() != n) {
        throw std::invalid_argument(
            "MpcController::Update: the state has the wrong size");
    }
    if (reference.size() != p) {
        throw std::invalid_argument(
            "MpcController::Update: the reference has the wrong size");
    }
    if (!state.allFinite() || !reference.allFinite()) {
        // Holding the previous command is the safe response to a sensor fault.
        *command = previous_command_;
        return 0;
    }

    // Augmented state: the measured state stacked with the previous command.
    Vector augmented(n + m);
    augmented.head(n) = state;
    augmented.tail(m) = previous_command_;

    // Stack the reference over the horizon: a constant reference is the usual
    // case, and the signature with a vector covers the rest.
    reference_sequence_.assign(static_cast<std::size_t>(steps), reference);

    // f = Gamma' Qbar (Phi xi - R_seq).
    Vector stacked_reference(steps * p);
    for (int step = 0; step < steps; ++step) {
        stacked_reference.segment(step * p, p) = reference;
    }
    const Vector prediction_error = design_.phi * augmented - stacked_reference;
    const bool terminal_active =
        options_.use_terminal_cost && design_.terminal_cost.size() > 0;
    const int weighted_steps = terminal_active ? steps - 1 : steps;
    Vector weighted = Vector::Zero(prediction_error.size());
    for (int step = 0; step < weighted_steps; ++step) {
        weighted.segment(step * p, p) =
            options_.output_weight * prediction_error.segment(step * p, p);
    }
    Vector f = design_.gamma.transpose() * weighted;
    // Terminal penalty. Without this term the optimisation is myopic over the
    // horizon and the unconstrained answer is not the LQR gain; with it, the
    // horizon can be as short as one step and the answer is still optimal.
    //
    // The penalty is on the deviation from the equilibrium that holds the
    // reference, not from the origin. Penalising xi itself is the same thing
    // when the reference is zero - which is why it survives every regulation
    // test - and is wrong for any other reference: the stage cost pulls the
    // output towards r while the terminal cost pulls the state towards 0, and
    // the loop settles quietly at a point in between. That is a wrong answer
    // that looks like a converged one, so the shift is computed rather than
    // assumed.
    if (options_.use_terminal_cost && design_.gamma_terminal.size() > 0) {
        const Vector equilibrium = design_.EquilibriumFor(reference);
        f += design_.gamma_terminal.transpose() * design_.terminal_cost *
             (design_.phi_terminal * augmented - equilibrium);
    }

    // Constraint on the decision variable: the input increment, i.e. the slew
    // rate times the sample time. A box, so the projection is a clamp - which
    // is what makes the constrained solve cheap and its iteration count
    // predictable.
    //
    // The *absolute* input limit is applied to the command that leaves this
    // function (see below) rather than inside the QP. That is the engineering
    // approximation the literature uses for embedded MPC: it keeps the
    // constraint violation at zero, at the cost of the move not being the exact
    // constrained optimum when the limit binds. Doing both exactly needs a
    // projection onto a band in the cumulative sum, which is a QP of its own;
    // the header says which of the two this implementation guarantees.
    const int variables = steps * m;
    Vector box_lower =
        Vector::Constant(variables, -std::numeric_limits<double>::infinity());
    Vector box_upper =
        Vector::Constant(variables, std::numeric_limits<double>::infinity());
    const bool has_box = options_.delta_input_min.size() == m &&
                         options_.delta_input_max.size() == m;
    if (has_box) {
        for (int step = 0; step < steps; ++step) {
            box_lower.segment(step * m, m) = options_.delta_input_min;
            box_upper.segment(step * m, m) = options_.delta_input_max;
        }
    }
    const bool has_input_limits =
        options_.input_min.size() == m && options_.input_max.size() == m;

    // Accelerated projected gradient (FISTA) on a convex QP with a box: it
    // converges to the constrained optimum, and the iteration count is bounded,
    // which is what makes the per-tick worst case knowable instead of hoped
    // for.
    const Matrix& hessian = design_.hessian;
    const double lipschitz = [&]() {
        Eigen::SelfAdjointEigenSolver<Matrix> solver(hessian);
        return solver.eigenvalues().maxCoeff();
    }();
    const double step = 1.0 / std::max(1e-12, lipschitz);
    auto project = [&](const Vector& point) {
        if (!has_box) {
            return point;
        }
        return ProjectBox(point, box_lower, box_upper);
    };

    Vector solution;
    int iterations = 0;
    converged_ = true;
    if (!has_box) {
        // Nothing is constrained: the analytic solution is exact and iterating
        // would only add error. This is the case that equals the LQR gain.
        solution = -hessian.partialPivLu().solve(f);
    } else {
        converged_ = false;
        solution = project(-step * f);
        Vector previous = solution;
        double momentum = 1.0;
        for (int iteration = 0; iteration < options_.max_iterations;
             ++iteration) {
            const Vector gradient = hessian * solution + f;
            const Vector candidate = project(solution - step * gradient);
            const double next_momentum =
                0.5 * (1.0 + std::sqrt(1.0 + 4.0 * momentum * momentum));
            const double extrapolation = (momentum - 1.0) / next_momentum;
            const Vector extrapolated =
                project(candidate + extrapolation * (candidate - previous));
            const double change = (candidate - previous).norm();
            previous = candidate;
            solution = extrapolated;
            momentum = next_momentum;
            ++iterations;
            const double scale = std::max(1.0, solution.norm());
            if (change <= options_.convergence_tolerance * scale) {
                converged_ = true;
                break;
            }
        }
        // Feasibility is not negotiable: the returned move is inside the box
        // even if the iteration cap was reached before convergence.
        solution = project(solution);
    }

    delta_inputs_ = solution;
    objective_ = solution.dot(hessian * solution) + 2.0 * f.dot(solution);

    // Apply only the first move, then discard the rest: that is the receding
    // horizon, and it is also what gives the loop its feedback.
    const Vector first = solution.head(m);
    Vector next_command = previous_command_ + first;
    constraints_active_ = false;
    // The absolute input limit, applied to the command that is actually sent.
    // Keeping the violation at zero matters more than the last percent of
    // optimality: an actuator asked for 30 N when it has 20 does not deliver
    // 20, it delivers something the model does not describe.
    if (has_input_limits) {
        for (int channel = 0; channel < m; ++channel) {
            const double limited = std::min(
                options_.input_max(channel),
                std::max(options_.input_min(channel), next_command(channel)));
            if (limited != next_command(channel)) {
                constraints_active_ = true;
            }
            next_command(channel) = limited;
        }
    }
    if (has_box) {
        for (int channel = 0; channel < m; ++channel) {
            if (std::abs(std::abs(first(channel)) -
                         options_.delta_input_max(channel)) < 1e-9) {
                constraints_active_ = true;
            }
        }
    }
    previous_command_ = next_command;
    *command = next_command;
    return iterations;
}

}  // namespace control
