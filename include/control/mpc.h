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

#ifndef CONTROL_MPC_H_
#define CONTROL_MPC_H_

// Model predictive control: receding-horizon optimisation with explicit
// constraints.
//
// The formulation is the velocity (incremental) form, which is the one worth
// having because it puts integral action into the structure rather than into an
// extra state someone has to remember to add:
//
//   xi[k] = [x[k]; u[k-1]]                      (augmented state)
//   xi[k+1] = [A B; 0 I] xi[k] + [B; I] du[k]   (decision variable is du)
//
// Stacking N steps gives the predicted outputs Y = Phi xi + Gamma dU, and the
// cost becomes a quadratic in dU:
//
//   J(dU) = dU' H dU + 2 f' dU + const,
//   H = Gamma' Qbar Gamma + Rbar + eps I,   f = Gamma' Qbar (Phi xi - R),
//
// so the whole problem is a small dense QP per tick. Three things about that QP
// are worth stating up front, because they are where MPC implementations go
// wrong:
//
//   * **Unconstrained, it is exactly the LQR.** With the terminal cost taken
//   from
//     the DARE solution, the unconstrained optimum equals -K x for *every*
//     horizon. That is a theorem, not an approximation, and the test suite
//     checks it - which is also how you know the prediction matrices are right.
//   * **Constraints are the only reason to solve a QP.** If nothing is active,
//   the
//     answer is the analytic one and there is no reason to iterate.
//   * **The deadline is the design constraint.** A QP that sometimes takes 500
//     microseconds in a loop that must close in 200 is not a controller. The
//     solve is bounded by an iteration limit and reports whether it converged,
//     so the timing question is answerable rather than hopeful.

#include <Eigen/Core>
#include <string>
#include <vector>

#include "control/design.h"
#include "control/lti.h"
#include "control/types.h"

namespace control {

struct MpcOptions {
    // Prediction horizon, in ticks. Cover about twice the dominant response
    // time.
    int horizon = 10;
    // Weight on the tracked output error, per output, applied at every
    // predicted step. This is the Q of the equivalent LQR.
    Matrix output_weight;
    // Weight on the input increments, per input. This is the R of the
    // equivalent LQR, but it penalises du rather than u, which is what gives
    // the velocity form its integral action.
    Matrix input_weight;
    // Bounds on the absolute input. An empty matrix means unbounded. Note that
    // these are what make the problem non-trivial: with no bounds the answer is
    // the analytic LQR gain.
    Matrix input_min;
    Matrix input_max;
    // Bounds on the input increment, i.e. the actuator slew rate times the
    // sample time. Empty means unbounded.
    Matrix delta_input_min;
    Matrix delta_input_max;
    // Terminal cost. When true, P is the DARE solution of the *augmented*
    // problem and the stage cost covers steps 1..N-1, which together make the
    // unconstrained controller exactly the LQR gain for any horizon. Nominal
    // stability then holds with a horizon as short as one step.
    bool use_terminal_cost = true;
    // Added to the diagonal of H. The QP is solved by a first-order method,
    // which wants a strictly positive definite Hessian even when Rbar is zero.
    double regularization = 1e-9;
    // Iteration cap for the projected-gradient solver. The per-tick worst case
    // is this many iterations, which is the number the real-time check needs.
    int max_iterations = 500;
    double convergence_tolerance = 1e-10;
    // Note on what is guaranteed. The input *increment* bounds are enforced
    // exactly, inside the QP: they are a box, so the projection is a clamp and
    // the constrained optimum is reached. The absolute input bounds are applied
    // to the command that leaves Update(), which keeps the violation at zero
    // but means the move is not the exact constrained optimum while the limit
    // binds. That is the standard embedded compromise; the alternative is a
    // projection onto a band in the cumulative sum, which is a QP of its own
    // and does not fit a fixed per-tick budget without more machinery than this
    // file has.
};

// Everything that can be computed before the loop runs. Building this is the
// expensive part; solving for a control move is then a fixed-size QP.
struct MpcDesign {
    LtiSystem augmented;   // xi = [x; u], input is du
    Matrix phi;            // (N*p) x (n+m)
    Matrix gamma;          // (N*p) x (N*m)
    Matrix hessian;        // H, (N*m) x (N*m)
    Matrix terminal_cost;  // P, the DARE solution of the augmented problem
    // Terminal prediction: xi[k+N] = phi_terminal xi[k] + gamma_terminal dU.
    // The terminal penalty acts on the augmented state, not on the output,
    // which is why it needs its own pair of matrices.
    Matrix phi_terminal;    // (n+m) x (n+m)
    Matrix gamma_terminal;  // (n+m) x (N*m)
    Matrix output_matrix;   // Cbar, (p) x (n+m): y = Cbar xi
    Matrix output_weight;   // Qbar, kept so the analytic solution can be formed
    int horizon = 0;
    int state_dimension = 0;
    int input_dimension = 0;
    int output_dimension = 0;

    // The augmented state that holds a constant reference: the least-squares
    // solution of [(A - I) B; C 0] [x; u] = [0; r]. The terminal cost is
    // measured from here rather than from the origin, which is what makes a
    // non-zero reference track instead of settling at a compromise.
    Vector EquilibriumFor(const Vector& reference) const;

    // Analytic optimum of the unconstrained QP: dU* = -H^-1 f. Exposed because
    // this is the reference the constrained solver is measured against, and
    // because for an unconstrained problem it is the whole answer.
    Vector UnconstrainedSolution(const Vector& augmented_state,
                                 const std::vector<Vector>& reference) const;
};

// Builds the prediction matrices and the terminal cost from a *discrete* model.
// The model has to be the sampled one the controller runs against: building a
// prediction from a continuous model is how an MPC ends up a tick behind
// reality.
MpcDesign BuildMpc(const LtiSystem& discrete_system, const MpcOptions& options);

// Stateful receding-horizon controller.
class MpcController {
 public:
    MpcController(const LtiSystem& discrete_system, const MpcOptions& options);

    // Clears the internal input memory.
    void Reset();

    // One tick. `state` is the measured state (n entries), `reference` the
    // setpoint for the output (p entries); writes the command (m entries).
    //
    // Returns the number of solver iterations used, which is what a real-time
    // check needs: the worst case over a long run is the number to compare with
    // the period.
    int Update(const Vector& state, const Vector& reference, Vector* command);

    // Constant reference over the whole horizon.
    int Update(const Vector& state, double reference, Vector* command);

    const MpcDesign& design() const { return design_; }
    // Value of the cost at the returned solution, for logging.
    double objective() const { return objective_; }
    // True when the returned move touches a constraint.
    bool constraints_active() const { return constraints_active_; }
    // True when the solver hit its iteration cap without meeting the tolerance:
    // the move is feasible but not optimal, and a caller that cares about the
    // difference can see it.
    bool solver_converged() const { return converged_; }
    // The full predicted input-increment sequence, for plotting.
    const Vector& predicted_increments() const { return delta_inputs_; }
    // Previous command, which the velocity form needs.
    const Vector& last_command() const { return previous_command_; }

 private:
    MpcDesign design_;
    MpcOptions options_;
    Vector previous_command_;
    Vector delta_inputs_;
    std::vector<Vector> reference_sequence_;
    double objective_ = 0.0;
    bool constraints_active_ = false;
    bool converged_ = true;
};

}  // namespace control

#endif  // CONTROL_MPC_H_
