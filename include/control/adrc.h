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

#ifndef CONTROL_ADRC_H_
#define CONTROL_ADRC_H_

// Active disturbance rejection control.
//
// The idea in one paragraph: rather than model the plant accurately enough to
// invert it, treat *everything* you did not model - unknown dynamics, parameter
// error, external disturbance, unmodelled coupling - as a single additive term
// `f` entering through the input channel, estimate it with an observer, and
// cancel it. What is left is an integrator chain, which a proportional (or PD)
// law handles. The engineering appeal is that the design needs one number with
// a physical meaning (the input gain `b0`) and two bandwidths, and it is robust
// to being wrong about all three.
//
//   first order:   y'  = b0 u + f,     estimate f with a 2nd-order ESO
//   second order:  y'' = b0 u + f,     estimate f with a 3rd-order ESO
//
//                    z1' = z2 + beta1 (y - z1) + b0 u
//   (2nd order ESO)  z2' =       beta2 (y - z1)
//
//                    z1' = z2 + beta1 (y - z1)
//   (3rd order ESO)  z2' = z3 + beta2 (y - z1) + b0 u
//                    z3' =       beta3 (y - z1)
//
// The observer gains come from a single bandwidth, which is what makes the
// method tunable on a commissioning afternoon instead of a week:
//
//   beta = [2 wo, wo^2]        for the 2nd-order ESO
//   beta = [3 wo, 3 wo^2, wo^3] for the 3rd-order ESO
//
// so that the estimation error dynamics have *all* their poles at `-wo`. That
// identity is exact, it is checked in the test suite, and it is the reason the
// observer has no overshoot of its own to tune out.
//
// The control law cancels the estimate and closes the loop on the remainder:
//
//   u = (u0 - z2) / b0  (first order),  u = (u0 - z3) / b0  (second order)
//   u0 = wc (r - z1)                    u0 = wc^2 (r - z1) - 2 zeta wc z2
//
// Two things worth saying before the numbers:
//
//   * `b0` does not have to be accurate. A factor-of-two error costs bandwidth,
//     not stability, because the loop gain scales with `b0_true/b0_used` and
//     the observer absorbs the rest. That is measurable and the example shows
//     it.
//   * The bandwidths are not free. `wo` multiplies the noise the observer
//   passes
//     into the control signal, so raising it buys disturbance rejection with
//     actuator noise. `AnalyseAdrc` reports the noise gain so the trade is a
//     number rather than an opinion.

#include <Eigen/Core>
#include <string>
#include <vector>

#include "control/types.h"

namespace control {

enum class AdrcOrder {
    kFirstOrder,   // plant behaves like y' = b0 u + f  (speed, current, flow)
    kSecondOrder,  // plant behaves like y'' = b0 u + f (position, angle)
};

struct AdrcOptions {
    // Desired closed-loop bandwidth, rad/s. This is the one performance number:
    // a first-order loop settles to 2 percent in about 4/wc, a second-order one
    // in about 4/(zeta wc).
    double controller_bandwidth = 0.0;
    // Observer bandwidth, rad/s. Defaults to `observer_bandwidth_ratio * wc`.
    // The usual advice is 3..10 times the controller bandwidth: higher rejects
    // disturbances faster and passes more sensor noise.
    double observer_bandwidth = 0.0;
    double observer_bandwidth_ratio = 5.0;
    // Input gain: the coefficient of u in the (nominal, first-order) plant
    // model. For a plant the user can measure, b0 = K/(tau) for a first-order
    // lag, or 1/(m) for a mass, and so on. An error here scales the loop gain.
    double critical_gain = 1.0;
    // Closed-loop damping for the second-order law. 1.0 gives no overshoot.
    double damping_ratio = 1.0;
    // Sample time of the discrete implementation. The observer is discretized
    // by exact zero-order hold, not by Euler: at the bandwidths ADRC runs at,
    // an Euler observer can be unstable for a sample time the exact one
    // handles.
    double sample_time = 0.0;
    // Actuator limit. Both bounds equal means unlimited.
    double output_min = 0.0;
    double output_max = 0.0;
};

struct AdrcDesign {
    AdrcOrder order = AdrcOrder::kFirstOrder;
    double controller_bandwidth = 0.0;
    double observer_bandwidth = 0.0;
    double critical_gain = 1.0;
    // Controller gains: kp (and kd for the second-order law).
    double kp = 0.0;
    double kd = 0.0;
    // Observer gains, beta.
    Vector observer_gains;
    // ESO state matrix and input matrices, continuous and discrete.
    Matrix eso_continuous;        // A_e
    Matrix eso_discrete;          // exp(A_e Ts)
    Matrix eso_input_continuous;  // columns: one per (measurement, input)
    Matrix eso_input_discrete;
    // Eigenvalues of A_e: all at -wo, which is the identity the
    // parameterization buys and the test asserts.
    ComplexVector observer_poles;
    // Noise gain: the multiple of the controller bandwidth by which measurement
    // noise reaches the control signal. Reported because it is the real cost of
    // raising wo.
    double noise_gain = 0.0;
    // Actuator limits, carried over from the options so the controller has
    // them.
    double output_limit_low = 0.0;
    double output_limit_high = 0.0;
    // Sample time the discrete observer was built for. The controller steps at
    // this rate; nothing checks that it does, which is why it is part of the
    // design record rather than a parameter of the update call.
    double sample_time = 0.0;
    std::string notes;
    std::vector<std::string> warnings;
    bool ok = false;
};

// Builds the observer and the control law. Throws when the bandwidths or the
// sample time are not usable, and warns when the observer is too slow relative
// to the controller to do its job.
AdrcDesign DesignAdrc(AdrcOrder order, const AdrcOptions& options);

struct AdrcStepMetrics {
    double overshoot_percent = 0.0;
    double settling_time = 0.0;
    // Steady-state error measured at the end of the run. What distinguishes
    // ADRC from a proportional law is that this goes to zero when a constant
    // input disturbance is applied, so the caller runs the simulation twice
    // (with and without the disturbance) and compares - the function does not
    // guess which comparison the caller wants.
    double steady_state_error = 0.0;
    double peak_command = 0.0;
    int saturated_ticks = 0;
    bool settled = false;
};

// Simulation of the discrete controller against a plant given as (A, B, C) plus
// a constant input disturbance, for the numbers a design review asks for.
AdrcStepMetrics SimulateAdrcStep(const LtiSystem& continuous_system,
                                 const AdrcDesign& design, double setpoint,
                                 double duration,
                                 double input_disturbance = 0.0);

// The deployable controller. Same structures as everything else in this
// library: all state in the object, fixed per-tick cost, a NaN guard that holds
// the last command, and an actuator limit that reports when it is active.
class AdrcController {
 public:
    AdrcController(const AdrcDesign& design);

    void Reset();

    // One control period. Returns the command.
    double Update(double measurement, double setpoint);

    // The disturbance estimate, in the units of the plant's acceleration. Worth
    // logging: a step change in it is a fault, a slow drift is a parameter
    // change.
    double estimated_disturbance() const { return disturbance_; }
    const Vector& observer_state() const { return observer_state_; }
    bool saturated() const { return saturated_; }
    bool error() const { return error_; }
    double last_command() const { return command_previous_; }

 private:
    AdrcDesign design_;
    Vector observer_state_;
    double disturbance_ = 0.0;
    double command_previous_ = 0.0;
    bool saturated_ = false;
    bool error_ = false;
};

}  // namespace control

#endif  // CONTROL_ADRC_H_
