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

#ifndef CONTROL_PID_H_
#define CONTROL_PID_H_

// PID: tuning from a model, and the runtime structure that makes it survive
// contact with an actuator.
//
// Two halves, deliberately separate:
//
//   * tuning - turn a model or a pair of (Ku, Pu) measurements into gains, and
//     report the closed-loop poles that result so the design can be judged;
//   * the controller - the discrete implementation with the four structures
//   that
//     are not optional in practice (derivative on the measurement, derivative
//     filtering, anti-windup, output and rate limiting).
//
// The split matters because they fail differently. A badly tuned PID
// oscillates; a badly structured one looks perfectly tuned in simulation and
// winds up, overshoots and chatters on hardware. The controller here therefore
// implements all four structures, and `PidDesign` reports the peaks a design
// will ask for so the limits can be chosen from a number rather than from hope.

#include <Eigen/Core>
#include <string>
#include <vector>

#include "control/design.h"
#include "control/lti.h"
#include "control/types.h"

namespace control {

// The two interchangeable parametrisations, with the conversion the standards
// use: Ti = Kp/Ki and Td = Kd/Kp.
struct PidGains {
    double kp = 0.0;
    double ki = 0.0;  // integral gain, per second
    double kd = 0.0;  // derivative gain, in seconds

    // Ti = Kp/Ki: the time in which the integral term repeats the proportional
    // action. Infinite when Ki is zero.
    double integral_time() const;
    // Td = Kd/Kp: the "lead time" the derivative acts over.
    double derivative_time() const;

    static PidGains FromIntegralAndDerivativeTimes(double kp, double ti,
                                                   double td);

    double proportional_only() const { return kp; }
};

enum class PidForm {
    kP,    // proportional only
    kPi,   // proportional + integral
    kPid,  // all three
};

struct PidTuningOptions {
    // Derivative filter divisor: D(s) = Kd s / (1 + s Td/N). 8..20 is the usual
    // range; a noisy sensor wants the low end (more filtering).
    double derivative_filter_divisor = 10.0;
    // Weights for the pole-placement route: the integrator pole is placed at
    // `integrator_pole_ratio` times the dominant natural frequency.
    double integrator_pole_ratio = 4.0;
};

struct PidConfig {
    PidGains gains;
    double sample_time = 0.0;                   // seconds
    double output_min = 0.0, output_max = 0.0;  // limits; equal means no limit
    bool rate_limit = false;
    double rate_max = 0.0;                    // units per second
    double derivative_filter_divisor = 10.0;  // N
    // Integrator clamp; 0 means "use the output limits".
    double integral_min = 0.0, integral_max = 0.0;
};

// Closed-loop simulation of the discrete controller against the continuous
// plant, sampled at the controller's rate, for reporting the numbers a design
// review asks for: overshoot, settling time, peak command, and whether the
// actuator limit was reached.
struct PidStepMetrics {
    double overshoot_percent = 0.0;
    double settling_time = 0.0;
    double steady_state_error = 0.0;
    double peak_command = 0.0;
    double peak_measurement = 0.0;
    int saturated_ticks = 0;
    int rate_limited_ticks = 0;
    bool settled = false;
};
struct PidDesign {
    PidGains gains;
    PidForm form = PidForm::kPid;
    // Closed-loop poles of the linear design (no saturation, no filtering), the
    // statement that matters for stability.
    ComplexVector closed_loop_poles;
    Vector damping_ratios;
    // Ultimate gain and period, when the tuning route went through them.
    double ultimate_gain = 0.0;
    double ultimate_period = 0.0;
    bool has_ultimate_values = false;
    // Gain and phase margin of the open loop with the controller in place. PID
    // tuning is a frequency-domain exercise, so these are the numbers that say
    // whether it worked.
    double gain_margin_db = 0.0;
    double phase_margin_degrees = 0.0;
    double bandwidth = 0.0;  // rad/s, the crossover frequency
    // The achieved closed-loop step response, simulated with the discrete
    // controller against the continuous plant. Pole placement matches the
    // characteristic polynomial, but the controller's own zeros still shape the
    // response, so the overshoot that results is *not* the second-order
    // formula's overshoot. Reporting the simulated value is what keeps a "10
    // percent overshoot" specification from being quietly missed by a factor of
    // three.
    PidStepMetrics predicted_step;
    bool has_predicted_step = false;

    std::string method;
    std::vector<std::string> warnings;
    std::string notes;
    bool ok = false;
};

// Critical-proportion route: needs the ultimate gain Ku (the proportional gain
// at which the loop oscillates) and the ultimate period Pu. The table is the
// classic one; note that it is deliberately aggressive, which is why the notes
// say so and `ztn_derating` exists.
//
//   P:   Kp = 0.50 Ku
//   PI:  Kp = 0.45 Ku, Ti = Pu / 2
//   PID: Kp = 0.60 Ku, Ti = Pu / 2, Td = Pu / 8
PidDesign TuneZieglerNichols(double ultimate_gain, double ultimate_period,
                             PidForm form, double derating = 1.0);

// The same route, with Ku and Pu computed from the model: Ku is the inverse of
// the plant gain at the phase crossover frequency, and Pu is the period there.
// Refuses a plant whose phase never reaches -180 degrees, because such a plant
// has no ultimate gain and the method does not apply - Ziegler-Nichols is not a
// universal recipe and pretending otherwise wastes a commissioning day.
PidDesign TuneZieglerNicholsFromModel(const LtiSystem& system, PidForm form,
                                      double derating = 1.0);

// Pole placement, the route that is tied to the model rather than to an
// experiment. For a plant whose transfer function the model exposes, the
// closed-loop characteristic polynomial is matched against
// `(s^2 + 2 zeta wn s + wn^2)(s + p0)` with p0 = `integrator_pole_ratio * wn`.
//
// Supported: a first-order plant with PI, and a second-order plant with PID -
// which is what the overwhelming majority of mechanical and electrical axes
// are. A higher-order plant is refused with a pointer to PlacePoles or the LQR,
// which handle it properly, rather than being approximated silently.
PidDesign TunePolePlacement(const LtiSystem& system, PidForm form,
                            double damping_ratio, double natural_frequency,
                            const PidTuningOptions& options = {});

// Suggests wn and zeta from the usual specifications: overshoot sets zeta, and
// the settling time sets wn by wn ~= 4 / (zeta * ts). A convenience for the
// common case, with the formula written down rather than buried.
struct PidSpecifications {
    double overshoot_percent = 10.0;  // first peak of the step response
    double settling_time = 1.0;       // seconds, 2 percent band
};
void SpecificationsToPoles(const PidSpecifications& specifications,
                           double* damping_ratio, double* natural_frequency);

// Margins of the open loop L(s) = C(s) G(s) with the controller in place, by
// scanning the frequency response. Returns false when the phase never crosses
// -180 degrees (no gain margin) or the gain never crosses 0 dB (no bandwidth).
struct OpenLoopMargins {
    double crossover_frequency = 0.0;  // rad/s, |L| = 1
    double phase_margin_degrees = 0.0;
    double phase_crossover_frequency = 0.0;  // arg L = -180 degrees
    double gain_margin_db = 0.0;
    bool has_crossover = false;
    bool has_phase_crossover = false;
};
OpenLoopMargins ComputeMargins(const LtiSystem& system, const PidGains& gains,
                               const PidTuningOptions& options = {});

// The deployable controller. Every structure below is required, not optional:
//
//   * the derivative acts on -y, never on the error, so a setpoint step does
//   not
//     produce a derivative spike (the "derivative kick" that trips an
//     actuator);
//   * the derivative is filtered with Td/N;
//   * the integrator is held when the output is saturated in the direction the
//     error would push it (clamping anti-windup);
//   * the output is limited, optionally with a rate limit, and the integral
//   term
//     is additionally bounded.
//
// It is also the reference the code generator emits, so the arithmetic order
// here and in the generated header is compared tick by tick by the test suite.
class PidController {
 public:
    explicit PidController(const PidConfig& config);

    // Clears the integrator, the filter state and the previous output.
    void Reset();

    // One control period. `measurement` is the process output, `setpoint` the
    // reference; returns the command.
    double Update(double measurement, double setpoint);

    // Bumpless transfer: adopt the current command as the integrator value so
    // that switching from manual to automatic does not produce a step. Without
    // this, every commissioning session starts with a jolt.
    void TrackManualOutput(double command);

    double integral_term() const { return integral_; }
    double derivative_term() const { return derivative_state_; }
    bool saturated() const { return saturated_; }
    bool rate_limited() const { return rate_limited_; }
    const PidConfig& config() const { return config_; }

 private:
    PidConfig config_;
    double integral_ = 0.0;
    double derivative_state_ = 0.0;
    double measurement_previous_ = 0.0;
    double output_previous_ = 0.0;
    bool first_call_ = true;
    bool saturated_ = false;
    bool rate_limited_ = false;
};

PidStepMetrics SimulatePidStep(const LtiSystem& continuous_system,
                               const PidConfig& config, double setpoint,
                               double duration);

}  // namespace control

#endif  // CONTROL_PID_H_
