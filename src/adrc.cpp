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
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>

namespace control {
namespace {

// Binomial coefficients of (s + wo)^m: the observer gains that place every
// estimation-error pole at -wo. Written as the binomial expansion rather than
// as a table so the pattern is visible: m = 2 gives 2wo, wo^2 and m = 3 gives
// 3wo, 3wo^2, wo^3.
double Binomial(int n, int k) {
    if (k < 0 || k > n) {
        return 0.0;
    }
    double result = 1.0;
    for (int i = 0; i < k; ++i) {
        result =
            result * static_cast<double>(n - i) / static_cast<double>(i + 1);
    }
    return result;
}

Vector ObserverGains(int order, double bandwidth) {
    Vector gains(order);
    for (int i = 0; i < order; ++i) {
        // beta_i = C(m, i+1) wo^(i+1) with m the ESO order.
        gains(i) = Binomial(order, i + 1) * std::pow(bandwidth, i + 1);
    }
    return gains;
}

}  // namespace

AdrcDesign DesignAdrc(AdrcOrder order, const AdrcOptions& options) {
    if (!(options.controller_bandwidth > 0.0)) {
        throw std::invalid_argument(
            "DesignAdrc: the controller bandwidth must be positive; it is the "
            "one "
            "number that sets the closed-loop speed (4/wc is the 2 percent "
            "settling time)");
    }
    if (!(options.critical_gain != 0.0)) {
        throw std::invalid_argument(
            "DesignAdrc: the critical gain b0 must be non-zero; it is the sign "
            "and "
            "scale of the input's effect, so zero makes the control law "
            "undefined");
    }
    if (!(options.sample_time > 0.0)) {
        throw std::invalid_argument(
            "DesignAdrc: the sample time must be positive");
    }

    AdrcDesign design;
    design.order = order;
    design.controller_bandwidth = options.controller_bandwidth;
    design.critical_gain = options.critical_gain;
    design.observer_bandwidth =
        options.observer_bandwidth > 0.0
            ? options.observer_bandwidth
            : options.observer_bandwidth_ratio * options.controller_bandwidth;
    const double wc = design.controller_bandwidth;
    const double wo = design.observer_bandwidth;

    // The controller gains. First order: y' = b0 u + f with u0 = wc (r - z1)
    // gives a closed loop at -wc. Second order: u0 = wc^2 (r - z1) - 2 zeta wc
    // z2 gives s^2 + 2 zeta wc s + wc^2.
    const int eso_order = order == AdrcOrder::kFirstOrder ? 2 : 3;
    if (order == AdrcOrder::kFirstOrder) {
        design.kp = wc;
        design.kd = 0.0;
    } else {
        design.kp = wc * wc;
        design.kd = 2.0 * options.damping_ratio * wc;
    }
    design.observer_gains = ObserverGains(eso_order, wo);

    // The ESO as a linear system driven by the measurement and the input:
    //   z' = A_e z + B_y y + B_u u
    // The estimation-error dynamics are governed by A_e itself, which is why
    // its eigenvalues are the observer poles that the gain choice places at
    // -wo.
    const int size = eso_order;
    Matrix a_e = Matrix::Zero(size, size);
    Vector b_measurement = Vector::Zero(size);
    Vector b_input = Vector::Zero(size);
    // The structure is  A_e = [[-b1, 1, 0], [-b2, 0, 1], [-b3, 0, 0]]: every
    // gain sits in the FIRST COLUMN and the chain of integrators runs along the
    // superdiagonal. Putting the gains on the diagonal instead (which an
    // earlier version did) produces a diagonal matrix whose eigenvalues are
    // -b_i, not -wo, so the observer poles are wrong, the disturbance estimate
    // is wrong, and the closed loop settles at about half the reference. The
    // identity below - every ESO pole at -wo - is what that mistake breaks, and
    // it is why the test asserts the poles rather than the response.
    for (int i = 0; i < size; ++i) {
        a_e(i, 0) = -design.observer_gains(i);
        if (i + 1 < size) {
            a_e(i, i + 1) = 1.0;
        }
        b_measurement(i) = design.observer_gains(i);
    }
    // The input enters one order below the top: through the state that
    // integrates into the measured output.
    b_input(size - 2) = options.critical_gain;
    design.eso_continuous = a_e;
    design.eso_input_continuous.resize(size, 2);
    design.eso_input_continuous.col(0) = b_measurement;
    design.eso_input_continuous.col(1) = b_input;

    Eigen::EigenSolver<Matrix> solver(a_e, false);
    design.observer_poles = solver.eigenvalues();
    std::sort(design.observer_poles.data(),
              design.observer_poles.data() + design.observer_poles.size(),
              [](const std::complex<double>& left,
                 const std::complex<double>& right) {
                  return left.real() > right.real();
              });

    // Exact zero-order-hold discretization through the augmented exponential,
    // the same route the rest of the library uses. At the bandwidths ADRC runs
    // at, forward Euler can be unstable for a sample time this handles, and the
    // difference is invisible until the loop is on hardware.
    Matrix augmented = Matrix::Zero(size + 2, size + 2);
    augmented.topLeftCorner(size, size) = a_e;
    augmented.topRightCorner(size, 2) = design.eso_input_continuous;
    const Matrix scaled = augmented * options.sample_time;
    const Matrix exponential = scaled.exp();
    design.eso_discrete = exponential.topLeftCorner(size, size);
    design.eso_input_discrete = exponential.topRightCorner(size, 2);

    // Noise gain: how much a measurement disturbance is amplified into the
    // command. The dominant term is beta_last / b0 through the cancellation
    // path, which is the price of a high observer bandwidth.
    design.noise_gain = design.observer_gains(size - 1) /
                        std::abs(options.critical_gain) * options.sample_time;
    design.sample_time = options.sample_time;
    design.output_limit_low = options.output_min;
    design.output_limit_high = options.output_max;

    std::ostringstream notes;
    notes << "linear ADRC, "
          << (order == AdrcOrder::kFirstOrder ? "first" : "second")
          << "-order plant: wc = " << wc << " rad/s, wo = " << wo
          << " rad/s (ratio " << wo / wc << "), b0 = " << options.critical_gain
          << ", ESO order " << eso_order
          << ". The observer poles are all at -wo by "
          << "construction; the control law cancels the estimated disturbance "
             "and "
          << "closes -wc (";
    if (order == AdrcOrder::kFirstOrder) {
        notes << "first-order loop, 4/wc to 2 percent)";
    } else {
        notes << "s^2 + 2*" << options.damping_ratio << "*wc s + wc^2)";
    }
    design.notes = notes.str();

    const double ratio = wo / wc;
    if (ratio < 2.0) {
        design.warnings.push_back("the observer bandwidth is only " +
                                  std::to_string(ratio) +
                                  " times the controller bandwidth: the "
                                  "disturbance estimate lags the "
                                  "control action and the loop will be "
                                  "sluggish or ring. Use 3 to 10");
    }
    if (ratio > 20.0) {
        design.warnings.push_back(
            "the observer bandwidth is " + std::to_string(ratio) +
            " times the controller bandwidth: the loop gain is now dominated "
            "by "
            "measurement noise reaching the actuator. Check the noise gain "
            "reported here against what the actuator can tolerate");
    }
    // The sample time has to resolve the observer, not the controller: the
    // observer runs faster than the loop it serves.
    const double observer_period = 2.0 * M_PI / wo;
    if (options.sample_time > observer_period / 10.0) {
        design.warnings.push_back(
            "the sample time is more than a tenth of the observer period (" +
            std::to_string(observer_period) +
            " s): the discrete observer will lag the one this design assumed. "
            "Raise the sample rate or lower wo");
    }
    design.ok = true;
    return design;
}

AdrcController::AdrcController(const AdrcDesign& design) : design_(design) {
    if (!design_.ok) {
        throw std::invalid_argument(
            "AdrcController: the design is not valid; build it with "
            "DesignAdrc");
    }
    Reset();
}

void AdrcController::Reset() {
    observer_state_.setZero(design_.eso_continuous.rows());
    disturbance_ = 0.0;
    command_previous_ = 0.0;
    saturated_ = false;
    error_ = false;
}

double AdrcController::Update(double measurement, double setpoint) {
    if (!std::isfinite(measurement) || !std::isfinite(setpoint)) {
        error_ = true;
        return command_previous_;
    }
    error_ = false;

    // Control law, using the *estimated* state and disturbance: this is what
    // makes the loop insensitive to the model being wrong.
    const double z1 = observer_state_(0);
    const double z_disturbance = observer_state_(observer_state_.size() - 1);
    double u0 = design_.kp * (setpoint - z1);
    if (design_.order == AdrcOrder::kSecondOrder) {
        u0 -= design_.kd * observer_state_(1);
    }
    double command = (u0 - z_disturbance) / design_.critical_gain;

    // Actuator limit, applied before the observer update so the observer sees
    // what the actuator was actually asked for.
    saturated_ = false;
    const double limit_low = design_.output_limit_low;
    const double limit_high = design_.output_limit_high;
    if (limit_low != limit_high) {
        const double limited =
            std::min(limit_high, std::max(limit_low, command));
        saturated_ = limited != command;
        command = limited;
    }

    // Observer update, exact zero-order hold: z+ = Ad z + Bd [y; u].
    Vector drive(2);
    drive(0) = measurement;
    drive(1) = command;
    observer_state_ = design_.eso_discrete * observer_state_ +
                      design_.eso_input_discrete * drive;
    disturbance_ = observer_state_(observer_state_.size() - 1);
    command_previous_ = command;
    return command;
}

AdrcStepMetrics SimulateAdrcStep(const LtiSystem& continuous_system,
                                 const AdrcDesign& design, double setpoint,
                                 double duration, double input_disturbance) {
    continuous_system.Validate("SimulateAdrcStep");
    if (!continuous_system.is_continuous()) {
        throw std::invalid_argument(
            "SimulateAdrcStep: the model must be continuous");
    }
    if (continuous_system.output_dimension() != 1 ||
        continuous_system.input_dimension() != 1) {
        throw std::invalid_argument(
            "SimulateAdrcStep: only SISO models are supported");
    }
    const int n = continuous_system.state_dimension();
    AdrcController controller(design);
    Vector state = Vector::Zero(n);
    AdrcStepMetrics metrics;

    const int ticks = static_cast<int>(duration / design.sample_time);
    if (ticks < 2) {
        throw std::invalid_argument(
            "SimulateAdrcStep: the duration is too short");
    }
    Vector measurements(ticks);
    for (int tick = 0; tick < ticks; ++tick) {
        const double measurement = (continuous_system.c() * state)(0, 0);
        const double command = controller.Update(measurement, setpoint);
        measurements(tick) = measurement;
        if (controller.saturated()) {
            ++metrics.saturated_ticks;
        }
        // Zero-order hold step of the continuous plant, with the disturbance
        // entering through the input channel.
        Matrix augmented = Matrix::Zero(n + 1, n + 1);
        augmented.topLeftCorner(n, n) = continuous_system.a();
        augmented.topRightCorner(n, 1) = continuous_system.b();
        const Matrix scaled = augmented * design.sample_time;
        const Matrix step = scaled.exp();
        Vector extended(n + 1);
        extended.head(n) = state;
        extended(n) = command + input_disturbance;
        state = (step * extended).head(n);
    }

    metrics.peak_command = controller.last_command();
    double peak = measurements(0);
    for (int tick = 0; tick < ticks; ++tick) {
        peak = std::max(peak, measurements(tick));
    }
    if (setpoint != 0.0) {
        metrics.overshoot_percent =
            std::max(0.0, (peak - setpoint) / std::abs(setpoint) * 100.0);
    }
    metrics.steady_state_error = setpoint - measurements(ticks - 1);
    // Settling time: the last moment the response left the 2 percent band. The
    // response starts outside it, so "settled" means it came back and stayed -
    // which is the last out-of-band sample not being the final one. Testing
    // "never left the band" instead marks every simulation unsettled.
    const double band = 0.02 * std::abs(setpoint);
    metrics.settled = false;
    for (int tick = ticks - 1; tick >= 0; --tick) {
        if (std::abs(measurements(tick) - setpoint) > band) {
            metrics.settling_time = (tick + 1) * design.sample_time;
            metrics.settled = (tick + 1) < ticks;
            break;
        }
    }
    if (metrics.settling_time == 0.0) {
        metrics.settled = true;
    }
    return metrics;
}

}  // namespace control
