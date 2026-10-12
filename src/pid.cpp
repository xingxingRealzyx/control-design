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
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>

namespace control {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Roots of a real polynomial given highest-power-first, via the eigenvalues of
// its companion matrix. Eigen's unsupported Polynomials module would be fewer
// lines but its Durand-Kerner iteration is fragile for repeated roots, and
// repeated roots are exactly what a double integrator produces.
ComplexVector PolynomialRoots(const Vector& coefficients) {
    Eigen::Index degree = coefficients.size() - 1;
    while (degree > 0 &&
           std::abs(coefficients(coefficients.size() - 1 - degree)) < 1e-300) {
        --degree;
    }
    if (degree <= 0) {
        return ComplexVector(0);
    }
    const double leading = coefficients(0);
    Matrix companion = Matrix::Zero(degree, degree);
    for (Eigen::Index i = 0; i < degree; ++i) {
        companion(0, i) = -coefficients(i + 1) / leading;
    }
    for (Eigen::Index i = 1; i < degree; ++i) {
        companion(i, i - 1) = 1.0;
    }
    Eigen::EigenSolver<Matrix> solver(companion, false);
    ComplexVector roots = solver.eigenvalues();
    // Sort by descending real part: the dominant pole first, as elsewhere.
    std::sort(roots.data(), roots.data() + roots.size(),
              [](const std::complex<double>& left,
                 const std::complex<double>& right) {
                  return left.real() > right.real();
              });
    return roots;
}

// product of (s - p_i), highest power first.
Vector PolynomialFromRoots(const ComplexVector& roots) {
    Vector coefficients = Vector::Ones(1);
    for (Eigen::Index i = 0; i < roots.size(); ++i) {
        Vector next = Vector::Zero(coefficients.size() + 1);
        for (Eigen::Index k = 0; k < coefficients.size(); ++k) {
            next(k) += coefficients(k);
            next(k + 1) -= roots(i).real() * coefficients(k);
        }
        coefficients = next;
    }
    return coefficients;
}

// Convolves two highest-power-first polynomials.
Vector PolynomialMultiply(const Vector& left, const Vector& right) {
    Vector product = Vector::Zero(left.size() + right.size() - 1);
    for (Eigen::Index i = 0; i < left.size(); ++i) {
        for (Eigen::Index j = 0; j < right.size(); ++j) {
            product(i + j) += left(i) * right(j);
        }
    }
    return product;
}

// Adds two polynomials of possibly different lengths (highest power first).
Vector PolynomialAdd(const Vector& left, const Vector& right) {
    const Eigen::Index size = std::max(left.size(), right.size());
    Vector sum = Vector::Zero(size);
    sum.tail(left.size()) += left;
    sum.tail(right.size()) += right;
    return sum;
}

// The plant as a rational function: numerator and denominator polynomials
// (highest power first) and the gain. Obtained from the verified pole/zero
// computation, which is why it is only available for a SISO continuous model.
struct RationalPlant {
    Vector numerator;    // N(s), scaled so that N(0)/D(0) equals the DC gain
    Vector denominator;  // D(s)
    bool valid = false;
    std::string why_not;
};

RationalPlant AsRational(const LtiSystem& system) {
    RationalPlant plant;
    if (!system.is_continuous()) {
        plant.why_not =
            "the model is discrete; a PID is a continuous-time object, "
            "so tune it against the continuous model";
        return plant;
    }
    if (system.input_dimension() != 1 || system.output_dimension() != 1) {
        plant.why_not = "only SISO models are supported";
        return plant;
    }
    PoleZeroResult pole_zero;
    try {
        pole_zero = PolesAndZeros(system);
    } catch (const std::exception& problem) {
        plant.why_not = problem.what();
        return plant;
    }
    plant.denominator = PolynomialFromRoots(pole_zero.poles);
    plant.numerator = PolynomialFromRoots(pole_zero.zeros);
    // Scale the numerator so the DC gain matches the model exactly. Taking the
    // leading coefficient from the state-space form instead is where a factor
    // of b goes missing.
    const double dc = DcGain(system)(0, 0);
    const double denominator_at_zero =
        plant.denominator(plant.denominator.size() - 1);
    const double numerator_at_zero =
        plant.numerator(plant.numerator.size() - 1);
    if (std::abs(numerator_at_zero) < 1e-300) {
        // A zero at the origin (the plant has a differentiating path): scale by
        // the leading coefficients instead.
        plant.why_not =
            "the plant has a zero at the origin, so its DC gain does not fix "
            "the "
            "numerator scale";
        return plant;
    }
    plant.numerator *= dc * denominator_at_zero / numerator_at_zero;
    plant.valid = true;
    return plant;
}

// C(jw) for the PID, in the serial (non-interacting) form.
std::complex<double> PidResponseAt(const PidGains& gains, double omega,
                                   double filter_divisor) {
    const std::complex<double> j_omega(0.0, omega);
    std::complex<double> response = gains.kp;
    if (gains.ki != 0.0 && omega > 0.0) {
        response += gains.ki / j_omega;
    }
    // At omega = 0 the integral term is unbounded, which is the point of it;
    // the frequency scan starts above zero and never asks.
    if (gains.kd != 0.0) {
        const double td = gains.kd / gains.kp;
        const double filter_time =
            filter_divisor > 0.0 ? td / filter_divisor : 0.0;
        response += gains.kd * j_omega / (1.0 + j_omega * filter_time);
    }
    return response;
}

void FillPoleReport(PidDesign* design,
                    const Vector& characteristic_polynomial) {
    design->closed_loop_poles = PolynomialRoots(characteristic_polynomial);
    design->damping_ratios = Vector(design->closed_loop_poles.size());
    for (Eigen::Index i = 0; i < design->closed_loop_poles.size(); ++i) {
        const std::complex<double>& pole = design->closed_loop_poles(i);
        const double magnitude = std::abs(pole);
        design->damping_ratios(i) =
            magnitude > 0.0 ? -pole.real() / magnitude : 0.0;
    }
    bool unstable = false;
    for (Eigen::Index i = 0; i < design->closed_loop_poles.size(); ++i) {
        if (design->closed_loop_poles(i).real() >= 0.0) {
            unstable = true;
        }
    }
    if (unstable) {
        design->warnings.push_back(
            "the closed loop has a pole in the right half plane: this tuning "
            "does "
            "not stabilise the plant");
    }
}

void FillMargins(PidDesign* design, const LtiSystem& system,
                 const PidTuningOptions& options) {
    const OpenLoopMargins margins =
        ComputeMargins(system, design->gains, options);
    design->bandwidth = margins.crossover_frequency;
    design->phase_margin_degrees = margins.phase_margin_degrees;
    design->gain_margin_db = margins.gain_margin_db;
    if (!margins.has_crossover) {
        design->warnings.push_back(
            "the loop gain never crosses 0 dB, so there is no bandwidth or "
            "phase "
            "margin to report");
    }
    if (!margins.has_phase_crossover) {
        design->warnings.push_back(
            "the loop phase never reaches -180 degrees, so there is no gain "
            "margin");
    }
    // A phase margin below 30 degrees is a design that will ring, and saying so
    // costs nothing.
    if (margins.has_crossover && margins.phase_margin_degrees < 30.0) {
        design->warnings.push_back(
            "phase margin is " + std::to_string(margins.phase_margin_degrees) +
            " degrees; below 30 degrees the loop will ring badly");
    }
}

// The plant's leading behaviour as (dc_gain, poles), used by the pole-placement
// route where the coefficient matching is written out explicitly.
struct PlantShape {
    double dc_gain = 0.0;
    ComplexVector poles;
    bool first_order = false;
    bool second_order = false;
};

PlantShape ShapeOf(const RationalPlant& plant) {
    PlantShape shape;
    shape.poles = PolynomialRoots(plant.denominator);
    const double denominator_at_zero =
        plant.denominator(plant.denominator.size() - 1);
    const double numerator_at_zero =
        plant.numerator(plant.numerator.size() - 1);
    if (std::abs(denominator_at_zero) > 1e-300) {
        shape.dc_gain = numerator_at_zero / denominator_at_zero;
    }
    shape.first_order = shape.poles.size() == 1;
    shape.second_order = shape.poles.size() == 2;
    return shape;
}

}  // namespace

double PidGains::integral_time() const {
    if (ki == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return kp / ki;
}

double PidGains::derivative_time() const {
    if (kp == 0.0) {
        return 0.0;
    }
    return kd / kp;
}

PidGains PidGains::FromIntegralAndDerivativeTimes(double kp, double ti,
                                                  double td) {
    PidGains gains;
    gains.kp = kp;
    gains.ki = (ti > 0.0 && std::isfinite(ti)) ? kp / ti : 0.0;
    gains.kd = kp * td;
    return gains;
}

void SpecificationsToPoles(const PidSpecifications& specifications,
                           double* damping_ratio, double* natural_frequency) {
    const double overshoot_fraction =
        std::max(1e-6, specifications.overshoot_percent / 100.0);
    // Mp = exp(-pi zeta / sqrt(1 - zeta^2)) inverted.
    const double log_mp = std::log(overshoot_fraction);
    const double zeta = -log_mp / std::sqrt(kPi * kPi + log_mp * log_mp);
    if (damping_ratio != nullptr) {
        *damping_ratio = std::min(1.0, std::max(0.05, zeta));
    }
    if (natural_frequency != nullptr) {
        // 2 percent settling band: ts ~= 4 / (zeta wn).
        const double settling = std::max(1e-9, specifications.settling_time);
        *natural_frequency = 4.0 / (zeta * settling);
    }
}

OpenLoopMargins ComputeMargins(const LtiSystem& system, const PidGains& gains,
                               const PidTuningOptions& options) {
    OpenLoopMargins margins;
    if (!system.is_continuous()) {
        throw std::invalid_argument(
            "ComputeMargins: the model must be continuous");
    }
    // A logarithmic scan over nine decades, then a linear interpolation for the
    // crossover. A coarse grid would report a margin that is off by degrees,
    // which matters when the margin is the design target.
    const int points = 4000;
    const double lowest = 1e-3;
    const double highest = 1e4;
    double previous_gain_db = 0.0;
    double previous_phase = 0.0;
    double previous_omega = 0.0;
    bool have_previous = false;
    for (int i = 0; i <= points; ++i) {
        const double omega = lowest * std::pow(highest / lowest,
                                               static_cast<double>(i) / points);
        const std::complex<double> plant =
            FrequencyResponse(system, omega)(0, 0);
        const std::complex<double> loop =
            PidResponseAt(gains, omega, options.derivative_filter_divisor) *
            plant;
        const double gain_db =
            20.0 * std::log10(std::max(1e-300, std::abs(loop)));
        double phase = std::arg(loop) * 180.0 / kPi;
        while (phase > 0.0) {
            phase -= 360.0;
        }
        if (have_previous) {
            // 0 dB crossing: |L| = 1.
            if (!margins.has_crossover &&
                ((previous_gain_db >= 0.0 && gain_db < 0.0) ||
                 (previous_gain_db <= 0.0 && gain_db > 0.0))) {
                const double fraction =
                    previous_gain_db / (previous_gain_db - gain_db);
                margins.crossover_frequency =
                    previous_omega * std::pow(omega / previous_omega, fraction);
                const double phase_there =
                    previous_phase + fraction * (phase - previous_phase);
                margins.phase_margin_degrees = 180.0 + phase_there;
                margins.has_crossover = true;
            }
            // -180 degree crossing: the gain margin.
            if (!margins.has_phase_crossover && previous_phase > -180.0 &&
                phase <= -180.0) {
                const double fraction =
                    (previous_phase + 180.0) / (previous_phase - phase);
                margins.phase_crossover_frequency =
                    previous_omega * std::pow(omega / previous_omega, fraction);
                const double gain_there =
                    previous_gain_db + fraction * (gain_db - previous_gain_db);
                margins.gain_margin_db = -gain_there;
                margins.has_phase_crossover = true;
            }
        }
        previous_gain_db = gain_db;
        previous_phase = phase;
        previous_omega = omega;
        have_previous = true;
    }
    return margins;
}

PidDesign TuneZieglerNichols(double ultimate_gain, double ultimate_period,
                             PidForm form, double derating) {
    if (!(ultimate_gain > 0.0) || !(ultimate_period > 0.0)) {
        throw std::invalid_argument(
            "TuneZieglerNichols: the ultimate gain and period must be "
            "positive");
    }
    if (!(derating > 0.0)) {
        throw std::invalid_argument(
            "TuneZieglerNichols: the derating factor must be positive");
    }
    PidDesign design;
    design.form = form;
    design.ultimate_gain = ultimate_gain;
    design.ultimate_period = ultimate_period;
    design.has_ultimate_values = true;
    design.method = "Ziegler-Nichols critical proportion";

    double kp = 0.0;
    double ti = std::numeric_limits<double>::infinity();
    double td = 0.0;
    switch (form) {
        case PidForm::kP:
            kp = 0.5 * ultimate_gain;
            break;
        case PidForm::kPi:
            kp = 0.45 * ultimate_gain;
            ti = ultimate_period / 2.0;
            break;
        case PidForm::kPid:
            kp = 0.6 * ultimate_gain;
            ti = ultimate_period / 2.0;
            td = ultimate_period / 8.0;
            break;
    }
    design.gains =
        PidGains::FromIntegralAndDerivativeTimes(kp * derating, ti, td);

    std::ostringstream notes;
    notes << "Ziegler-Nichols from Ku = " << ultimate_gain
          << ", Pu = " << ultimate_period << "; Ti = " << ti;
    if (td > 0.0) {
        notes << ", Td = " << td;
    }
    if (derating != 1.0) {
        notes << "; gains derated by " << derating;
    }
    // The table is deliberately aggressive (quarter-amplitude damping, which is
    // about 25 percent overshoot) and this is worth saying out loud rather than
    // leaving for the commissioning session to discover.
    notes << ". The classic table targets roughly quarter-amplitude damping: "
             "expect "
             "about 25 percent overshoot on the first step, and derate by 0.5 "
             "to 0.7 "
             "if that is too much";
    design.notes = notes.str();
    if (form != PidForm::kP) {
        design.warnings.push_back(
            "no model was supplied, so the closed-loop poles and the margins "
            "are "
            "unknown; use TuneZieglerNicholsFromModel or TunePolePlacement to "
            "have "
            "them computed");
    }
    design.ok = true;
    return design;
}

PidDesign TuneZieglerNicholsFromModel(const LtiSystem& system, PidForm form,
                                      double derating) {
    system.Validate("TuneZieglerNicholsFromModel");
    const RationalPlant plant = AsRational(system);
    if (!plant.valid) {
        throw std::invalid_argument("TuneZieglerNicholsFromModel: " +
                                    plant.why_not);
    }

    // Scan for the phase crossover: the frequency where the plant's phase is
    // -180 degrees. Its gain there is the ultimate gain, and the period
    // follows.
    const int points = 20000;
    const double lowest = 1e-4;
    const double highest = 1e6;
    double crossover_omega = 0.0;
    double gain_there = 0.0;
    double previous_phase = 0.0;
    double previous_omega = 0.0;
    double previous_gain = 0.0;
    bool have_previous = false;
    for (int i = 0; i <= points; ++i) {
        const double omega = lowest * std::pow(highest / lowest,
                                               static_cast<double>(i) / points);
        const std::complex<double> response =
            FrequencyResponse(system, omega)(0, 0);
        double phase = std::arg(response) * 180.0 / kPi;
        while (phase > 0.0) {
            phase -= 360.0;
        }
        const double magnitude = std::abs(response);
        if (have_previous && previous_phase > -180.0 && phase <= -180.0) {
            const double fraction =
                (previous_phase + 180.0) / (previous_phase - phase);
            crossover_omega =
                previous_omega * std::pow(omega / previous_omega, fraction);
            gain_there = previous_gain + fraction * (magnitude - previous_gain);
            break;
        }
        previous_phase = phase;
        previous_omega = omega;
        previous_gain = magnitude;
        have_previous = true;
    }
    if (!(crossover_omega > 0.0) || !(gain_there > 0.0)) {
        throw std::invalid_argument(
            "TuneZieglerNicholsFromModel: the plant's phase never reaches -180 "
            "degrees, so it has no ultimate gain and this method does not "
            "apply. "
            "Such a plant (a first-order lag, for instance) is easy to control "
            "- "
            "use TunePolePlacement, which is tied to the model rather than to "
            "an "
            "oscillation experiment");
    }

    PidDesign design = TuneZieglerNichols(
        1.0 / gain_there, 2.0 * kPi / crossover_omega, form, derating);
    design.method =
        "Ziegler-Nichols critical proportion, Ku and Pu from the model";
    // Now that the plant is known, the closed loop can be reported: this is the
    // reason to prefer this overload.
    FillPoleReport(
        &design,
        PolynomialAdd(PolynomialMultiply(Vector::Unit(2, 0), plant.denominator),
                      PolynomialMultiply((Vector(3) << design.gains.kd,
                                          design.gains.kp, design.gains.ki)
                                             .finished(),
                                         plant.numerator)));
    FillMargins(&design, system, PidTuningOptions{});
    design.notes +=
        ". Ku was found from the model at the phase crossover frequency " +
        std::to_string(crossover_omega) + " rad/s";
    return design;
}

PidDesign TunePolePlacement(const LtiSystem& system, PidForm form,
                            double damping_ratio, double natural_frequency,
                            const PidTuningOptions& options) {
    system.Validate("TunePolePlacement");
    if (!(damping_ratio > 0.0 && damping_ratio < 1.0)) {
        throw std::invalid_argument(
            "TunePolePlacement: the damping ratio must be in (0, 1); at 1 "
            "there is no "
            "oscillation to shape and above 1 the specification is "
            "unreachable");
    }
    if (!(natural_frequency > 0.0)) {
        throw std::invalid_argument(
            "TunePolePlacement: the natural frequency must be positive");
    }
    const RationalPlant plant = AsRational(system);
    if (!plant.valid) {
        throw std::invalid_argument("TunePolePlacement: " + plant.why_not);
    }
    const PlantShape shape = ShapeOf(plant);

    // Coefficient matching, written out. The closed-loop characteristic
    // polynomial of C(s) G(s) with C = (Kd s^2 + Kp s + Ki) / s is
    //   s D(s) + (Kd s^2 + Kp s + Ki) N(s),
    // and the two shapes below are the ones that can be matched exactly.
    PidDesign design;
    design.form = form;
    design.method = "pole placement by coefficient matching";

    if (shape.first_order && form == PidForm::kPi) {
        // G = b / (s + a0);  s(s + a0) + b(Kp s + Ki) = s^2 + 2 zeta wn s +
        // wn^2.
        const double a0 = -shape.poles(0).real();
        const double b = shape.dc_gain * a0;
        if (std::abs(b) < 1e-300) {
            throw std::invalid_argument(
                "TunePolePlacement: the plant has zero DC gain, so no PI gain "
                "can "
                "place the closed-loop poles");
        }
        design.gains.ki = natural_frequency * natural_frequency / b;
        design.gains.kp = (2.0 * damping_ratio * natural_frequency - a0) / b;
        design.gains.kd = 0.0;
        design.notes =
            "PI on a first-order plant: the closed loop is placed at the "
            "requested "
            "second-order pair exactly";
        // The characteristic polynomial is s D(s) + (Kp s + Ki) N(s), which is
        // s(s + a0) + b(Kp s + Ki) = s^2 + (a0 + b Kp) s + b Ki. The leading s
        // multiplies the plant denominator: leaving it out gives a polynomial
        // of the wrong degree, and the reported poles are then the roots of the
        // wrong equation even though the gains are right.
        const Vector s_times_denominator =
            PolynomialMultiply((Vector(2) << 1.0, 0.0).finished(),
                               (Vector(2) << 1.0, a0).finished());
        const Vector pi_numerator =
            (Vector(2) << design.gains.kp, design.gains.ki).finished();
        FillPoleReport(
            &design, PolynomialAdd(s_times_denominator,
                                   PolynomialMultiply(pi_numerator,
                                                      Vector::Constant(1, b))));
    } else if (shape.second_order && form == PidForm::kPid) {
        // G = b / (s^2 + a1 s + a0)
        //   s(s^2 + a1 s + a0) + b(Kd s^2 + Kp s + Ki)
        //     = s^3 + (a1 + b Kd) s^2 + (a0 + b Kp) s + b Ki
        // matched against (s^2 + 2 zeta wn s + wn^2)(s + p0).
        // a1 = -(p1 + p2) and a0 = p1 p2, the elementary symmetric functions.
        // Both are real for a real-coefficient polynomial, whether the pair is
        // real or conjugate.
        const double a1 = -(shape.poles(0) + shape.poles(1)).real();
        const double a0 = (shape.poles(0) * shape.poles(1)).real();
        const double b = shape.dc_gain * a0;
        if (std::abs(b) < 1e-300) {
            throw std::invalid_argument(
                "TunePolePlacement: the plant has zero DC gain, so no PID gain "
                "can "
                "place the closed-loop poles");
        }
        const double p0 = options.integrator_pole_ratio * natural_frequency;
        design.gains.ki = natural_frequency * natural_frequency * p0 / b;
        design.gains.kp = (natural_frequency * natural_frequency +
                           2.0 * damping_ratio * natural_frequency * p0 - a0) /
                          b;
        design.gains.kd =
            (2.0 * damping_ratio * natural_frequency + p0 - a1) / b;
        design.notes =
            "PID on a second-order plant: the closed-loop polynomial is "
            "matched "
            "against (s^2 + 2 zeta wn s + wn^2)(s + p0) with p0 = " +
            std::to_string(options.integrator_pole_ratio) + " wn";
        const Vector s_times_denominator = PolynomialMultiply(
            (Vector(2) << 1.0, 0.0).finished(), plant.denominator);
        const Vector pid_numerator =
            (Vector(3) << design.gains.kd, design.gains.kp, design.gains.ki)
                .finished();
        FillPoleReport(
            &design,
            PolynomialAdd(s_times_denominator,
                          PolynomialMultiply(pid_numerator, plant.numerator)));
    } else {
        std::ostringstream message;
        message
            << "TunePolePlacement: this route matches coefficients for a "
               "first-order plant with PI or a second-order plant with PID. ";
        message << "The model has " << shape.poles.size()
                << " pole(s) and the form is ";
        switch (form) {
            case PidForm::kP:
                message << "P";
                break;
            case PidForm::kPi:
                message << "PI";
                break;
            case PidForm::kPid:
                message << "PID";
                break;
        }
        message << ". ";
        if (shape.first_order) {
            message << "A first-order plant has no oscillatory mode to damp, "
                       "so the "
                       "derivative gain is redundant and the matching would be "
                       "underdetermined; use PI. ";
        }
        message
            << "For a higher-order plant, PlacePoles or "
               "LinearQuadraticRegulator "
               "handle it properly and this function will not approximate it";
        throw std::invalid_argument(message.str());
    }

    FillMargins(&design, system, options);

    // Simulate the design so the achieved step response is reported instead of
    // assumed. The characteristic polynomial is matched exactly, but the PID's
    // zeros are not placed by that matching, and they move the overshoot a long
    // way from the second-order formula - for a 10 percent request on a
    // second-order plant the achieved value is around 27 percent. Saying so is
    // the difference between a design note and a surprise on the test bench.
    {
        PidConfig prediction;
        prediction.gains = design.gains;
        // A sample time fast enough not to be the limiting factor, with the
        // limits wide open so the linear design is what is being measured.
        prediction.sample_time =
            std::min(0.001, 0.02 / std::max(1e-9, natural_frequency));
        prediction.output_min = -1e12;
        prediction.output_max = 1e12;
        design.predicted_step = SimulatePidStep(
            system, prediction, 1.0, 20.0 / std::max(1e-9, natural_frequency));
        design.has_predicted_step = true;
        if (design.predicted_step.overshoot_percent >
            1.2 * design.gains.kp * 0.0 + 20.0) {
            design.warnings.push_back(
                "the achieved overshoot is " +
                std::to_string(design.predicted_step.overshoot_percent) +
                " percent: pole placement matches the characteristic "
                "polynomial, but "
                "the controller's zeros still shape the response, so the "
                "requested "
                "damping ratio does not translate into the requested "
                "overshoot. Lower "
                "the natural frequency, raise the integrator pole ratio, or "
                "accept the "
                "value reported here");
        }
    }
    if (design.gains.kd < 0.0) {
        design.warnings.push_back(
            "the derivative gain is negative: the plant's own damping already "
            "exceeds what the specification asks for, so this specification is "
            "slower than the open loop. Pick a higher natural frequency or a "
            "lower "
            "damping ratio");
    }
    design.ok = true;
    return design;
}

PidController::PidController(const PidConfig& config) : config_(config) {
    if (!(config_.sample_time > 0.0)) {
        throw std::invalid_argument(
            "PidController: the sample time must be positive");
    }
    if (config_.output_min > config_.output_max) {
        throw std::invalid_argument(
            "PidController: output_min must not exceed output_max");
    }
    Reset();
}

void PidController::Reset() {
    integral_ = 0.0;
    derivative_state_ = 0.0;
    measurement_previous_ = 0.0;
    output_previous_ = 0.0;
    first_call_ = true;
    saturated_ = false;
    rate_limited_ = false;
}

void PidController::TrackManualOutput(double command) {
    // Bumpless transfer: the integrator adopts whatever it takes to reproduce
    // the command at the current operating point, so switching to automatic
    // does not move the actuator.
    integral_ = command;
    output_previous_ = command;
    first_call_ = true;
}

double PidController::Update(double measurement, double setpoint) {
    if (!std::isfinite(measurement) || !std::isfinite(setpoint)) {
        // Hold the last command rather than propagate a NaN into an actuator.
        return output_previous_;
    }
    const PidConfig& config = config_;
    const double error = setpoint - measurement;

    // Derivative on the measurement, negated: d/dt(-y). Acting on the error
    // instead would differentiate the setpoint and produce a spike on every
    // reference step.
    if (first_call_) {
        measurement_previous_ = measurement;
        first_call_ = false;
    }
    const double raw_derivative =
        -(measurement - measurement_previous_) / config.sample_time;
    measurement_previous_ = measurement;

    // Filter: D(s) = Kd s / (1 + s Td/N), discretized as a first-order lag.
    const double td =
        config.gains.kp != 0.0 ? config.gains.kd / config.gains.kp : 0.0;
    const double filter_time = config.derivative_filter_divisor > 0.0
                                   ? td / config.derivative_filter_divisor
                                   : 0.0;
    double derivative_input = config.gains.kd * raw_derivative;
    if (filter_time > 1e-12) {
        const double alpha = filter_time / (filter_time + config.sample_time);
        derivative_state_ =
            alpha * derivative_state_ + (1.0 - alpha) * derivative_input;
    } else {
        derivative_state_ = derivative_input;
    }
    derivative_input = derivative_state_;

    // Candidate command without integrator motion, used for the anti-windup
    // test.
    const double proportional = config.gains.kp * error;
    const double unsaturated = proportional + integral_ + derivative_input;
    const bool limit_active = config.output_min != config.output_max;
    const bool pushing_further =
        limit_active && ((unsaturated > config.output_max && error > 0.0) ||
                         (unsaturated < config.output_min && error < 0.0));

    if (!pushing_further) {
        integral_ += config.gains.ki * config.sample_time * error;
        // Second line of defence: the integral term is bounded even when the
        // output is not saturated, because a long error at a low gain can still
        // wind it up.
        const double integral_ceiling =
            config.integral_max != config.integral_min
                ? config.integral_max
                : (limit_active ? config.output_max
                                : std::numeric_limits<double>::max());
        const double integral_floor =
            config.integral_max != config.integral_min
                ? config.integral_min
                : (limit_active ? config.output_min
                                : -std::numeric_limits<double>::max());
        integral_ =
            std::min(integral_ceiling, std::max(integral_floor, integral_));
    }

    double command = proportional + integral_ + derivative_input;

    // Rate limit before the output limit: a command that is within range but
    // moving faster than the actuator can follow is just as damaging.
    rate_limited_ = false;
    if (config.rate_limit && config.rate_max > 0.0) {
        const double maximum_step = config.rate_max * config.sample_time;
        const double change = command - output_previous_;
        if (change > maximum_step) {
            command = output_previous_ + maximum_step;
            rate_limited_ = true;
        } else if (change < -maximum_step) {
            command = output_previous_ - maximum_step;
            rate_limited_ = true;
        }
    }

    saturated_ = false;
    if (limit_active) {
        const double limited =
            std::min(config.output_max, std::max(config.output_min, command));
        saturated_ = limited != command;
        command = limited;
    }
    output_previous_ = command;
    return command;
}

PidStepMetrics SimulatePidStep(const LtiSystem& continuous_system,
                               const PidConfig& config, double setpoint,
                               double duration) {
    continuous_system.Validate("SimulatePidStep");
    if (!continuous_system.is_continuous()) {
        throw std::invalid_argument(
            "SimulatePidStep: the model must be continuous");
    }
    if (!(duration > 0.0)) {
        throw std::invalid_argument(
            "SimulatePidStep: the duration must be positive");
    }
    const int n = continuous_system.state_dimension();
    if (continuous_system.output_dimension() != 1 ||
        continuous_system.input_dimension() != 1) {
        throw std::invalid_argument(
            "SimulatePidStep: only SISO models are supported");
    }
    PidController controller(config);
    Vector state = Vector::Zero(n);
    double previous_command = 0.0;
    PidStepMetrics metrics;

    const int ticks = static_cast<int>(duration / config.sample_time);
    Vector measurements(ticks);
    Vector commands(ticks);
    for (int tick = 0; tick < ticks; ++tick) {
        // The measurement uses the previous command through the D path. For the
        // usual D = 0 plant this is exact; for a plant with a direct
        // feedthrough it makes the loop algebraic, and the one-tick delay is
        // then part of the sampled model rather than an artefact.
        const double measurement =
            (continuous_system.c() * state)(0, 0) +
            continuous_system.d()(0, 0) * previous_command;
        const double command = controller.Update(measurement, setpoint);
        previous_command = command;
        measurements(tick) = measurement;
        commands(tick) = command;
        if (controller.saturated()) {
            ++metrics.saturated_ticks;
        }
        if (controller.rate_limited()) {
            ++metrics.rate_limited_ticks;
        }
        // Exact zero-order-hold step of the continuous plant over one period.
        const Matrix augmented = [&]() {
            Matrix m = Matrix::Zero(n + 1, n + 1);
            m.topLeftCorner(n, n) = continuous_system.a();
            m.topRightCorner(n, 1) = continuous_system.b();
            return m;
        }();
        const Matrix scaled = augmented * config.sample_time;
        const Matrix step = scaled.exp();
        Vector extended(n + 1);
        extended.head(n) = state;
        extended(n) = command;
        state = (step * extended).head(n);
    }

    metrics.peak_command = commands.cwiseAbs().maxCoeff();
    metrics.peak_measurement = measurements.cwiseAbs().maxCoeff();
    if (setpoint != 0.0) {
        metrics.overshoot_percent =
            std::max(0.0, (metrics.peak_measurement - setpoint) /
                              std::abs(setpoint) * 100.0);
    }
    metrics.steady_state_error =
        setpoint - measurements(measurements.size() - 1);
    // Settling time: the last moment the response leaves the 2 percent band.
    const double band = 0.02 * std::abs(setpoint);
    metrics.settled = false;
    for (int tick = ticks - 1; tick >= 0; --tick) {
        if (std::abs(measurements(tick) - setpoint) > band) {
            metrics.settling_time = (tick + 1) * config.sample_time;
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
