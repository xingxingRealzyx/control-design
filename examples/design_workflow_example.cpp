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

// Example: the design half of the workflow, on a DC motor.
//
// What it shows, in the order a design actually happens:
//
//   1. the model, and what the model *is* (poles, zeros, DC gain, margin);
//   2. the discretization, with the sample time derived from a rule rather than
//      picked by feel, and the deviation from the continuous behaviour measured
//      instead of assumed;
//   3. what the discretization did to the zeros, which is where sampled-data
//      design bites;
//   4. the numbers a firmware engineer actually needs, printed as a table.
//
// The controller design itself (LQR, pole placement) is the next module; this
// example stops at the point where the library's analysis is what decides
// whether a design is even worth attempting.

#include <Eigen/Core>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "control/analysis.h"
#include "control/codegen.h"
#include "control/design.h"
#include "control/discretize.h"
#include "control/lti.h"

namespace {

control::LtiSystem DcMotor() {
    // Armature-controlled DC motor, states [current, speed], input voltage:
    //   di/dt = -(R/L) i - (Ke/L) w + u/L
    //   dw/dt =  (Kt/J) i - (b/J) w
    // with R = 1 ohm, L = 0.5 mH, Ke = Kt = 0.05, J = 1e-5, b = 1e-6.
    const double resistance = 1.0;
    const double inductance = 5e-4;
    const double back_emf = 0.05;
    const double torque_constant = 0.05;
    const double inertia = 1e-5;
    const double friction = 1e-6;

    control::Matrix a(2, 2);
    a << -resistance / inductance, -back_emf / inductance,
        torque_constant / inertia, -friction / inertia;
    control::Matrix b(2, 1);
    b << 1.0 / inductance, 0.0;
    control::Matrix c(1, 2);
    c << 0.0, 1.0;  // speed is measured
    return control::LtiSystem::Continuous(a, b, c);
}

void PrintPoleTable(const control::LtiSystem& system) {
    const control::PoleCharacteristics characteristics =
        control::DescribePoles(system);
    std::cout << "  pole            damping    frequency [rad/s]\n";
    for (Eigen::Index i = 0; i < characteristics.poles.size(); ++i) {
        std::cout << "  " << std::setw(10) << std::fixed << std::setprecision(2)
                  << characteristics.poles(i).real();
        if (std::abs(characteristics.poles(i).imag()) > 1e-12) {
            std::cout << " +/- " << std::setw(8)
                      << std::abs(characteristics.poles(i).imag()) << "j";
        } else {
            std::cout << std::setw(13) << " ";
        }
        std::cout << std::setw(10) << std::setprecision(3)
                  << characteristics.damping_ratios(i) << std::setw(14)
                  << characteristics.natural_frequencies(i) << "\n";
    }
}

}  // namespace

int main() {
    std::cout << std::fixed;
    const control::LtiSystem motor = DcMotor();
    std::cout << "DC motor: " << motor << "\n\n";

    // ---- 1. what the model is -------------------------------------------
    std::cout << "=== the model ===\n";
    PrintPoleTable(motor);
    // A first-order-plus-lag plant: one slow pole (the mechanics), one fast
    // electrical pole. The ratio between them is what makes this model stiff
    // and what sets the sample rate below.
    std::cout << "  stability margin: " << std::setprecision(1)
              << control::StabilityMargin(motor) << " rad/s\n"
              << "  DC gain:          " << std::setprecision(4)
              << control::DcGain(motor)(0, 0) << " rad/s per volt\n";

    const control::PoleZeroResult pole_zero = control::PolesAndZeros(motor);
    std::cout << "  finite zeros:     "
              << (pole_zero.has_zeros ? std::to_string(pole_zero.zeros.size())
                                      : "none")
              << "\n";

    // ---- 2. sample time from a rule -------------------------------------
    // The mechanical pole is the bandwidth a speed loop will be designed
    // around, so the sample rate follows from it: 20 samples per period.
    const double bandwidth = std::abs(control::Poles(motor)(0).real());
    const double sample_time = control::RequiredSampleTime(bandwidth, 20.0);
    std::cout << "\n=== discretization ===\n"
              << "  mechanical bandwidth: " << std::setprecision(2) << bandwidth
              << " rad/s (" << bandwidth / (2.0 * M_PI) << " Hz)\n"
              << "  sample time from RequiredSampleTime(bw, 20) = "
              << std::setprecision(6) << sample_time << " s ("
              << 1.0 / sample_time << " Hz)\n";

    const control::DiscretizationResult result =
        control::Convert(motor, sample_time);
    std::cout << "  method: " << control::ToString(result.report.method) << "\n"
              << "  exact for a piecewise-constant input: "
              << (result.report.exact_for_constant_input ? "yes" : "no") << "\n"
              << "  step-response deviation from continuous: "
              << std::scientific << std::setprecision(2)
              << result.report.step_response_deviation << std::fixed << "\n"
              << "  discrete poles: ";
    const control::ComplexVector discrete_poles = control::Poles(result.system);
    for (Eigen::Index i = 0; i < discrete_poles.size(); ++i) {
        std::cout << std::setprecision(6) << std::abs(discrete_poles(i)) << " ";
    }
    std::cout << "(continuous poles mapped through exp(-p Ts))\n";

    // ---- 3. what sampling did to the zeros ------------------------------
    std::cout
        << "\n=== what sampling did ===\n"
           "  The continuous model has no finite zeros. Zero-order hold adds\n"
           "  sampling zeros, and for a plant this fast relative to the "
           "sample\n"
           "  time they can land outside the unit circle - a minimum-phase "
           "plant\n"
           "  becoming non-minimum-phase is normal once sampled, not a "
           "mistake.\n"
           "  This is why the library refuses to report discrete zeros from "
           "the\n"
           "  continuous pencil: it would report the wrong ones.\n";
    try {
        control::PolesAndZeros(result.system);
        std::cout << "  (this should not be reachable)\n";
    } catch (const std::invalid_argument& problem) {
        std::cout << "  PolesAndZeros(discrete) refuses, as documented: "
                  << problem.what() << "\n";
    }

    // ---- 4. what can be done with it ------------------------------------
    std::cout << "\n=== controllability and observability ===\n";
    const control::ControllabilityResult controllability =
        control::AnalyseControllability(result.system);
    const control::ObservabilityResult observability =
        control::AnalyseObservability(result.system);
    std::cout << "  controllability: "
              << control::DescribeControllability(controllability)
              << "\n  observability:   "
              << control::DescribeObservability(observability) << "\n";

    // ---- 5. design ------------------------------------------------------
    // An LQR with integral action on the speed error, so a constant speed
    // reference is tracked without offset. The weights are the design choice;
    // everything else follows from them.
    std::cout << "\n=== design: DLQR with integral action ===\n";
    control::Matrix state_weight = control::Matrix::Zero(2, 2);
    state_weight << 1.0, 0.0, 0.0, 0.1;  // speed matters more than current
    control::Matrix integral_weight = control::Matrix::Zero(1, 1);
    integral_weight << 10.0;
    control::Matrix input_weight = control::Matrix::Zero(1, 1);
    input_weight << 1.0;
    const control::AugmentedDesign design = control::AddIntegralActionLqr(
        result.system, 0, state_weight, integral_weight, input_weight);
    std::cout << control::FormatDesignSummary(
        design.design, "gain on [current; speed; integral of speed error]:");

    // ---- 6. the numbers that go into firmware ---------------------------
    std::cout
        << "\n=== the model to deploy ===\n"
        << "  Ad = \n"
        << result.system.a() << "\n  Bd = \n"
        << result.system.b() << "\n  Cd = \n"
        << result.system.c() << "\n  Ts = " << result.system.sample_time()
        << " s\n\n"
           "  These four matrices are the complete interface: a firmware loop\n"
           "  that runs x[k+1] = Ad x[k] + Bd u[k] at this sample time "
           "reproduces\n"
           "  the continuous behaviour at every sample instant, exactly.\n";

    // ---- 7. emit the controller -----------------------------------------
    // The gain above is a design; this is the artefact that runs on the target.
    // The emitted header is compiled by the test suite and compared against an
    // independent implementation over 3000 ticks, so it is not taken on trust.
    std::cout << "\n=== generated controller ===\n";
    control::CodegenOptions codegen;
    codegen.controller_name = "MotorSpeedController";
    codegen.namespace_name = "motor";
    codegen.real_type =
        "float";  // a typical MCU has no reason to use double here
    codegen.saturate = true;
    codegen.input_min = -24.0;  // supply voltage limit
    codegen.input_max = 24.0;
    codegen.provenance =
        "DLQR with integral action on the speed error, from "
        "design_workflow_example";
    codegen.state_units = {"A", "rad/s"};
    codegen.input_unit = "V";
    codegen.output_unit = "rad/s";
    const control::CodegenResult generated =
        control::GenerateControllerWithIntegral(result.system, design, codegen);
    std::cout << "  " << generated.suggested_filename << ": "
              << generated.header.size() << " bytes, states "
              << generated.state_dimension << ", observer "
              << (generated.has_observer ? "yes" : "no") << ", integral action "
              << (generated.has_integral_action ? "yes" : "no") << "\n\n"
              << generated.usage_example << "\n  Write generated.header to "
              << generated.suggested_filename
              << " and compile it with the\n  firmware: it needs <array>, "
                 "<cmath> and "
                 "<cstddef>, nothing else.\n";
    return 0;
}
