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

#ifndef CONTROL_CODEGEN_H_
#define CONTROL_CODEGEN_H_

// Stage 9: emit the design as deployable C++17.
//
// The output is a self-contained header holding one controller class, written
// for the target rather than for the host:
//
//   * no exceptions, no RTTI, no dynamic allocation, no recursion;
//   * all state in the object, so several instances can run side by side;
//   * dimensions are compile-time constants, so every loop has a fixed trip
//     count and the per-tick cost is constant;
//   * `Real` is a single typedef at the top, so switching the target to float
//     is one edit and not a search-and-replace;
//   * every gain carries a comment saying which design produced it, because a
//     number in firmware with no provenance is a number nobody dares change;
//   * saturation and a NaN guard on the input, because the alternative is a
//     controller that happily drives an actuator to full scale on a sensor
//     glitch.
//
// The emitted code is not trusted on the strength of having been generated: the
// test suite compiles it and runs it against an independent implementation over
// a long input sequence (the cross-check gate in the README), because a
// transcription error in an unrolled matrix loop is exactly the kind of mistake
// that compiles cleanly and behaves almost right.

#include <Eigen/Core>
#include <string>
#include <vector>

#include "control/design.h"
#include "control/types.h"

namespace control {

struct CodegenOptions {
    // Class name and namespace of the emitted controller.
    std::string controller_name = "Controller";
    std::string namespace_name = "generated";
    // "double" or "float". The emitted header defines `using Real = ...`.
    std::string real_type = "double";
    // Actuator limits, in the input's unit. Saturation is emitted whenever
    // `saturate` is set; the limits are then also the clipping bounds.
    bool saturate = false;
    double input_min = 0.0;
    double input_max = 0.0;
    // Description of where the design came from. Emitted as a comment; it is
    // the only record the firmware engineer has of how these numbers were
    // chosen.
    std::string provenance;
    // Units of the state, input and output, for the emitted comments. The
    // generator cannot know them, and "x[0] in rad" is worth more than "x[0]".
    std::vector<std::string> state_units;
    std::string input_unit;
    std::string output_unit;
};

struct CodegenResult {
    // The complete header, ready to write to a file.
    std::string header;
    // A short usage example, emitted as a comment at the end of the header and
    // also returned separately so a report can quote it.
    std::string usage_example;
    // Names of the files the caller would normally write.
    std::string suggested_filename;
    int state_dimension = 0;
    int input_dimension = 0;
    int output_dimension = 0;
    bool has_observer = false;
    bool has_integral_action = false;
    // Every gain embedded in the header, for an audit trail.
    Matrix state_feedback_gain;
    Matrix observer_gain;
    Matrix integral_gain;
};

// State feedback u = -K x (plus optional integral action and optional
// observer).
//
// `system` must be the discrete model the controller will run against: the
// emitted code steps its observer with the same Ad, Bd the plant uses, and
// pairing a controller with a different sample time than the plant is the
// classic way a design that simulated perfectly fails on hardware.
//
// `design` supplies K (the whole gain matrix is emitted). When `observer` is
// non-null its gain L is emitted too, and the controller takes output
// measurements instead of the full state.
CodegenResult GenerateController(const LtiSystem& system,
                                 const DesignResult& design,
                                 const CodegenOptions& options,
                                 const DesignResult* observer = nullptr);

// The same, for a design with integral action: the gain acts on [x; z] and the
// emitted controller integrates the tracking error as the design assumed.
CodegenResult GenerateControllerWithIntegral(
    const LtiSystem& system, const AugmentedDesign& augmented_design,
    const CodegenOptions& options, const DesignResult* observer = nullptr);

// Human-readable table of the numbers in a result, for a report or a commit
// message: gains, poles, damping, margins.
std::string FormatDesignSummary(const DesignResult& result,
                                const std::string& label);

}  // namespace control

#endif  // CONTROL_CODEGEN_H_
