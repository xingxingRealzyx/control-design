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

#ifndef EXAMPLES_POSITIONING_DESIGN_H_
#define EXAMPLES_POSITIONING_DESIGN_H_

// The design used by both the code generator and the cross-check test, in one
// place on purpose: if the generator and the reference implementation were fed
// different numbers, the comparison between them would prove nothing.

#include "control/analysis.h"
#include "control/design.h"
#include "control/discretize.h"
#include "control/lti.h"

namespace positioning {

// A positioning stage: a mass on a spring-damper, driven by a force.
//   m x'' = -k x - c x' + F,   states [position, velocity], input force.
// m = 1 kg, k = 4 N/m, c = 0.4 N*s/m. The open-loop poles are at -0.2 and
// -2 rad/s, so the sample time follows from the slower one.
inline control::LtiSystem ContinuousPlant() {
    const double mass = 1.0;
    const double stiffness = 4.0;
    const double damping = 0.4;
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -stiffness / mass, -damping / mass;
    control::Matrix b(2, 1);
    b << 0.0, 1.0 / mass;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;  // position is measured
    return control::LtiSystem::Continuous(a, b, c);
}

inline double SampleTime() {
    // 20 samples per period of the fastest open-loop pole (2 rad/s).
    return control::RequiredSampleTime(2.0, 20.0);
}

inline control::LtiSystem DiscretePlant() {
    return control::Discretize(ContinuousPlant(), SampleTime());
}

inline control::Matrix StateWeight() {
    control::Matrix q(2, 2);
    q << 10.0, 0.0, 0.0, 1.0;
    return q;
}

inline control::Matrix IntegralWeight() {
    control::Matrix qi(1, 1);
    qi << 5.0;
    return qi;
}

inline control::Matrix InputWeight() {
    control::Matrix r(1, 1);
    r << 1.0;
    return r;
}

// Force limit of the actuator, in newtons.
inline double ForceLimit() { return 20.0; }

inline control::CodegenOptions Options() {
    control::CodegenOptions options;
    options.controller_name = "PositioningController";
    options.namespace_name = "positioning";
    options.real_type = "double";
    options.saturate = true;
    options.input_min = -ForceLimit();
    options.input_max = ForceLimit();
    options.provenance =
        "DLQR with integral action on the position error; Q = diag(10, 1), "
        "Qi = 5, R = 1, zero-order hold at the sample time below";
    options.state_units = {"m", "m/s"};
    options.input_unit = "N";
    options.output_unit = "m";
    return options;
}

// The full design, so the generator and the test cannot disagree about it.
inline control::AugmentedDesign Design() {
    return control::AddIntegralActionLqr(DiscretePlant(), 0, StateWeight(),
                                         IntegralWeight(), InputWeight());
}

}  // namespace positioning

#endif  // EXAMPLES_POSITIONING_DESIGN_H_
