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

#ifndef CONTROL_IDENT_BRIDGE_H_
#define CONTROL_IDENT_BRIDGE_H_

// Identification to design: the one conversion that makes the two halves one
// pipeline.
//
// This header is the reason the identification module lives in this repository
// rather than beside it. Without it, an identified model is four matrices that
// have to be copied by hand into whatever the design code expects - which is
// where the sample time gets dropped, the sign of D gets flipped, or the
// identified model and the designed controller quietly end up describing
// different plants.
//
//   data -> IdentifyArx / RealizeFromImpulseResponse -> model
//        -> ToLtiSystem                                -> LtiSystem
//        -> AnalyseControllability / AddIntegralActionLqr / GenerateController
//
// The conversion checks that the identified model is usable *for design* before
// handing it over, because a model that fits held-out data well can still be
// uncontrollable at the input you have, or have a pole outside the unit circle,
// and finding that out at the design step with a clear message beats finding it
// out from a diverging machine.

#include <string>
#include <vector>

#include "control/ident/ls.h"
#include "control/ident/subspace.h"
#include "control/lti.h"

namespace control {
namespace ident {

// Converts an identified model to the design library's representation. The
// sample time travels with the model: an identified model is discrete, and a
// design built on the wrong sample time is wrong by that factor everywhere.
//
// Throws when the model cannot be used for design at all - a non-positive
// sample time, or a non-finite coefficient - rather than returning something
// that will fail later with a less informative message.
LtiSystem ToLtiSystem(const StateSpaceModel& model);

// The same, from an ARX model, converting through its state-space form.
LtiSystem ToLtiSystem(const ArxModel& model);

// A short report on whether the identified model is fit to design against:
// stability, controllability, observability, and the DC gain. Written for a
// person deciding whether to proceed, so it says what is wrong and what it
// implies rather than printing numbers.
struct IdentifiedModelReport {
    bool stable = false;
    bool controllable = false;
    bool observable = false;
    double stability_margin = 0.0;
    double dc_gain = 0.0;
    ComplexVector poles;
    bool fit_for_design = false;
    std::vector<std::string> warnings;
    std::string notes;
};
IdentifiedModelReport AssessIdentifiedModel(const LtiSystem& model);

}  // namespace ident
}  // namespace control

#endif  // CONTROL_IDENT_BRIDGE_H_
