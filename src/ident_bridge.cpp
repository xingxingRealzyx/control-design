// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "control/analysis.h"
#include "control/ident/bridge.h"

namespace control {
namespace ident {

LtiSystem ToLtiSystem(const StateSpaceModel& model) {
    if (model.a.rows() != model.a.cols() || model.a.rows() == 0) {
        throw std::invalid_argument(
            "ToLtiSystem: A must be square and non-empty; the identified model "
            "is not "
            "a state-space model");
    }
    if (!(model.sample_time > 0.0)) {
        throw std::invalid_argument(
            "ToLtiSystem: the sample time must be positive. An identified "
            "model is "
            "discrete and the sample time is part of it - a design built on "
            "the wrong "
            "one is wrong by that factor in every pole and every gain");
    }
    const int n = static_cast<int>(model.a.rows());
    if (model.b.rows() != n || model.c.cols() != n ||
        model.b.cols() != model.d.rows() || model.c.rows() != model.d.cols()) {
        throw std::invalid_argument(
            "ToLtiSystem: the four matrices do not agree on their dimensions");
    }
    if (!model.a.allFinite() || !model.b.allFinite() || !model.c.allFinite() ||
        !model.d.allFinite()) {
        throw std::invalid_argument(
            "ToLtiSystem: a matrix entry is not finite");
    }
    return LtiSystem(model.a, model.b, model.c, model.d, model.sample_time);
}

LtiSystem ToLtiSystem(const ArxModel& model) {
    return ToLtiSystem(ToStateSpace(model));
}

IdentifiedModelReport AssessIdentifiedModel(const LtiSystem& model) {
    model.Validate("AssessIdentifiedModel");
    IdentifiedModelReport report;
    if (!model.is_discrete()) {
        report.notes =
            "the model is continuous: an identified model is normally "
            "discrete, so "
            "check that this is what you meant before designing against it";
    }
    report.stable = IsStable(model);
    report.stability_margin = StabilityMargin(model);
    report.poles = Poles(model);
    report.dc_gain = DcGain(model)(0, 0);

    const ControllabilityResult controllability = AnalyseControllability(model);
    const ObservabilityResult observability = AnalyseObservability(model);
    report.controllable = controllability.fully_controllable;
    report.observable = observability.fully_observable;

    // The three ways an identified model fails to be designable, each with what
    // it implies. An identification can be excellent and the model still
    // unusable: the fit says nothing about whether the actuator can move the
    // modes.
    if (!report.stable) {
        report.warnings.push_back(
            "the identified model is unstable (stability margin " +
            std::to_string(report.stability_margin) +
            "). Either the plant is genuinely unstable - in which case design "
            "a "
            "stabilising controller and check that the poles are real, not an "
            "artefact - or the identification went wrong, which is more "
            "common. An "
            "unstable pole from identification usually means the input did not "
            "excite "
            "the plant enough for the estimator to separate the slow mode from "
            "a "
            "drift");
    }
    if (!report.controllable) {
        report.warnings.push_back(
            "the identified model is not fully controllable: " +
            DescribeControllability(controllability) +
            ". A pole placement or LQR design will refuse this model, and it "
            "is right "
            "to: the unreachable states cannot be moved");
    }
    if (!report.observable) {
        report.warnings.push_back(
            "the identified model is not fully observable: " +
            DescribeObservability(observability) +
            ". An observer cannot estimate what the sensor does not see");
    }
    if (controllability.marginal) {
        report.warnings.push_back(
            "controllability is marginal (condition number " +
            std::to_string(controllability.rank.condition_number()) +
            "): a design will ask for very large gains. This is often an "
            "identification artefact rather than a property of the plant - the "
            "estimator has put a nearly-unreachable mode in the model");
    }

    report.fit_for_design = report.stable && report.controllable;
    std::ostringstream notes;
    notes << "identified model: " << model.state_dimension() << " states, "
          << (report.stable ? "stable" : "UNSTABLE") << ", margin "
          << report.stability_margin << ", DC gain " << report.dc_gain << ", "
          << (report.controllable ? "controllable" : "NOT controllable") << ", "
          << (report.observable ? "observable" : "NOT observable");
    if (report.fit_for_design) {
        notes << ". Fit to design against";
    } else {
        notes << ". NOT fit to design against - fix the identification before "
                 "designing, because a controller designed on a wrong model is "
                 "worse "
                 "than no controller: it moves the actuator with confidence";
    }
    report.notes = notes.str();
    return report;
}

}  // namespace ident
}  // namespace control
