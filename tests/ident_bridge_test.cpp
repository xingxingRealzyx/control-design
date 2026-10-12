// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0

// The pipeline, end to end: data -> identified model -> designed controller.
//
// This is the test that justifies the identification module living in the
// design repository at all. Everything before it verifies a stage; this
// verifies that the stages connect, which is where a sample time gets dropped
// or a sign flips.

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>

#include "control/analysis.h"
#include "control/codegen.h"
#include "control/design.h"
#include "control/ident/bridge.h"
#include "control/ident/ls.h"
#include "control/lti.h"
#include "ident_test_systems.h"
#include "tests/testing.h"

using ident_test::KnownSystem;

TEST(IdentBridge, IdentifiedModelSurvivesTheTripIntoTheDesignLibrary) {
    // Identify, convert, and then check that the design library sees the same
    // system: the poles and the DC gain must be unchanged by the conversion. A
    // conversion that silently drops the sample time or transposes a matrix
    // produces a model that is still a model - and describes a different plant.
    const KnownSystem known;
    const control::ident::Dataset data =
        ident_test::MakeTrainingData(known, 2000, 0.0);
    const control::ident::ArxFitReport fit =
        control::ident::FitArx(data, 2, 2, 1);
    const control::LtiSystem model = control::ident::ToLtiSystem(fit.model);

    CHECK_TRUE(model.is_discrete());
    CHECK_NEAR(model.sample_time(), known.sample_time, 0.0);
    CHECK_EQ(model.state_dimension(), 2);
    CHECK_EQ(model.input_dimension(), 1);
    CHECK_EQ(model.output_dimension(), 1);

    // The poles the design library reports are the ones the identification
    // found.
    const control::ComplexVector library_poles = control::Poles(model);
    const Eigen::VectorXcd identified_poles =
        control::ident::ArxPoles(fit.model);
    CHECK_EQ(library_poles.size(), identified_poles.size());
    for (Eigen::Index i = 0; i < library_poles.size(); ++i) {
        CHECK_NEAR(std::abs(library_poles(i)), std::abs(identified_poles(i)),
                   1e-12);
    }
    // And the DC gain matches the difference equation's, computed
    // independently.
    const double dc_from_arx =
        std::abs(control::ident::ArxFrequencyResponse(fit.model, 0.0));
    CHECK_NEAR(control::DcGain(model)(0, 0), dc_from_arx, 1e-9);
}

TEST(IdentBridge, AnIdentifiedModelCanBeDesignedAgainst) {
    // The point of the whole exercise: the model that came out of data goes
    // into the LQR, and the resulting controller is stable. Nothing here is
    // special-cased for identified models - it is the same design path a
    // hand-derived model would take, which is the claim being tested.
    const KnownSystem known;
    // Noise, because a real identification always has some, and the design has
    // to survive the parameters moving a little.
    const control::ident::Dataset data =
        ident_test::MakeTrainingData(known, 4000, 0.02);
    const control::ident::ArxFitReport fit =
        control::ident::FitArx(data, 2, 2, 1);
    const control::LtiSystem model = control::ident::ToLtiSystem(fit.model);

    // Assess before designing: an identified model can fit held-out data well
    // and still be unusable for design.
    const control::ident::IdentifiedModelReport assessment =
        control::ident::AssessIdentifiedModel(model);
    CHECK_TRUE(assessment.fit_for_design);
    CHECK_TRUE(assessment.controllable);
    CHECK_TRUE(assessment.observable);
    CHECK_EQ(assessment.warnings.size(), 0u);

    control::Matrix q = control::Matrix::Identity(2, 2);
    control::Matrix r = control::Matrix::Constant(1, 1, 0.1);
    const control::DesignResult design =
        control::DiscreteLinearQuadraticRegulator(model, q, r);
    CHECK_TRUE(design.ok);
    // The closed loop must be stable. The gain is not compared against a
    // reference value because the model came from noisy data and the parameters
    // moved; what is checked is the property a design has to have.
    for (Eigen::Index i = 0; i < design.closed_loop_poles.size(); ++i) {
        CHECK_TRUE(std::abs(design.closed_loop_poles(i)) < 1.0);
    }
    // And the controller can be emitted for the target, from the identified
    // model.
    control::CodegenOptions options;
    options.controller_name = "IdentifiedController";
    options.namespace_name = "identified";
    const control::CodegenResult generated =
        control::GenerateController(model, design, options);
    CHECK_TRUE(generated.header.find("using Real = double;") !=
               std::string::npos);
    CHECK_EQ(generated.state_dimension, 2);
}

TEST(IdentBridge, AnUnstableModelIsFlaggedBeforeDesigning) {
    // An identified model with a pole outside the unit circle: the assessment
    // must say so, and say what it usually means, rather than letting the
    // design proceed and fail later with a message about controllability.
    control::ident::ArxModel model;
    model.a.resize(2);
    model.a << -2.2, 1.1;  // 1 + a0 z^-1 + a1 z^-2 = 1 - 2.2 z^-1 + 1.1 z^-2
    model.b.resize(2);
    model.b << 0.5, 0.2;
    model.delay = 1;
    model.sample_time = 0.1;
    const control::LtiSystem system = control::ident::ToLtiSystem(model);
    const control::ident::IdentifiedModelReport assessment =
        control::ident::AssessIdentifiedModel(system);
    CHECK_FALSE(assessment.stable);
    CHECK_FALSE(assessment.fit_for_design);
    CHECK_TRUE(assessment.warnings.size() > 0);
    CHECK_TRUE(assessment.notes.find("NOT fit to design against") !=
               std::string::npos);
    // The warning must mention that an unstable pole is usually an
    // identification problem rather than an unstable plant, because that is the
    // actionable part.
    CHECK_TRUE(assessment.warnings[0].find("identification") !=
               std::string::npos);
}

TEST(IdentBridge, RefusesModelsThatAreNotUsableAtAll) {
    control::ident::StateSpaceModel broken;
    broken.a = control::Matrix::Identity(2, 2);
    broken.b = control::Matrix::Zero(2, 1);
    broken.c = control::Matrix::Zero(1, 2);
    broken.d = control::Matrix::Zero(1, 1);
    broken.sample_time = 0.0;  // an identified model always has one
    CHECK_THROWS(control::ident::ToLtiSystem(broken));

    broken.sample_time = 0.1;
    broken.a = control::Matrix::Zero(2, 3);  // not square
    CHECK_THROWS(control::ident::ToLtiSystem(broken));

    broken.a = control::Matrix::Identity(2, 2);
    broken.b = control::Matrix::Zero(3, 1);  // wrong number of rows
    CHECK_THROWS(control::ident::ToLtiSystem(broken));

    // A non-proper ARX model has no state-space form at all: the numerator
    // would have to differentiate the input.
    control::ident::ArxModel non_proper;
    non_proper.a.resize(1);
    non_proper.a << -0.5;
    non_proper.b.resize(3);
    non_proper.b << 1.0, 1.0, 1.0;
    non_proper.delay = 1;  // delay + nb - 1 = 3 > na = 1
    non_proper.sample_time = 0.1;
    CHECK_THROWS(control::ident::ToStateSpace(non_proper));
}
