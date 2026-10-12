// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#include <Eigen/Core>
#include <cmath>
#include <stdexcept>

#include "control/ident/subspace.h"
#include "ident_test_systems.h"
#include "tests/testing.h"

using ident_test::KnownSystem;

TEST(Subspace, RealizesThePolesOfAKnownSystemFromItsImpulseResponse) {
    // The decisive test: an impulse response generated from a system whose
    // poles are known, realized back into state space, and the realized poles
    // compared with the truth. The Hankel singular values decide the order
    // without anyone trying orders until the fit stops improving.
    const KnownSystem known;
    const control::ident::ArxModel truth = ident_test::Truth(known);
    const Eigen::VectorXd impulse =
        control::ident::ArxImpulseResponse(truth, 120);

    control::ident::EraOptions options;
    options.block_rows = 20;
    const control::ident::EraResult result =
        control::ident::RealizeFromImpulseResponse(impulse, known.sample_time,
                                                   options);

    std::printf("  order %d (gap suggests %d), largest singular value %.4f\n",
                result.order, result.order_from_largest_gap,
                result.largest_singular_value);
    CHECK_TRUE(result.ok);
    CHECK_EQ(result.order, 2);

    const Eigen::VectorXcd truth_poles = control::ident::ArxPoles(truth);
    const Eigen::VectorXcd realized_poles =
        control::ident::StateSpacePoles(result.model);
    CHECK_EQ(realized_poles.size(), 2);
    for (Eigen::Index i = 0; i < 2; ++i) {
        CHECK_NEAR(std::abs(realized_poles(i)), std::abs(truth_poles(i)), 1e-6);
        CHECK_NEAR(realized_poles(i).real(), truth_poles(i).real(), 1e-6);
        CHECK_NEAR(realized_poles(i).imag(), truth_poles(i).imag(), 1e-6);
    }
}

TEST(Subspace, RealizedModelReproducesTheImpulseResponseItCameFrom) {
    // The realization is a different coordinate system for the same system, so
    // its impulse response must match the data it was built from. A realization
    // that gets the poles right and the residues wrong passes a pole check and
    // fails this one.
    const KnownSystem known;
    const Eigen::VectorXd impulse =
        control::ident::ArxImpulseResponse(ident_test::Truth(known), 150);
    control::ident::EraOptions options;
    options.block_rows = 20;
    options.order = 2;
    const control::ident::EraResult result =
        control::ident::RealizeFromImpulseResponse(impulse, known.sample_time,
                                                   options);
    const Eigen::VectorXd reproduced =
        control::ident::StateSpaceImpulseResponse(result.model, 150);
    const double worst = (impulse - reproduced).cwiseAbs().maxCoeff();
    std::printf("  impulse response reproduction: worst error %.3e\n", worst);
    CHECK_TRUE(worst < 1e-9);
}

TEST(Subspace, SingularValuesRevealTheOrderOfAHigherOrderSystem) {
    // A fourth-order system: the singular value sequence must show a clear drop
    // after four, which is how the order is determined without guessing.
    control::ident::ArxModel truth;
    truth.a.resize(4);
    truth.a << -2.2, 2.1, -1.0, 0.2;  // poles inside the unit circle
    truth.b.resize(4);
    truth.b << 0.1, 0.2, 0.15, 0.05;
    truth.delay = 1;
    truth.sample_time = 0.05;
    const Eigen::VectorXd impulse =
        control::ident::ArxImpulseResponse(truth, 200);

    control::ident::EraOptions options;
    options.block_rows = 30;
    const control::ident::EraResult result =
        control::ident::RealizeFromImpulseResponse(impulse, truth.sample_time,
                                                   options);
    std::printf("  singular values:");
    for (int i = 0;
         i < std::min<int>(
                 8, static_cast<int>(result.hankel_singular_values.size()));
         ++i) {
        std::printf(" %.3e", result.hankel_singular_values(i));
    }
    std::printf("\n  order from threshold %d, from the largest gap %d\n",
                result.order, result.order_from_largest_gap);
    CHECK_EQ(result.order, 4);
    CHECK_EQ(result.order_from_largest_gap, 4);
}

TEST(Subspace, RefusesRequestsItCannotServe) {
    const Eigen::VectorXd impulse = Eigen::VectorXd::Zero(10);
    control::ident::EraOptions options;
    options.block_rows = 20;  // needs 41 Markov parameters
    CHECK_THROWS(
        control::ident::RealizeFromImpulseResponse(impulse, 0.1, options));

    control::ident::EraOptions too_many_rows;
    too_many_rows.block_rows = 4;
    too_many_rows.order = 50;
    CHECK_THROWS(control::ident::RealizeFromImpulseResponse(
        control::ident::ArxImpulseResponse(ident_test::Truth(KnownSystem{}),
                                           40),
        0.1, too_many_rows));

    control::ident::EraOptions bad_sample;
    bad_sample.block_rows = 3;
    CHECK_THROWS(control::ident::RealizeFromImpulseResponse(
        control::ident::ArxImpulseResponse(ident_test::Truth(KnownSystem{}),
                                           40),
        0.0, bad_sample));
}
