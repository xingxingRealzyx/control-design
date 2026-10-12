// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#include <Eigen/Core>
#include <cmath>
#include <stdexcept>

#include "control/ident/rls.h"
#include "ident_test_systems.h"
#include "tests/testing.h"

using ident_test::KnownSystem;

TEST(Rls, WithNoForgettingMatchesTheBatchLeastSquaresSolution) {
    // The identity that makes the recursion trustworthy: with a forgetting
    // factor of one, the recursive estimate and the batch least-squares
    // estimate agree up to the influence of the initial covariance, which is
    // (Phi'Phi + P0^-1)^-1. A large P0 makes that influence negligible, so the
    // agreement is a numerical statement rather than a vague one - and the test
    // states the reason instead of just allowing a loose tolerance.
    const KnownSystem known;
    const control::ident::Dataset data =
        ident_test::MakeTrainingData(known, 1200, 0.02);

    control::ident::RlsOptions options;
    options.forgetting_factor = 1.0;
    options.initial_covariance = 1e12;
    control::ident::ArxModel recursive_model;
    const control::ident::RlsStep final_step =
        control::ident::RunRecursiveLeastSquares(data, 2, 2, 1, options,
                                                 &recursive_model);

    const control::ident::ArxFitReport batch =
        control::ident::FitArx(data, 2, 2, 1);

    // The parameter vector, assembled the same way for both.
    Eigen::VectorXd recursive(4);
    recursive << recursive_model.a, recursive_model.b;
    Eigen::VectorXd batch_parameters(4);
    batch_parameters << batch.model.a, batch.model.b;
    const double difference =
        (recursive - batch_parameters).cwiseAbs().maxCoeff();
    std::printf("  recursive vs batch: max parameter difference %.3e\n",
                difference);
    // 1e-6 rather than round-off: the residual influence of P0 = 1e12 is of
    // order 1e-12 on the normal matrix, amplified by the conditioning of the
    // problem.
    CHECK_TRUE(difference < 1e-6);
    CHECK_TRUE(final_step.covariance_trace > 0.0);
}

TEST(Rls, ForgettingTracksAParameterChangeThatBatchLeastSquaresCannot) {
    // The reason to use a forgetting factor at all. The plant changes half way
    // through the record; a batch fit reports the average of the two plants,
    // which describes neither, while a recursive fit with forgetting follows
    // the change.
    const KnownSystem before;
    KnownSystem after = before;
    after.b1 = 1.5;  // a large gain change, as if a valve were opened
    const control::ident::Dataset first =
        ident_test::MakeTrainingData(before, 1200, 0.0, 1u);
    const control::ident::Dataset second =
        ident_test::MakeTrainingData(after, 1200, 0.0, 2u);
    control::ident::Dataset combined;
    combined.sample_time = first.sample_time;
    combined.input.resize(first.Size() + second.Size());
    combined.output.resize(first.Size() + second.Size());
    combined.input << first.input, second.input;
    combined.output << first.output, second.output;

    // Batch: one answer for both halves.
    const control::ident::ArxFitReport batch =
        control::ident::FitArx(combined, 2, 2, 1);
    // Recursive with forgetting.
    control::ident::RlsOptions options;
    options.forgetting_factor = 0.99;
    options.initial_covariance = 1e6;
    control::ident::ArxModel tracked;
    control::ident::RunRecursiveLeastSquares(combined, 2, 2, 1, options,
                                             &tracked);

    // The batch gain sits between the two, the tracked one is near the new
    // value.
    const double batch_gain = batch.model.b(0);
    const double tracked_gain = tracked.b(0);
    std::printf(
        "  batch b1 = %.4f (between %.1f and %.1f), tracked b1 = %.4f\n",
        batch_gain, before.b1, after.b1, tracked_gain);
    CHECK_TRUE(batch_gain > before.b1 + 0.2);
    CHECK_TRUE(batch_gain < after.b1 - 0.2);
    CHECK_TRUE(std::abs(tracked_gain - after.b1) < 0.1);
}

TEST(Rls, RecoversTheCoefficientsFromCleanData) {
    const KnownSystem known;
    const control::ident::Dataset data =
        ident_test::MakeTrainingData(known, 2000, 0.0);
    control::ident::ArxModel model;
    control::ident::RunRecursiveLeastSquares(data, 2, 2, 1, {}, &model);
    CHECK_NEAR(model.a(0), known.a1, 1e-6);
    CHECK_NEAR(model.a(1), known.a2, 1e-6);
    CHECK_NEAR(model.b(0), known.b1, 1e-6);
    CHECK_NEAR(model.b(1), known.b2, 1e-6);
}

TEST(Rls, RefusesForgettingFactorsOutsideTheirRange) {
    control::ident::RlsOptions too_large;
    too_large.forgetting_factor = 1.5;
    CHECK_THROWS(control::ident::RecursiveLeastSquares(4, too_large));
    control::ident::RlsOptions zero;
    zero.forgetting_factor = 0.0;
    CHECK_THROWS(control::ident::RecursiveLeastSquares(4, zero));
    CHECK_THROWS(control::ident::RecursiveLeastSquares(0));
}

TEST(Rls, NonFiniteSampleLeavesTheEstimateAlone) {
    // One infinity in a covariance update poisons the estimate permanently;
    // there is no recovery path once P has an infinity in it.
    control::ident::RecursiveLeastSquares estimator(2);
    Eigen::VectorXd regressor(2);
    regressor << 0.5, -0.3;
    estimator.Update(regressor, 1.0);
    const Eigen::VectorXd before = estimator.parameters();
    regressor << std::numeric_limits<double>::quiet_NaN(), 0.0;
    estimator.Update(regressor, 1.0);
    CHECK_NEAR((estimator.parameters() - before).cwiseAbs().maxCoeff(), 0.0,
               0.0);
    CHECK_TRUE(estimator.parameters().allFinite());
}
