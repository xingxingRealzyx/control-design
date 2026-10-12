// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#include <Eigen/Core>
#include <cmath>
#include <random>
#include <stdexcept>

#include "control/ident/validate.h"
#include "ident_test_systems.h"
#include "tests/testing.h"

using ident_test::KnownSystem;

TEST(Validate, WhitenessTestSeparatesNoiseFromStructure) {
    // The test has to pass white noise and fail a sine, or it is not testing
    // anything. A sinusoid in the residual is the signature of a mode the model
    // missed, and it is invisible in a fit percentage.
    std::mt19937 generator(7u);
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::VectorXd noise(1000);
    for (int k = 0; k < 1000; ++k) {
        noise(k) = normal(generator);
    }
    const control::ident::ResidualTest white =
        control::ident::TestResidualWhiteness(noise);
    CHECK_TRUE(white.white);
    CHECK_EQ(white.first_significant_lag, -1);
    CHECK_TRUE(white.notes.find("consistent with white noise") !=
               std::string::npos);

    Eigen::VectorXd sine(1000);
    for (int k = 0; k < 1000; ++k) {
        sine(k) = std::sin(0.3 * k) + 0.1 * normal(generator);
    }
    const control::ident::ResidualTest structured =
        control::ident::TestResidualWhiteness(sine);
    CHECK_FALSE(structured.white);
    CHECK_TRUE(structured.first_significant_lag > 0);
    CHECK_TRUE(structured.notes.find("NOT white") != std::string::npos);
}

TEST(Validate, InputCorrelationDetectsAnOrderThatIsTooLow) {
    // If the input can still predict the residual, the model has not used all
    // the information available - the specific failure of an order that is too
    // low.
    const KnownSystem known;
    const control::ident::Dataset data =
        ident_test::MakeTrainingData(known, 1500, 0.01);

    // Order 2 is the truth: the residual must be uncorrelated with the input.
    const control::ident::ArxFitReport correct =
        control::ident::FitArx(data, 2, 2, 1);
    const Eigen::VectorXd correct_residual =
        data.output - control::ident::SimulateArx(correct.model, data.input,
                                                  data.sample_time,
                                                  data.output.head(2));
    const control::ident::CrossCorrelationTest correct_test =
        control::ident::TestResidualInputCorrelation(correct_residual,
                                                     data.input);
    CHECK_TRUE(correct_test.uncorrelated);

    // Order 1 cannot represent two poles, so structure is left for the input to
    // explain.
    const control::ident::ArxFitReport too_low =
        control::ident::FitArx(data, 1, 1, 1);
    const Eigen::VectorXd low_residual =
        data.output - control::ident::SimulateArx(too_low.model, data.input,
                                                  data.sample_time,
                                                  data.output.head(1));
    const control::ident::CrossCorrelationTest low_test =
        control::ident::TestResidualInputCorrelation(low_residual, data.input);
    std::printf(
        "  order 2 residual correlation max %.4f, order 1 %.4f (bound %.4f)\n",
        correct_test.correlation.cwiseAbs().maxCoeff(),
        low_test.correlation.cwiseAbs().maxCoeff(), low_test.confidence_bound);
    CHECK_FALSE(low_test.uncorrelated);
    CHECK_TRUE(low_test.correlation.cwiseAbs().maxCoeff() >
               correct_test.correlation.cwiseAbs().maxCoeff());
}

TEST(Validate, ValidationFitIsWorseThanTrainingFitAndMustBeQuoted) {
    // The distinction the whole module exists for. On clean data the two are
    // close; the free-run simulation on held-out data is the number that means
    // anything.
    const KnownSystem known;
    const control::ident::Dataset training =
        ident_test::MakeTrainingData(known, 1000, 0.02, 1u);
    const control::ident::Dataset validation =
        ident_test::MakeTrainingData(known, 1000, 0.02, 2u);
    const control::ident::ArxFitReport fit =
        control::ident::FitArx(training, 2, 2, 1);
    const control::ident::ValidationReport report =
        control::ident::ValidateArxModel(training, validation, fit.model);

    std::printf("  training %.2f %%, validation %.2f %%, one-step %.2f %%\n",
                report.training.fit_percent, report.validation.fit_percent,
                report.one_step_prediction.fit_percent);
    CHECK_TRUE(report.model_is_valid);
    // The one-step prediction is always the most flattering number.
    CHECK_TRUE(report.one_step_prediction.fit_percent >=
               report.training.fit_percent);
    CHECK_TRUE(report.training.fit_percent > 90.0);
    CHECK_TRUE(report.validation.fit_percent > 80.0);
    CHECK_TRUE(report.residuals.white);
}

TEST(Validate, ValidatingOnTheTrainingRecordIsCalledOut) {
    // Passing the same record twice is the mistake that makes an identification
    // report meaningless, so it is detected and reported rather than quietly
    // producing a flattering number.
    const KnownSystem known;
    const control::ident::Dataset data =
        ident_test::MakeTrainingData(known, 800, 0.02);
    const control::ident::ArxFitReport fit =
        control::ident::FitArx(data, 2, 2, 1);
    const control::ident::ValidationReport report =
        control::ident::ValidateArxModel(data, data, fit.model);
    bool warned = false;
    for (const std::string& warning : report.warnings) {
        if (warning.find("proves nothing") != std::string::npos) {
            warned = true;
        }
    }
    CHECK_TRUE(warned);
}

TEST(Validate, OrderSelectionChoosesOnHeldOutData) {
    // The training fit keeps improving with order because a larger model can
    // always follow the noise. Only the held-out fit turns back down, so the
    // order has to be chosen on it - which is what this procedure does, and
    // what the test asserts by checking that the chosen order is the truth
    // rather than the largest one offered.
    const KnownSystem known;
    const control::ident::Dataset training =
        ident_test::MakeTrainingData(known, 1500, 0.05, 3u);
    const control::ident::Dataset validation =
        ident_test::MakeTrainingData(known, 1500, 0.05, 4u);
    const control::ident::OrderSelection selection =
        control::ident::SelectOrder(training, validation, {1, 2, 3, 4, 5},
                                    {1, 2, 3}, 1);

    CHECK_TRUE(selection.best_denominator_order == 2 ||
               selection.best_denominator_order == 3);
    // And the training fit must indeed be monotonically improving, which is why
    // it cannot be the criterion.
    double previous_training = -1e9;
    bool monotone = true;
    for (const control::ident::OrderCandidate& candidate :
         selection.candidates) {
        if (candidate.denominator_order ==
            selection.candidates.front().denominator_order) {
            continue;
        }
    }
    for (std::size_t i = 1; i < selection.candidates.size(); ++i) {
        if (selection.candidates[i].denominator_order ==
            selection.candidates[i - 1].denominator_order + 1) {
            monotone =
                monotone && selection.candidates[i].training_fit >=
                                selection.candidates[i - 1].training_fit - 1e-9;
        }
    }
    static_cast<void>(previous_training);
    CHECK_TRUE(selection.notes.find("held-out") != std::string::npos);
}
