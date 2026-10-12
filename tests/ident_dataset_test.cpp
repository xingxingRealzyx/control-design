// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#include <Eigen/Core>
#include <cmath>
#include <stdexcept>

#include "control/ident/dataset.h"
#include "tests/testing.h"

TEST(Dataset, FitIsMeasuredAgainstTheMeanNotAgainstZero) {
    // fit = 100 (1 - ||y - y_hat|| / ||y - mean(y)||). The denominator is the
    // deviation from the mean, so an all-zero prediction scores zero - "as good
    // as the average" - rather than a large negative number. That is the
    // baseline a fit percentage has to be measured against.
    Eigen::VectorXd measured(5);
    measured << 1.0, 2.0, 3.0, 4.0, 5.0;
    const control::ident::FitMetrics exact =
        control::ident::ComputeFit(measured, measured);
    CHECK_NEAR(exact.fit_percent, 100.0, 1e-9);
    CHECK_NEAR(exact.rmse, 0.0, 0.0);

    Eigen::VectorXd mean_prediction =
        Eigen::VectorXd::Constant(5, measured.mean());
    const control::ident::FitMetrics mean =
        control::ident::ComputeFit(measured, mean_prediction);
    CHECK_NEAR(mean.fit_percent, 0.0, 1e-9);

    // Worse than the mean is negative, which is the signal that the model is
    // not a model.
    Eigen::VectorXd worse = Eigen::VectorXd::Constant(5, 0.0);
    CHECK_TRUE(control::ident::ComputeFit(measured, worse).fit_percent < 0.0);
}

TEST(Dataset, SplitIsContiguousNotInterleaved) {
    // The split takes the first part for training and the second for
    // validation. Interleaving samples would look fairer and be worse:
    // consecutive samples of a dynamic system are correlated, so an interleaved
    // validation set shares most of its information with the training set.
    control::ident::Dataset source;
    source.sample_time = 0.1;
    source.input = Eigen::VectorXd::LinSpaced(100, 0.0, 99.0);
    source.output = Eigen::VectorXd::LinSpaced(100, 100.0, 199.0);

    control::ident::Dataset training;
    control::ident::Dataset validation;
    control::ident::SplitDataset(source, 0.6, &training, &validation);
    CHECK_EQ(training.Size(), 60);
    CHECK_EQ(validation.Size(), 40);
    CHECK_NEAR(training.input(0), 0.0, 0.0);
    CHECK_NEAR(training.input(59), 59.0, 0.0);
    CHECK_NEAR(validation.input(0), 60.0, 0.0);
    // No overlap: every validation sample comes after every training sample.
    CHECK_TRUE(validation.input.minCoeff() > training.input.maxCoeff());
}

TEST(Dataset, ValidationDescribesWhatIsWrong) {
    control::ident::Dataset data;
    data.sample_time = 0.1;
    data.input = Eigen::VectorXd::Zero(10);
    data.output = Eigen::VectorXd::Zero(9);
    CHECK_FALSE(data.IsValid());
    CHECK_TRUE(data.DescribeProblem().find("9") != std::string::npos);
    CHECK_THROWS(data.Validate("test"));

    data.output = Eigen::VectorXd::Zero(10);
    CHECK_TRUE(data.IsValid());

    data.sample_time = 0.0;
    CHECK_FALSE(data.IsValid());
    CHECK_THROWS(data.Validate("test"));

    data.sample_time = 0.1;
    data.output(3) = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(data.IsValid());
    CHECK_TRUE(data.DescribeProblem().find("non-finite") != std::string::npos);
}

TEST(Dataset, SplitRefusesFractionsThatLeaveNothing) {
    control::ident::Dataset source;
    source.sample_time = 0.1;
    source.input = Eigen::VectorXd::Zero(10);
    source.output = Eigen::VectorXd::Zero(10);
    control::ident::Dataset training;
    control::ident::Dataset validation;
    CHECK_THROWS(
        control::ident::SplitDataset(source, 1.0, &training, &validation));
    CHECK_THROWS(
        control::ident::SplitDataset(source, 0.0, &training, &validation));
}
