// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <stdexcept>

#include "control/ident/rls.h"

namespace control {
namespace ident {

RecursiveLeastSquares::RecursiveLeastSquares(int parameter_count,
                                             const RlsOptions& options)
    : options_(options) {
    if (parameter_count < 1) {
        throw std::invalid_argument(
            "RecursiveLeastSquares: no parameters to estimate");
    }
    if (!(options_.forgetting_factor > 0.0 &&
          options_.forgetting_factor <= 1.0)) {
        throw std::invalid_argument(
            "RecursiveLeastSquares: the forgetting factor must be in (0, 1]; "
            "above one "
            "the estimator diverges, and at or below zero it forgets "
            "everything");
    }
    if (!(options_.initial_covariance > 0.0)) {
        throw std::invalid_argument(
            "RecursiveLeastSquares: the initial covariance must be positive; "
            "it is the "
            "estimator's statement about how little it knows at the start");
    }
    parameters_ = Eigen::VectorXd::Zero(parameter_count);
    Reset();
}

void RecursiveLeastSquares::Reset() {
    parameters_.setZero();
    covariance_ =
        Eigen::MatrixXd::Identity(parameters_.size(), parameters_.size()) *
        options_.initial_covariance;
    samples_ = 0;
    worst_change_ = 0.0;
}

RlsStep RecursiveLeastSquares::Update(const Eigen::VectorXd& regressor,
                                      double output) {
    if (regressor.size() != parameters_.size()) {
        throw std::invalid_argument(
            "RecursiveLeastSquares::Update: regressor size mismatch");
    }
    if (!regressor.allFinite() || !std::isfinite(output)) {
        // A non-finite sample must not enter the covariance update: one
        // infinity poisons the estimate permanently and there is no recovery
        // path.
        RlsStep step;
        step.parameters = parameters_;
        step.covariance_trace = covariance_.trace();
        return step;
    }
    const double lambda = options_.forgetting_factor;
    const Eigen::VectorXd phi_times_p = covariance_ * regressor;
    const double denominator = lambda + regressor.dot(phi_times_p);
    // A zero regressor (or a vanishing one) carries no information; dividing by
    // the resulting denominator would be dividing by lambda, which is harmless,
    // but the update is empty and skipping it keeps the covariance from being
    // needlessly contracted by samples that say nothing.
    if (denominator <= 1e-300 || regressor.norm() < 1e-300) {
        RlsStep step;
        step.parameters = parameters_;
        step.covariance_trace = covariance_.trace();
        return step;
    }
    const Eigen::VectorXd gain = phi_times_p / denominator;
    const double prediction_error = output - regressor.dot(parameters_);
    const Eigen::VectorXd update = gain * prediction_error;
    parameters_ += update;
    covariance_ = (covariance_ - gain * phi_times_p.transpose()) / lambda;
    // Keep it symmetric: the recursion is symmetric in exact arithmetic and
    // drifts otherwise, and an asymmetric covariance eventually produces a
    // non-physical gain.
    covariance_ = 0.5 * (covariance_ + covariance_.transpose());
    ++samples_;
    const double change = update.norm();
    worst_change_ = std::max(worst_change_, change);

    if (options_.covariance_reset_threshold > 0.0 &&
        covariance_.trace() <
            options_.covariance_reset_threshold * options_.initial_covariance) {
        // The estimator has stopped learning (with a forgetting factor it can
        // also go to sleep); resetting the covariance says "look again".
        covariance_ =
            Eigen::MatrixXd::Identity(parameters_.size(), parameters_.size()) *
            options_.initial_covariance;
    }

    RlsStep step;
    step.parameters = parameters_;
    step.prediction_error = prediction_error;
    step.parameter_change = change;
    step.covariance_trace = covariance_.trace();
    return step;
}

RlsStep RunRecursiveLeastSquares(const Dataset& data, int denominator_order,
                                 int numerator_order, int delay,
                                 const RlsOptions& options, ArxModel* model) {
    data.Validate("RunRecursiveLeastSquares");
    const int first = std::max(denominator_order, delay + numerator_order - 1);
    RecursiveLeastSquares estimator(denominator_order + numerator_order,
                                    options);
    RlsStep step;
    for (int index = first; index < data.Size(); ++index) {
        step = estimator.Update(ArxRegressor(data, index, denominator_order,
                                             numerator_order, delay),
                                data.output(index));
    }
    if (model != nullptr) {
        model->a = step.parameters.head(denominator_order);
        model->b = step.parameters.tail(numerator_order);
        model->delay = delay;
        model->sample_time = data.sample_time;
    }
    return step;
}

}  // namespace ident
}  // namespace control
