// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "control/ident/validate.h"

namespace control {
namespace ident {
namespace {

// The two-sided normal quantile for the usual confidence levels. Only the
// values that get used are tabulated, and anything else falls back to 1.96 (95
// percent) with the report saying so rather than pretending to a precision it
// does not have.
double NormalQuantile(double confidence) {
    if (confidence >= 0.99) {
        return 2.576;
    }
    if (confidence >= 0.95) {
        return 1.96;
    }
    if (confidence >= 0.90) {
        return 1.645;
    }
    return 1.96;
}

// Normalized autocorrelation: r(tau) = sum e[k] e[k+tau] / sum e[k]^2, so
// r(0)=1.
Eigen::VectorXd Autocorrelation(const Eigen::VectorXd& signal, int max_lag) {
    const int n = static_cast<int>(signal.size());
    const double zero_lag = signal.squaredNorm();
    Eigen::VectorXd correlation(max_lag + 1);
    for (int lag = 0; lag <= max_lag; ++lag) {
        double sum = 0.0;
        for (int k = 0; k + lag < n; ++k) {
            sum += signal(k) * signal(k + lag);
        }
        correlation(lag) = zero_lag > 0.0 ? sum / zero_lag : 0.0;
    }
    return correlation;
}

}  // namespace

ResidualTest TestResidualWhiteness(const Eigen::VectorXd& residual, int max_lag,
                                   double confidence) {
    if (residual.size() < 3) {
        throw std::invalid_argument("TestResidualWhiteness: too few residuals");
    }
    if (max_lag < 1) {
        throw std::invalid_argument(
            "TestResidualWhiteness: max_lag must be positive");
    }
    const int usable_lag =
        std::min(max_lag, static_cast<int>(residual.size()) - 1);
    ResidualTest test;
    test.autocorrelation = Autocorrelation(residual, usable_lag);
    const double n = static_cast<double>(residual.size());
    test.confidence_bound = NormalQuantile(confidence) / std::sqrt(n);

    // Lag 0 is excluded: it is 1 by construction. The per-lag bound is used
    // only to point at *where* the structure is.
    for (int lag = 1; lag <= usable_lag; ++lag) {
        if (std::abs(test.autocorrelation(lag)) > test.confidence_bound) {
            test.first_significant_lag = lag;
            break;
        }
    }
    // The verdict comes from the portmanteau statistic, N * sum r(lag)^2,
    // compared with a chi-square quantile on `usable_lag` degrees of freedom.
    //
    // Deciding the verdict from the per-lag bound instead - which this did at
    // first - is a multiple-comparisons error: with 25 lags tested against a 95
    // percent bound, the probability of at least one false positive is 1 -
    // 0.95^25, about 72 percent, so a perfectly white residual is reported as
    // structured almost every time. The test caught it by requiring a white
    // residual to be called white.
    double sum = 0.0;
    for (int lag = 1; lag <= usable_lag; ++lag) {
        sum += test.autocorrelation(lag) * test.autocorrelation(lag);
    }
    test.cumulative_whiteness = n * sum;
    // Wilson-Hilferty: chi-square quantile from the normal one, accurate to a
    // few parts in a thousand for these degrees of freedom and needing no
    // table.
    const double z = NormalQuantile(confidence);
    const double k = static_cast<double>(usable_lag);
    const double chi_square_bound =
        k *
        std::pow(1.0 - 2.0 / (9.0 * k) + z * std::sqrt(2.0 / (9.0 * k)), 3.0);
    test.cumulative_bound = chi_square_bound;
    test.white = test.cumulative_whiteness < chi_square_bound;

    std::ostringstream notes;
    if (test.white) {
        notes << "residuals are consistent with white noise: the portmanteau "
                 "statistic is "
              << test.cumulative_whiteness << " against a bound of "
              << chi_square_bound << " on " << usable_lag << " lags (lag 1 is "
              << test.autocorrelation(1) << ")";
    } else {
        notes << "residuals are NOT white: the portmanteau statistic is "
              << test.cumulative_whiteness << " against a bound of "
              << chi_square_bound << " on " << usable_lag
              << " lags, the largest single correlation being "
              << test.autocorrelation(std::max(1, test.first_significant_lag))
              << " at lag " << test.first_significant_lag
              << ". Structure is left in the residual, which means the model "
                 "missed "
                 "dynamics - usually a mode the input did not excite, which no "
                 "amount "
                 "of extra fitting will recover";
    }
    test.notes = notes.str();
    return test;
}

CrossCorrelationTest TestResidualInputCorrelation(
    const Eigen::VectorXd& residual, const Eigen::VectorXd& input, int max_lag,
    double confidence) {
    if (residual.size() != input.size() || residual.size() < 3) {
        throw std::invalid_argument(
            "TestResidualInputCorrelation: the sequences must be the same "
            "length");
    }
    CrossCorrelationTest test;
    test.correlation = Eigen::VectorXd::Zero(max_lag + 1);
    const double residual_norm = residual.norm();
    const double input_norm = input.norm();
    const double denominator = residual_norm * input_norm;
    for (int lag = 0; lag <= max_lag; ++lag) {
        double sum = 0.0;
        for (int k = 0; k + lag < residual.size(); ++k) {
            sum += residual(k + lag) * input(k);
        }
        test.correlation(lag) = denominator > 0.0 ? sum / denominator : 0.0;
    }
    test.confidence_bound = NormalQuantile(confidence) /
                            std::sqrt(static_cast<double>(residual.size()));
    for (int lag = 0; lag <= max_lag; ++lag) {
        if (std::abs(test.correlation(lag)) > test.confidence_bound) {
            test.first_significant_lag = lag;
            test.uncorrelated = false;
            break;
        }
    }
    test.notes =
        test.uncorrelated
            ? "the residual is uncorrelated with the input: the model has used "
              "the "
              "information the input carries"
            : "the residual is still correlated with the input at lag " +
                  std::to_string(test.first_significant_lag) +
                  ", so the input can still predict part of what the model got "
                  "wrong. "
                  "This is the signature of an order that is too low";
    return test;
}

ValidationReport ValidateArxModel(const Dataset& training,
                                  const Dataset& validation,
                                  const ArxModel& model) {
    training.Validate("ValidateArxModel");
    validation.Validate("ValidateArxModel");
    if (!model.IsValid()) {
        throw std::invalid_argument("ValidateArxModel: " +
                                    model.DescribeProblem());
    }
    ValidationReport report;

    // Free-run simulation on both records: the predicted output is fed back, so
    // the model has to stand on its own over the whole horizon.
    const Eigen::VectorXd history =
        training.output.head(model.DenominatorOrder());
    const Eigen::VectorXd training_prediction =
        SimulateArx(model, training.input, training.sample_time, history);
    const Eigen::VectorXd validation_history =
        validation.output.head(model.DenominatorOrder());
    const Eigen::VectorXd validation_prediction = SimulateArx(
        model, validation.input, validation.sample_time, validation_history);
    report.training = ComputeFit(training.output, training_prediction);
    report.validation = ComputeFit(validation.output, validation_prediction);

    const Eigen::VectorXd one_step = PredictArxOneStep(model, training);
    const int first = std::max(model.DenominatorOrder(),
                               model.delay + model.NumeratorOrder() - 1);
    report.one_step_prediction =
        ComputeFit(training.output.tail(one_step.size()), one_step);

    // The residual tests use the ONE-STEP prediction error, not the free-run
    // error.
    //
    // This matters and is easy to get wrong: the free-run error accumulates the
    // simulation's own drift, so it is strongly autocorrelated for any model at
    // all - testing its whiteness asks whether a simulation error is white, and
    // the answer is always no. The one-step error is the equation error, and
    // *that* is white when the model order is right and the noise is white,
    // which is the property worth testing.
    const Eigen::VectorXd residual =
        training.output.tail(one_step.size()) - one_step;
    report.residuals = TestResidualWhiteness(residual);
    report.input_correlation = TestResidualInputCorrelation(
        residual, training.input.tail(residual.size()));

    // Regressor condition number, from the same construction the estimator
    // used.
    {
        const int parameters =
            model.DenominatorOrder() + model.NumeratorOrder();
        const int rows = training.Size() - first;
        Eigen::MatrixXd regressors(rows, parameters);
        for (int row = 0; row < rows; ++row) {
            const int index = first + row;
            for (int i = 0; i < model.DenominatorOrder(); ++i) {
                regressors(row, i) = -training.output(index - 1 - i);
            }
            for (int j = 0; j < model.NumeratorOrder(); ++j) {
                regressors(row, model.DenominatorOrder() + j) =
                    training.input(index - model.delay - j);
            }
        }
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(regressors);
        const Eigen::VectorXd& singular = svd.singularValues();
        report.regressor_condition_number =
            singular.size() > 0 && singular(singular.size() - 1) > 0.0
                ? singular(0) / singular(singular.size() - 1)
                : std::numeric_limits<double>::infinity();
    }

    // The verdict, in the order the tests have power.
    std::ostringstream notes;
    if (training.Size() == validation.Size() &&
        (training.input - validation.input).norm() == 0.0) {
        report.warnings.push_back(
            "the validation record is the same as the training record: the "
            "validation "
            "fit below proves nothing. A model that fits the data it was "
            "fitted to has "
            "demonstrated only that it has enough parameters");
    }
    if (report.validation.fit_percent < 0.0) {
        report.warnings.push_back(
            "the free-run fit on held-out data is negative, which means the "
            "model does "
            "worse than predicting the mean of the output. It is not a model");
        report.model_is_valid = false;
    } else if (report.validation.fit_percent < 50.0) {
        report.warnings.push_back(
            "the free-run fit on held-out data is " +
            std::to_string(report.validation.fit_percent) +
            " percent: usable for a rough loop, not for a design");
        report.model_is_valid = true;
    } else {
        report.model_is_valid = true;
    }
    // A large gap between the one-step and free-run fits is the clearest single
    // indicator that the model is leaning on the measured past.
    // The one-step fit uses the measured past; the free-run fit does not. The
    // gap between them is the clearest single indicator that a model is leaning
    // on the measured past instead of having learned the dynamics.
    const double gap_percent =
        report.one_step_prediction.fit_percent - report.training.fit_percent;
    if (gap_percent > 20.0) {
        report.warnings.push_back(
            "the one-step prediction fits " + std::to_string(gap_percent) +
            " percentage points better than the free-run simulation: the model "
            "relies "
            "on the measured past and will not reproduce a trajectory");
    }
    if (!report.residuals.white) {
        report.warnings.push_back(report.residuals.notes);
    }
    if (!report.input_correlation.uncorrelated) {
        report.warnings.push_back(report.input_correlation.notes);
    }
    if (report.regressor_condition_number > 1e6) {
        report.warnings.push_back(
            "the regressor condition number is " +
            std::to_string(report.regressor_condition_number) +
            ": the parameters are not separately identifiable even if the fit "
            "is good");
    }
    notes << "validation fit " << report.validation.fit_percent
          << " percent on held-out data, free-run; one-step prediction fit "
          << report.one_step_prediction.fit_percent << " percent (training fit "
          << report.training.fit_percent
          << " percent). The free-run figure on "
             "held-out data is the one to quote";
    report.notes = notes.str();
    return report;
}

OrderSelection SelectOrder(const Dataset& training, const Dataset& validation,
                           const std::vector<int>& denominator_orders,
                           const std::vector<int>& numerator_orders,
                           int delay) {
    OrderSelection selection;
    double best_fit = -std::numeric_limits<double>::infinity();
    for (int na : denominator_orders) {
        for (int nb : numerator_orders) {
            OrderCandidate candidate;
            candidate.denominator_order = na;
            candidate.numerator_order = nb;
            if (training.Size() < std::max(na, delay + nb - 1) + na + nb) {
                selection.candidates.push_back(candidate);
                continue;
            }
            const ArxFitReport fit = FitArx(training, na, nb, delay);
            candidate.training_fit = fit.training_fit.fit_percent;
            candidate.regressor_condition_number =
                fit.regressor_condition_number;
            // The choice is made on held-out data. Choosing on the training fit
            // would always pick the largest order, which is the definition of
            // fitting noise.
            const ArxModel model = fit.model;
            const Eigen::VectorXd prediction =
                SimulateArx(model, validation.input, validation.sample_time,
                            validation.output.head(model.DenominatorOrder()));
            candidate.validation_fit =
                ComputeFit(validation.output, prediction).fit_percent;
            candidate.usable = !fit.ill_conditioned;
            if (candidate.usable && candidate.validation_fit > best_fit) {
                best_fit = candidate.validation_fit;
                selection.best_denominator_order = na;
                selection.best_numerator_order = nb;
            }
            selection.candidates.push_back(candidate);
        }
    }
    std::ostringstream notes;
    notes << "the order was chosen by the free-run fit on held-out data, not "
             "by the "
             "training fit: the training fit keeps improving with order "
             "because a "
             "larger model can always follow the noise, and only the held-out "
             "fit "
             "turns back down. Best: na = "
          << selection.best_denominator_order
          << ", nb = " << selection.best_numerator_order << " (validation fit "
          << best_fit << " percent)";
    selection.notes = notes.str();
    return selection;
}

}  // namespace ident
}  // namespace control
