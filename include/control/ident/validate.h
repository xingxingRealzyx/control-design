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

#ifndef CONTROL_IDENT_VALIDATE_H_
#define CONTROL_IDENT_VALIDATE_H_

// Deciding whether to believe the model.
//
// Three questions, in order of how often they catch a bad model:
//
//  1. Does it predict data it was not fitted to? (The only question that
//     matters. A model that fails here is wrong regardless of everything else.)
//  2. Are its residuals white? Leftover structure in the residuals means the
//     model missed dynamics; the fit can still be 99 percent while the
//     residuals are a clean sine wave, and that sine is the mode you did not
//     identify.
//  3. Are the parameters separately identifiable? A large regressor condition
//     number means the fit found a combination of parameters, not the
//     parameters, and a small change in the data will move them a long way.
//
// The order matters because the tests have different power. A whiteness test
// passes easily on a model that is wrong in a way the input never excited; a
// validation-set fit fails immediately.

#include <Eigen/Core>
#include <string>
#include <vector>

#include "control/ident/dataset.h"
#include "control/ident/ls.h"

namespace control {
namespace ident {

// Normalized autocorrelation of a residual sequence, with the confidence band
// for white noise (+/- 1.96/sqrt(N) at 95 percent). A residual that is white
// has no structure left for the model to explain.
struct ResidualTest {
    Eigen::VectorXd
        autocorrelation;  // lags 0..max_lag, normalised so lag 0 is 1
    double confidence_bound = 0.0;
    int first_significant_lag = -1;  // -1 when none is significant
    bool white = true;
    double cumulative_whiteness = 0.0;  // the portmanteau statistic
    // The chi-square bound that statistic is compared against. Exposed because
    // a verdict at 95 percent is by construction close to its bound on a finite
    // record, so a caller that wants a test with margin needs the number, not
    // the boolean: the last bits of a floating-point summation can flip a
    // verdict that sits on the bound.
    double cumulative_bound = 0.0;
    std::string notes;
};
ResidualTest TestResidualWhiteness(const Eigen::VectorXd& residual,
                                   int max_lag = 25, double confidence = 0.95);

// Cross-correlation between the residual and the input. If the input can still
// predict the residual, the model has not used all the information the input
// carries - which is the specific failure of an order that is too low, and it
// is invisible in a fit percentage.
struct CrossCorrelationTest {
    Eigen::VectorXd correlation;
    double confidence_bound = 0.0;
    int first_significant_lag = -1;
    bool uncorrelated = true;
    std::string notes;
};
CrossCorrelationTest TestResidualInputCorrelation(
    const Eigen::VectorXd& residual, const Eigen::VectorXd& input,
    int max_lag = 25, double confidence = 0.95);

struct ValidationReport {
    // Fit on the data the parameters were fitted to, and on data they never
    // saw.
    FitMetrics training;
    FitMetrics validation;
    // The one-step prediction fit, which is what the least-squares fit actually
    // optimises and which is always better than the free-run simulation. Both
    // are reported so the gap is visible: a large gap means the model leans on
    // the measured past and will not reproduce a trajectory.
    FitMetrics one_step_prediction;
    ResidualTest residuals;
    CrossCorrelationTest input_correlation;
    double regressor_condition_number = 0.0;
    // The verdict, and the reasons for it, in the order the tests were applied.
    bool model_is_valid = false;
    std::vector<std::string> warnings;
    std::string notes;
};

// The full battery: free-run fit on both records, one-step fit, residual
// whiteness, residual-input correlation, and the condition number. `validation`
// may be the same record as `training`, in which case the report says loudly
// that the validation number proves nothing.
ValidationReport ValidateArxModel(const Dataset& training,
                                  const Dataset& validation,
                                  const ArxModel& model);

// Order selection by validation fit: fits a range of orders on the training
// record and reports the validation fit of each, so the choice is made on
// held-out data rather than on the fit the estimator optimised. This is the
// procedure that keeps the order from being chosen to fit the noise.
struct OrderCandidate {
    int denominator_order = 0;
    int numerator_order = 0;
    double training_fit = 0.0;
    double validation_fit = 0.0;
    double regressor_condition_number = 0.0;
    bool usable = false;
};
struct OrderSelection {
    std::vector<OrderCandidate> candidates;
    // The order with the best validation fit among the usable candidates.
    int best_denominator_order = 0;
    int best_numerator_order = 0;
    std::string notes;
};
OrderSelection SelectOrder(const Dataset& training, const Dataset& validation,
                           const std::vector<int>& denominator_orders,
                           const std::vector<int>& numerator_orders,
                           int delay = 1);

}  // namespace ident
}  // namespace control

#endif  // CONTROL_IDENT_VALIDATE_H_
