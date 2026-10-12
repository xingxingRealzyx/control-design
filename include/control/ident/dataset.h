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

#ifndef CONTROL_IDENT_DATASET_H_
#define CONTROL_IDENT_DATASET_H_

// The input/output record, and the split that makes identification honest.
//
// The single most important thing in this header is `Split`. A model that fits
// the data it was fitted to has demonstrated nothing: any model with enough
// parameters does that. The only evidence that a model is a model - rather than
// a curve drawn through the samples - is that it predicts data it has never
// seen. So identification here is written as a function of a *training* record,
// and the validation entry points take a second record the fit never touched.

#include <Eigen/Core>
#include <string>
#include <vector>

namespace control {
namespace ident {

struct Dataset {
    // Sampled at a constant rate, one entry per sample, same length.
    Eigen::VectorXd input;
    Eigen::VectorXd output;
    double sample_time = 0.0;
    // Names for reports, optional.
    std::string input_name;
    std::string output_name;

    int Size() const { return static_cast<int>(input.size()); }
    bool IsValid() const;
    // Describes what is wrong, or returns an empty string. A sentence rather
    // than a boolean, because "the input is 3 samples longer than the output"
    // is actionable and "invalid" is not.
    std::string DescribeProblem() const;
    void Validate(const std::string& context) const;

    // A contiguous portion, for hold-out and for the usual train/test split.
    Dataset Slice(int begin, int end) const;
};

// Splits a record in two, taking the *first* part as training and the second as
// validation rather than interleaving samples. Interleaving looks like a fairer
// split and is worse: consecutive samples of a dynamic system are correlated,
// so an interleaved validation set shares most of its information with the
// training set and the reported fit is optimistic.
void SplitDataset(const Dataset& source, double training_fraction,
                  Dataset* training, Dataset* validation);

// How well a model reproduces a record, on the scale a control engineer reads:
//   fit = 100 * (1 - ||y - y_hat|| / ||y - mean(y)||)
// so 100 is exact and 0 is "no better than predicting the average". The mean,
// not the norm, is the denominator: a model that outputs zero is not 100
// percent wrong, it is as good as the mean, which is the honest baseline.
struct FitMetrics {
    double fit_percent = 0.0;
    double rmse = 0.0;                    // in the output's units
    double variance_accounted_for = 0.0;  // percent, VAF
    double worst_absolute_error = 0.0;
    int samples = 0;
};
FitMetrics ComputeFit(const Eigen::VectorXd& measured,
                      const Eigen::VectorXd& predicted);

}  // namespace ident
}  // namespace control

#endif  // CONTROL_IDENT_DATASET_H_
