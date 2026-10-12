// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "control/ident/dataset.h"

namespace control {
namespace ident {

bool Dataset::IsValid() const { return DescribeProblem().empty(); }

std::string Dataset::DescribeProblem() const {
    if (input.size() != output.size()) {
        return "the input has " + std::to_string(input.size()) +
               " samples and the output has " + std::to_string(output.size()) +
               "; they must be recorded together";
    }
    if (input.size() < 2) {
        return "fewer than two samples: there is nothing to identify";
    }
    if (!(sample_time > 0.0)) {
        return "the sample time must be positive";
    }
    if (!input.allFinite()) {
        return "the input contains a non-finite sample";
    }
    if (!output.allFinite()) {
        return "the output contains a non-finite sample";
    }
    return std::string();
}

void Dataset::Validate(const std::string& context) const {
    const std::string problem = DescribeProblem();
    if (!problem.empty()) {
        throw std::invalid_argument(context + ": " + problem);
    }
}

Dataset Dataset::Slice(int begin, int end) const {
    if (begin < 0 || end > Size() || begin >= end) {
        throw std::invalid_argument(
            "Dataset::Slice: the range is out of bounds");
    }
    Dataset part;
    part.input = input.segment(begin, end - begin);
    part.output = output.segment(begin, end - begin);
    part.sample_time = sample_time;
    part.input_name = input_name;
    part.output_name = output_name;
    return part;
}

void SplitDataset(const Dataset& source, double training_fraction,
                  Dataset* training, Dataset* validation) {
    if (training == nullptr || validation == nullptr) {
        throw std::invalid_argument(
            "SplitDataset: both outputs must be provided");
    }
    source.Validate("SplitDataset");
    if (!(training_fraction > 0.0 && training_fraction < 1.0)) {
        throw std::invalid_argument(
            "SplitDataset: the training fraction must be strictly between 0 "
            "and 1; "
            "keeping everything for training leaves nothing to validate "
            "against");
    }
    const int split = static_cast<int>(source.Size() * training_fraction);
    if (split < 2 || source.Size() - split < 2) {
        throw std::invalid_argument(
            "SplitDataset: the record is too short to split into two usable "
            "parts");
    }
    // Contiguous, not interleaved: consecutive samples of a dynamic system are
    // correlated, so an interleaved validation set shares most of its
    // information with the training set.
    *training = source.Slice(0, split);
    *validation = source.Slice(split, source.Size());
}

FitMetrics ComputeFit(const Eigen::VectorXd& measured,
                      const Eigen::VectorXd& predicted) {
    if (measured.size() != predicted.size() || measured.size() == 0) {
        throw std::invalid_argument(
            "ComputeFit: the two sequences must have the same non-zero length");
    }
    FitMetrics metrics;
    metrics.samples = static_cast<int>(measured.size());
    const double mean = measured.mean();
    const double error_norm = (measured - predicted).norm();
    const double centred_norm =
        (measured - Eigen::VectorXd::Constant(measured.size(), mean)).norm();
    // The denominator is the deviation from the mean, so an all-zero prediction
    // scores 0 rather than -infinity: "as good as the average" is the baseline
    // a fit percentage should be measured against.
    metrics.fit_percent =
        centred_norm > 0.0 ? 100.0 * (1.0 - error_norm / centred_norm) : 0.0;
    metrics.rmse = std::sqrt((measured - predicted).squaredNorm() /
                             static_cast<double>(measured.size()));
    const double variance =
        (measured - Eigen::VectorXd::Constant(measured.size(), mean))
            .squaredNorm();
    metrics.variance_accounted_for =
        variance > 0.0
            ? 100.0 * (1.0 - (measured - predicted).squaredNorm() / variance)
            : 0.0;
    metrics.worst_absolute_error = (measured - predicted).cwiseAbs().maxCoeff();
    return metrics;
}

}  // namespace ident
}  // namespace control
