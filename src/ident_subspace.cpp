// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>
#include <sstream>
#include <stdexcept>

#include "control/ident/subspace.h"

namespace control {
namespace ident {

EraResult RealizeFromMarkovParameters(const Eigen::VectorXd& markov, int inputs,
                                      int outputs, double sample_time,
                                      const EraOptions& options) {
    if (inputs != 1 || outputs != 1) {
        throw std::invalid_argument(
            "RealizeFromMarkovParameters: only SISO is supported; a "
            "multi-variable "
            "realization needs the block Hankel built per channel and is not "
            "implemented here");
    }
    if (!(sample_time > 0.0)) {
        throw std::invalid_argument(
            "RealizeFromMarkovParameters: the sample time must be positive");
    }
    if (options.block_rows < 1) {
        throw std::invalid_argument(
            "RealizeFromMarkovParameters: block_rows must be positive");
    }
    // The shifted Hankel needs one more Markov parameter than the unshifted
    // one.
    const int needed = 2 * options.block_rows + 1;
    if (markov.size() < needed) {
        throw std::invalid_argument(
            "RealizeFromMarkovParameters: " + std::to_string(markov.size()) +
            " Markov parameters for " + std::to_string(options.block_rows) +
            " block rows; at least " + std::to_string(needed) +
            " are needed (the shifted Hankel consumes one more)");
    }

    const int rows = options.block_rows;
    const int columns = static_cast<int>(markov.size()) - rows - 1;
    // H(r, c) = H_{1 + r + c}, and the shifted one starts at H_2.
    Eigen::MatrixXd hankel(rows, columns);
    Eigen::MatrixXd shifted(rows, columns);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < columns; ++c) {
            hankel(r, c) = markov(1 + r + c);
            shifted(r, c) = markov(2 + r + c);
        }
    }

    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        hankel, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& singular = svd.singularValues();
    EraResult result;
    result.hankel_singular_values = singular;
    result.largest_singular_value = singular.size() > 0 ? singular(0) : 0.0;

    // Order from the singular values, and separately from the largest ratio
    // between consecutive ones. The two can disagree, and the disagreement is
    // worth reporting: a hand-picked threshold has no units, while a gap is a
    // statement about the data.
    int order = 0;
    if (options.order > 0) {
        order = options.order;
        if (order > singular.size()) {
            throw std::invalid_argument(
                "RealizeFromMarkovParameters: the requested order exceeds the "
                "rank of "
                "the Hankel matrix; supply more Markov parameters or a lower "
                "order");
        }
    } else {
        const double threshold =
            options.singular_value_threshold * result.largest_singular_value;
        for (Eigen::Index i = 0; i < singular.size(); ++i) {
            if (singular(i) > threshold) {
                ++order;
            }
        }
        order = std::max(1, order);
    }
    result.order = order;
    result.singular_value_ratios = Eigen::VectorXd::Zero(
        std::max(0, static_cast<int>(singular.size()) - 1));
    int gap_order = order;
    double largest_ratio = 0.0;
    for (int i = 0; i + 1 < singular.size(); ++i) {
        const double ratio =
            singular(i + 1) > 0.0 ? singular(i) / singular(i + 1) : 1e300;
        result.singular_value_ratios(i) = ratio;
        if (ratio > largest_ratio) {
            largest_ratio = ratio;
            gap_order = i + 1;
        }
    }
    result.order_from_largest_gap = gap_order;

    // Ho-Kalman: H = Gamma Delta, so Gamma = U Sigma^(1/2) and Delta =
    // Sigma^(1/2) V', and A follows from the shifted Hankel.
    const Eigen::MatrixXd u = svd.matrixU().leftCols(order);
    const Eigen::MatrixXd v = svd.matrixV().leftCols(order);
    Eigen::VectorXd sigma_root(order);
    Eigen::VectorXd sigma_inverse_root(order);
    for (int i = 0; i < order; ++i) {
        if (!(singular(i) > 0.0)) {
            throw std::invalid_argument(
                "RealizeFromMarkovParameters: a singular value used by the "
                "requested "
                "order is zero, so the realization is not defined; lower the "
                "order");
        }
        sigma_root(i) = std::sqrt(singular(i));
        sigma_inverse_root(i) = 1.0 / sigma_root(i);
    }
    const Eigen::MatrixXd observability = u * sigma_root.asDiagonal();
    const Eigen::MatrixXd controllability =
        sigma_root.asDiagonal() * v.transpose();

    const Eigen::MatrixXd a = sigma_inverse_root.asDiagonal() * u.transpose() *
                              shifted * v * sigma_inverse_root.asDiagonal();
    Eigen::MatrixXd c(1, order);
    c = observability.row(0);
    Eigen::MatrixXd b(order, 1);
    b = controllability.col(0);
    Eigen::MatrixXd d(1, 1);
    d(0, 0) = markov(0);

    result.model.a = a;
    result.model.b = b;
    result.model.c = c;
    result.model.d = d;
    result.model.sample_time = sample_time;

    std::ostringstream notes;
    notes << "ERA from " << markov.size() << " Markov parameters with "
          << options.block_rows << " block rows: order " << order;
    if (!options.order && gap_order != order) {
        notes << " chosen by threshold; the largest gap between consecutive "
                 "singular "
                 "values suggests "
              << gap_order << " instead (ratio " << largest_ratio
              << "). Look at the singular values rather than trusting either "
                 "number";
        result.warnings.push_back(
            "the threshold and the largest gap disagree about the order (" +
            std::to_string(order) + " against " + std::to_string(gap_order) +
            ")");
    }
    if (order > 0 && singular.size() > order &&
        singular(order - 1) < 1e-10 * result.largest_singular_value) {
        result.warnings.push_back(
            "the smallest singular value kept is below 1e-10 of the largest: "
            "the "
            "realization is being built on numerical noise and the A matrix "
            "will not "
            "reproduce the data");
    }
    result.notes = notes.str();
    result.ok = true;
    return result;
}

EraResult RealizeFromImpulseResponse(const Eigen::VectorXd& impulse_response,
                                     double sample_time,
                                     const EraOptions& options) {
    // The impulse response *is* the Markov sequence: h[0] = D, h[1] = C B,
    // h[2] = C A B, and so on. It must be passed through unchanged.
    //
    // An earlier version prepended a zero "for a strictly proper model". That
    // shifts the Hankel so its first entry is D instead of C B, and since the
    // Ho-Kalman factorization H = Gamma Delta needs the entries to be C A^(r+c)
    // B, the result is the correct Hankel plus D in a single entry - a rank-one
    // perturbation. The symptom is a Hankel that keeps one singular value more
    // than the true order, which is exactly what a threshold-based order count
    // then reports: the estimated order was always one too high.
    return RealizeFromMarkovParameters(impulse_response, 1, 1, sample_time,
                                       options);
}

Eigen::VectorXcd StateSpacePoles(const StateSpaceModel& model) {
    Eigen::EigenSolver<Eigen::MatrixXd> solver(model.a, false);
    Eigen::VectorXcd poles = solver.eigenvalues();
    std::sort(poles.data(), poles.data() + poles.size(),
              [](const std::complex<double>& left,
                 const std::complex<double>& right) {
                  return std::abs(left) > std::abs(right);
              });
    return poles;
}

Eigen::VectorXd StateSpaceImpulseResponse(const StateSpaceModel& model,
                                          int samples) {
    if (samples < 1) {
        throw std::invalid_argument(
            "StateSpaceImpulseResponse: samples must be positive");
    }
    Eigen::VectorXd response(samples);
    Eigen::VectorXd state = Eigen::VectorXd::Zero(model.a.rows());
    for (int k = 0; k < samples; ++k) {
        const double command = k == 0 ? 1.0 : 0.0;
        response(k) = (model.c * state)(0, 0) + model.d(0, 0) * command;
        state = model.a * state + model.b * command;
    }
    return response;
}

}  // namespace ident
}  // namespace control
