// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>
#include <sstream>
#include <stdexcept>

#include "control/ident/ls.h"
#include "control/ident/rls.h"

namespace control {
namespace ident {
namespace {

// The first index at which a full regressor exists: it needs `na` past outputs
// and `delay + nb - 1` past inputs.
int FirstUsableIndex(int denominator_order, int numerator_order, int delay) {
    return std::max(denominator_order, delay + numerator_order - 1);
}

}  // namespace

bool ArxModel::IsValid() const { return DescribeProblem().empty(); }

std::string ArxModel::DescribeProblem() const {
    if (a.size() < 1) {
        return "the denominator is empty: an ARX model needs at least one pole";
    }
    if (b.size() < 1) {
        return "the numerator is empty";
    }
    if (delay < 1) {
        return "the delay must be at least one sample for a discrete model";
    }
    if (!a.allFinite() || !b.allFinite()) {
        return "a coefficient is not finite";
    }
    return std::string();
}

Eigen::VectorXd ArxModel::DenominatorCoefficients() const {
    Eigen::VectorXd coefficients(a.size() + 1);
    coefficients(0) = 1.0;
    coefficients.tail(a.size()) = a;
    return coefficients;
}

Eigen::VectorXd ArxRegressor(const Dataset& data, int index,
                             int denominator_order, int numerator_order,
                             int delay) {
    Eigen::VectorXd regressor(denominator_order + numerator_order);
    for (int i = 0; i < denominator_order; ++i) {
        regressor(i) = -data.output(index - 1 - i);
    }
    for (int j = 0; j < numerator_order; ++j) {
        regressor(denominator_order + j) = data.input(index - delay - j);
    }
    return regressor;
}

ArxFitReport FitArx(const Dataset& training, int denominator_order,
                    int numerator_order, int delay) {
    training.Validate("FitArx");
    if (denominator_order < 1 || numerator_order < 1) {
        throw std::invalid_argument(
            "FitArx: the denominator and numerator orders must both be at "
            "least one");
    }
    if (delay < 1) {
        throw std::invalid_argument(
            "FitArx: the delay must be at least one sample");
    }
    const int first =
        FirstUsableIndex(denominator_order, numerator_order, delay);
    const int rows = training.Size() - first;
    const int parameters = denominator_order + numerator_order;
    if (rows < parameters) {
        throw std::invalid_argument("FitArx: " + std::to_string(rows) +
                                    " usable samples for " +
                                    std::to_string(parameters) +
                                    " parameters; the fit would be "
                                    "underdetermined and would interpolate "
                                    "the noise");
    }

    Eigen::MatrixXd regressors(rows, parameters);
    Eigen::VectorXd targets(rows);
    for (int row = 0; row < rows; ++row) {
        const int index = first + row;
        regressors.row(row) = ArxRegressor(training, index, denominator_order,
                                           numerator_order, delay);
        targets(row) = training.output(index);
    }

    // QR rather than the normal equations: forming Phi'Phi squares the
    // condition number, and the condition number is exactly what this function
    // reports as the diagnosis of a poor experiment.
    const Eigen::VectorXd theta = regressors.householderQr().solve(targets);

    ArxFitReport report;
    report.model.a = theta.head(denominator_order);
    report.model.b = theta.tail(numerator_order);
    report.model.delay = delay;
    report.model.sample_time = training.sample_time;

    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(regressors);
    const Eigen::VectorXd& singular = svd.singularValues();
    report.regressor_condition_number =
        singular.size() > 0 && singular(singular.size() - 1) > 0.0
            ? singular(0) / singular(singular.size() - 1)
            : std::numeric_limits<double>::infinity();
    report.ill_conditioned = report.regressor_condition_number > 1e6;

    const Eigen::VectorXd predicted = regressors * theta;
    report.training_fit = ComputeFit(targets, predicted);

    std::ostringstream notes;
    notes
        << "ARX(" << denominator_order << "," << numerator_order
        << ") with a delay of " << delay
        << " sample(s), fitted by QR least squares on " << rows
        << " samples. The reported fit is the one-step prediction, which is "
           "what "
           "the estimator optimises; the free-run fit on held-out data is the "
           "number that says whether the model is real";
    report.notes = notes.str();
    if (report.ill_conditioned) {
        report.warnings.push_back(
            "the regressor condition number is " +
            std::to_string(report.regressor_condition_number) +
            ": the parameters are not separately identifiable. The fit may "
            "look "
            "excellent while the individual coefficients are arbitrary - "
            "excite the "
            "plant over a wider frequency range, or reduce the order");
    }
    report.ok = true;
    return report;
}

Eigen::VectorXd SimulateArx(const ArxModel& model, const Eigen::VectorXd& input,
                            double sample_time,
                            const Eigen::VectorXd& history) {
    if (!model.IsValid()) {
        throw std::invalid_argument("SimulateArx: " + model.DescribeProblem());
    }
    const int na = model.DenominatorOrder();
    const int nb = model.NumeratorOrder();
    const int n = static_cast<int>(input.size());
    if (n == 0) {
        throw std::invalid_argument("SimulateArx: the input is empty");
    }
    if (history.size() < na) {
        throw std::invalid_argument(
            "SimulateArx: the history must supply at least " +
            std::to_string(na) +
            " past outputs (pass zeros to start from rest)");
    }
    static_cast<void>(sample_time);

    // The free-run simulation: predicted outputs are fed back into the
    // recursion. Using the measured past instead would be a one-step
    // prediction, which is a different and much easier claim.
    Eigen::VectorXd output = Eigen::VectorXd::Zero(n);
    for (int k = 0; k < n; ++k) {
        double value = 0.0;
        for (int i = 0; i < na; ++i) {
            const int past = k - 1 - i;
            const double previous =
                past >= 0 ? output(past)
                          : history(history.size() - 1 + past + 1);
            value -= model.a(i) * previous;
        }
        for (int j = 0; j < nb; ++j) {
            const int index = k - model.delay - j;
            if (index >= 0) {
                value += model.b(j) * input(index);
            }
        }
        output(k) = value;
    }
    return output;
}

Eigen::VectorXd PredictArxOneStep(const ArxModel& model, const Dataset& data) {
    if (!model.IsValid()) {
        throw std::invalid_argument("PredictArxOneStep: " +
                                    model.DescribeProblem());
    }
    const int first = FirstUsableIndex(model.DenominatorOrder(),
                                       model.NumeratorOrder(), model.delay);
    Eigen::VectorXd predicted(data.Size() - first);
    for (int row = 0; row < static_cast<int>(predicted.size()); ++row) {
        const int index = first + row;
        predicted(row) = ArxRegressor(data, index, model.DenominatorOrder(),
                                      model.NumeratorOrder(), model.delay)
                             .dot((Eigen::VectorXd(model.DenominatorOrder() +
                                                   model.NumeratorOrder())
                                       << model.a,
                                   model.b)
                                      .finished());
    }
    return predicted;
}

StateSpaceModel ToStateSpace(const ArxModel& model) {
    if (!model.IsValid()) {
        throw std::invalid_argument("ToStateSpace: " + model.DescribeProblem());
    }
    const int na = model.DenominatorOrder();
    const int nb = model.NumeratorOrder();
    // The numerator in positive powers of z occupies z^(na-delay) down to
    // z^(na-delay-nb+1). If it reaches z^(na) or above, the transfer function
    // is improper and no proper state-space model represents it.
    if (model.delay + nb - 1 > na) {
        throw std::invalid_argument(
            "ToStateSpace: the numerator has a higher degree than the "
            "denominator "
            "(delay " +
            std::to_string(model.delay) + " with " + std::to_string(nb) +
            " numerator coefficients against " + std::to_string(na) +
            " poles), so the model is not proper and has no state-space form. "
            "Reduce "
            "the numerator order or the delay");
    }

    // Numerator coefficients in descending powers of z, index i holding the
    // coefficient of z^(na-i). The difference equation has b_j multiplying
    // u[k-delay-j], which is the z^(-delay-j) coefficient, and z^(na-delay-j)
    // after multiplying through by z^na - so it lands at index delay + j.
    Eigen::VectorXd numerator = Eigen::VectorXd::Zero(na + 1);
    for (int j = 0; j < nb; ++j) {
        numerator(model.delay + j) = model.b(j);
    }

    // Controllable canonical form, which is the one whose transfer function can
    // be written down and checked in a line:
    //
    //   A = [[-a1, -a2, ..., -an], [1, 0, ...], [0, 1, ...], ...],  B = e1,
    //   C = [n1, n2, ..., nn],  D = n0,
    //
    // and for a companion matrix (zI - A)^-1 e1 = [z^(n-1), z^(n-2), ..., 1]' /
    // det, so C (zI-A)^-1 B = (n1 z^(n-1) + ... + nn) / det exactly.
    //
    // An earlier version used this A (first row -a, subdiagonal ones) together
    // with the B and C of the *observable* canonical form, which is a different
    // companion convention. The result had the right poles and the wrong
    // numerator - the kind of error a pole check passes and an impulse-response
    // check fails.
    StateSpaceModel state_space;
    state_space.a = Eigen::MatrixXd::Zero(na, na);
    // The denominator is written 1 + a0 z^-1 + ... , which in positive powers
    // of z is z^n + a0 z^(n-1) + ... , and the companion matrix's first row is
    // the *negated* coefficient vector: A(0, i) = -a_i. Using +a_i instead
    // gives a realisation with the poles reflected - z^2 - 1.5 z + 0.7 becomes
    // z^2 + 1.5 z + 0.7 - so the impulse response diverges from the difference
    // equation it is supposed to represent while the shape still looks
    // plausible.
    for (int column = 0; column < na; ++column) {
        state_space.a(0, column) = -model.a(column);
    }
    for (int row = 1; row < na; ++row) {
        state_space.a(row, row - 1) = 1.0;
    }
    state_space.b = Eigen::VectorXd::Zero(na);
    state_space.b(0) = 1.0;
    state_space.c = Eigen::MatrixXd::Zero(1, na);
    for (int i = 0; i < na; ++i) {
        state_space.c(0, i) = numerator(i + 1);
    }
    state_space.d = Eigen::MatrixXd::Zero(1, 1);
    state_space.d(0, 0) = numerator(0);
    state_space.sample_time = model.sample_time;
    return state_space;
}

Eigen::VectorXcd ArxPoles(const ArxModel& model) {
    const StateSpaceModel state_space = ToStateSpace(model);
    Eigen::EigenSolver<Eigen::MatrixXd> solver(state_space.a, false);
    Eigen::VectorXcd poles = solver.eigenvalues();
    std::sort(poles.data(), poles.data() + poles.size(),
              [](const std::complex<double>& left,
                 const std::complex<double>& right) {
                  return std::abs(left) > std::abs(right);
              });
    return poles;
}

std::complex<double> ArxFrequencyResponse(const ArxModel& model, double omega) {
    const std::complex<double> z =
        std::exp(std::complex<double>(0.0, omega * model.sample_time));
    std::complex<double> numerator(0.0, 0.0);
    for (int j = 0; j < model.NumeratorOrder(); ++j) {
        numerator += model.b(j) / std::pow(z, model.delay + j);
    }
    std::complex<double> denominator(1.0, 0.0);
    for (int i = 0; i < model.DenominatorOrder(); ++i) {
        denominator += model.a(i) / std::pow(z, i + 1);
    }
    return numerator / denominator;
}

Eigen::VectorXd ArxImpulseResponse(const ArxModel& model, int samples) {
    if (samples < 1) {
        throw std::invalid_argument(
            "ArxImpulseResponse: samples must be positive");
    }
    Eigen::VectorXd input = Eigen::VectorXd::Zero(samples);
    input(0) = 1.0;
    Eigen::VectorXd history = Eigen::VectorXd::Zero(model.DenominatorOrder());
    return SimulateArx(model, input, model.sample_time, history);
}

}  // namespace ident
}  // namespace control
