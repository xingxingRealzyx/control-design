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

// Constructors, validation, and the small text helpers shared by all modules.

#include "control/types.h"

#include <Eigen/SVD>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace control {
namespace {

void AppendIfNotEmpty(std::ostringstream* stream, const std::string& text) {
    if (!text.empty()) {
        *stream << text << "; ";
    }
}

}  // namespace

LtiSystem::LtiSystem(Matrix a, Matrix b, Matrix c, Matrix d, double sample_time)
    : a_(std::move(a)),
      b_(std::move(b)),
      c_(std::move(c)),
      d_(std::move(d)),
      sample_time_(sample_time) {}

LtiSystem LtiSystem::Continuous(Matrix a, Matrix b, Matrix c) {
    const int outputs = static_cast<int>(c.rows());
    const int inputs = static_cast<int>(b.cols());
    return LtiSystem(std::move(a), std::move(b), std::move(c),
                     Matrix::Zero(outputs, inputs), 0.0);
}

LtiSystem LtiSystem::Discrete(Matrix a, Matrix b, Matrix c,
                              double sample_time) {
    const int outputs = static_cast<int>(c.rows());
    const int inputs = static_cast<int>(b.cols());
    return LtiSystem(std::move(a), std::move(b), std::move(c),
                     Matrix::Zero(outputs, inputs), sample_time);
}

std::string LtiSystem::DescribeProblem() const {
    std::ostringstream problems;
    const int n = static_cast<int>(a_.rows());

    if (a_.size() == 0) {
        problems << "A is empty";
    } else {
        if (a_.rows() != a_.cols()) {
            problems << "A is not square (" << a_.rows() << "x" << a_.cols()
                     << ")";
        }
        if (b_.rows() != n) {
            AppendIfNotEmpty(&problems, "");
            problems << "B has " << b_.rows() << " rows but A has " << n;
        }
        if (c_.cols() != n) {
            AppendIfNotEmpty(&problems, "");
            problems << "C has " << c_.cols() << " columns but A has " << n;
        }
        if (d_.rows() != c_.rows() || d_.cols() != b_.cols()) {
            AppendIfNotEmpty(&problems, "");
            problems << "D should be " << c_.rows() << "x" << b_.cols()
                     << " but is " << d_.rows() << "x" << d_.cols();
        }
    }

    if (!a_.allFinite() || !b_.allFinite() || !c_.allFinite() ||
        !d_.allFinite()) {
        AppendIfNotEmpty(&problems, "");
        problems << "the model contains NaN or infinity";
    }
    if (sample_time_ < 0.0 || !std::isfinite(sample_time_)) {
        AppendIfNotEmpty(&problems, "");
        problems << "sample time is negative or not finite (" << sample_time_
                 << ")";
    }

    return problems.str();
}

bool LtiSystem::IsValid() const { return DescribeProblem().empty(); }

void LtiSystem::Validate(const char* context) const {
    const std::string problem = DescribeProblem();
    if (!problem.empty()) {
        throw std::invalid_argument(std::string(context) + ": " + problem);
    }
}

double RankResult::condition_number() const {
    if (smallest_singular_value <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return largest_singular_value / smallest_singular_value;
}

const char* ToString(DiscretizationMethod method) {
    switch (method) {
        case DiscretizationMethod::kZeroOrderHold:
            return "zero-order hold";
        case DiscretizationMethod::kTustin:
            return "Tustin (bilinear)";
        case DiscretizationMethod::kForwardEuler:
            return "forward Euler";
        case DiscretizationMethod::kBackwardEuler:
            return "backward Euler";
        case DiscretizationMethod::kMatchedPoleZero:
            return "matched pole-zero";
    }
    return "unknown";
}

const char* ToString(RiccatiMethod method) {
    switch (method) {
        case RiccatiMethod::kAuto:
            return "auto";
        case RiccatiMethod::kSchur:
            return "Schur";
        case RiccatiMethod::kKleinman:
            return "Kleinman";
        case RiccatiMethod::kIteration:
            return "fixed-point iteration";
    }
    return "unknown";
}

std::ostream& operator<<(std::ostream& out, const LtiSystem& system) {
    out << (system.is_continuous() ? "continuous" : "discrete")
        << " state-space: " << system.state_dimension() << " states, "
        << system.input_dimension() << " inputs, " << system.output_dimension()
        << " outputs";
    if (system.is_discrete()) {
        out << ", Ts = " << system.sample_time() << " s";
    }
    out << "\nA =\n"
        << system.a() << "\nB =\n"
        << system.b() << "\nC =\n"
        << system.c();
    if (system.d().size() > 0 && system.d().cwiseAbs().maxCoeff() > 0.0) {
        out << "\nD =\n" << system.d();
    }
    return out;
}

}  // namespace control
