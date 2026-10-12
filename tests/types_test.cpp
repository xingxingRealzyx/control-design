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

#include "control/types.h"

#include <limits>
#include <sstream>
#include <stdexcept>

#include "tests/testing.h"

namespace {

control::LtiSystem SecondOrder() {
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -2.0, -3.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    return control::LtiSystem::Continuous(a, b, c);
}

}  // namespace

TEST(LtiSystem, ConstructorsSetTheShapesAndTheKind) {
    const control::LtiSystem continuous = SecondOrder();
    CHECK_EQ(continuous.state_dimension(), 2);
    CHECK_EQ(continuous.input_dimension(), 1);
    CHECK_EQ(continuous.output_dimension(), 1);
    CHECK_TRUE(continuous.is_continuous());
    CHECK_FALSE(continuous.is_discrete());
    // The convenience constructor must produce a D of the right shape, not an
    // empty matrix: a 1x0 D used to make validation fail on a valid model.
    CHECK_EQ(continuous.d().rows(), 1);
    CHECK_EQ(continuous.d().cols(), 1);
    CHECK_NEAR(continuous.d()(0, 0), 0.0, 0.0);
    CHECK_TRUE(continuous.IsValid());
    CHECK_TRUE(continuous.DescribeProblem().empty());

    const control::LtiSystem discrete = control::LtiSystem::Discrete(
        continuous.a(), continuous.b(), continuous.c(), 0.01);
    CHECK_TRUE(discrete.is_discrete());
    CHECK_NEAR(discrete.sample_time(), 0.01, 0.0);
    CHECK_TRUE(discrete.IsValid());
}

TEST(LtiSystem, ValidationNamesTheProblem) {
    const control::LtiSystem good = SecondOrder();

    // A is not square.
    control::Matrix bad_a = control::Matrix::Zero(2, 3);
    control::LtiSystem not_square(bad_a, good.b(), good.c(),
                                  control::Matrix::Zero(1, 1), 0.0);
    CHECK_FALSE(not_square.IsValid());
    CHECK_TRUE(not_square.DescribeProblem().find("not square") !=
               std::string::npos);
    CHECK_THROWS(not_square.Validate("test"));

    // B has the wrong number of rows.
    control::Matrix bad_b = control::Matrix::Zero(3, 1);
    control::LtiSystem wrong_b(good.a(), bad_b, good.c(),
                               control::Matrix::Zero(1, 1), 0.0);
    CHECK_FALSE(wrong_b.IsValid());

    // A non-finite entry is reported as such, wherever it is.
    control::Matrix nan_a = good.a();
    nan_a(1, 1) = std::numeric_limits<double>::quiet_NaN();
    control::LtiSystem not_finite(nan_a, good.b(), good.c(),
                                  control::Matrix::Zero(1, 1), 0.0);
    CHECK_FALSE(not_finite.IsValid());
    CHECK_TRUE(not_finite.DescribeProblem().find("NaN") != std::string::npos);

    // A negative sample time is a mistake, not a continuous model.
    control::LtiSystem negative_ts(good.a(), good.b(), good.c(),
                                   control::Matrix::Zero(1, 1), -1.0);
    CHECK_FALSE(negative_ts.IsValid());
}

TEST(LtiSystem, RankResultReportsMarginalCasesNotJustABoolean) {
    control::RankResult full;
    full.rank = 3;
    full.dimension = 3;
    full.smallest_singular_value = 1.0;
    full.largest_singular_value = 10.0;
    CHECK_TRUE(full.IsFullRank());
    CHECK_NEAR(full.condition_number(), 10.0, 0.0);

    // "Controllable, but only just" is the case a design tool has to surface.
    control::RankResult marginal;
    marginal.rank = 3;
    marginal.dimension = 3;
    marginal.smallest_singular_value = 1e-12;
    marginal.largest_singular_value = 1.0;
    CHECK_TRUE(marginal.IsFullRank());
    CHECK_TRUE(marginal.condition_number() > 1e11);

    control::RankResult rank_deficient;
    rank_deficient.rank = 2;
    rank_deficient.dimension = 3;
    rank_deficient.smallest_singular_value = 0.0;
    rank_deficient.largest_singular_value = 1.0;
    CHECK_FALSE(rank_deficient.IsFullRank());
    CHECK_TRUE(std::isinf(rank_deficient.condition_number()));
}

TEST(LtiSystem, StreamOutputNamesTheModel) {
    std::ostringstream text;
    text << SecondOrder();
    const std::string rendered = text.str();
    CHECK_TRUE(rendered.find("continuous") != std::string::npos);
    CHECK_TRUE(rendered.find("2 states") != std::string::npos);
}

TEST(Types, MethodNamesAreStable) {
    // The enumerations are part of the report text a user reads, so their names
    // are pinned: a future refactor should not silently change the output.
    CHECK_TRUE(std::string(control::ToString(
                   control::DiscretizationMethod::kZeroOrderHold)) ==
               "zero-order hold");
    CHECK_TRUE(std::string(
                   control::ToString(control::DiscretizationMethod::kTustin)) ==
               "Tustin (bilinear)");
    CHECK_TRUE(std::string(control::ToString(control::RiccatiMethod::kAuto)) ==
               "auto");
}
