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

#include "control/codegen.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "control/design.h"
#include "control/discretize.h"
#include "control/lti.h"
#include "tests/testing.h"

namespace {

control::LtiSystem DiscretePlant() {
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -2.0, -3.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    return control::Discretize(control::LtiSystem::Continuous(a, b, c), 0.05);
}

control::DesignResult Feedback(const control::LtiSystem& discrete) {
    control::Matrix q(2, 2);
    q << 10.0, 0.0, 0.0, 1.0;
    control::Matrix r(1, 1);
    r << 1.0;
    return control::DiscreteLinearQuadraticRegulator(discrete, q, r);
}

control::CodegenOptions Named(const std::string& name) {
    control::CodegenOptions options;
    options.controller_name = name;
    options.namespace_name = "test_ns";
    options.state_units = {"m", "m/s"};
    options.input_unit = "N";
    options.output_unit = "m";
    return options;
}

bool Contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

// Extracts the first numeric literal of a `static constexpr Real kK[...]`
// block, so the emitted text can be parsed back and compared with the design.
double FirstGainLiteral(const std::string& header) {
    const std::size_t start = header.find("kK[kInputs][kStates] = {");
    if (start == std::string::npos) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // The emitted form is static_cast<Real>(<17-digit literal>), so the number
    // starts after the innermost opening parenthesis.
    const std::size_t brace = header.find('{', header.find('{', start) + 1);
    const std::size_t open = header.find('(', brace);
    const std::size_t end = header.find_first_of(",}", open + 1);
    return std::strtod(header.substr(open + 1, end - open - 1).c_str(),
                       nullptr);
}

}  // namespace

TEST(Codegen, EmitsASelfContainedHeaderWithTheDesignInsideIt) {
    const control::LtiSystem discrete = DiscretePlant();
    const control::DesignResult design = Feedback(discrete);
    const control::CodegenResult generated =
        control::GenerateController(discrete, design, Named("TestController"));

    const std::string& header = generated.header;
    // Include guard, namespace and class name all come from the options.
    CHECK_TRUE(Contains(header, "#ifndef test_ns_TestController_H_"));
    CHECK_TRUE(Contains(header, "#define test_ns_TestController_H_"));
    CHECK_TRUE(Contains(header, "#endif  // test_ns_TestController_H_"));
    CHECK_TRUE(Contains(header, "namespace test_ns {"));
    CHECK_TRUE(Contains(header, "class TestController {"));
    CHECK_TRUE(Contains(header, "using Real = double;"));
    // Dimensions, sample time and the gains.
    CHECK_TRUE(Contains(header, "static constexpr std::size_t kStates = 2;"));
    CHECK_TRUE(Contains(header, "static constexpr std::size_t kInputs = 1;"));
    CHECK_TRUE(
        Contains(header, "static constexpr Real kK[kInputs][kStates] = {"));
    // The units the generator was told about reach the comments, because the
    // generator cannot invent them and a bare `measurement[0]` is not a spec.
    CHECK_TRUE(
        Contains(header, "[m, m/s]"));  // every state unit, not just the first
    CHECK_TRUE(Contains(header, "[N]"));
    CHECK_EQ(generated.suggested_filename, std::string("TestController.h"));
    CHECK_FALSE(generated.has_observer);
    CHECK_FALSE(generated.has_integral_action);
    // The usage example names the right measurement width for the chosen mode.
    CHECK_TRUE(Contains(generated.usage_example, "measurement[2]"));
}

TEST(Codegen, TargetConstraintsAreRespected) {
    // The emitted code runs on a target with no exceptions, no heap and no
    // RTTI, so none of those may appear. This is a text check, but it is the
    // only automated guard against someone adding a std::vector to the
    // template.
    const control::LtiSystem discrete = DiscretePlant();
    const control::CodegenResult generated = control::GenerateController(
        discrete, Feedback(discrete), Named("TestController"));
    const std::string& header = generated.header;
    CHECK_FALSE(Contains(header, "throw"));
    CHECK_FALSE(Contains(header, "new "));
    CHECK_FALSE(Contains(header, "malloc"));
    CHECK_FALSE(Contains(header, "std::vector"));
    CHECK_FALSE(Contains(header, "dynamic_cast"));
    CHECK_FALSE(Contains(header, "typeid"));
    // std::array is the one container, and it is fixed size.
    CHECK_TRUE(Contains(header, "#include <array>"));
    CHECK_TRUE(Contains(header, "std::array<Real, kStates>"));
    // Every gain carries its provenance.
    CHECK_TRUE(Contains(header, "// State feedback gain K, from"));
}

TEST(Codegen, GainLiteralsRoundTripExactly) {
    // A gain written with fewer digits is a different controller. The literal
    // in the header must parse back to the same double the design produced.
    const control::LtiSystem discrete = DiscretePlant();
    const control::DesignResult design = Feedback(discrete);
    const control::CodegenResult generated =
        control::GenerateController(discrete, design, Named("TestController"));

    const double parsed = FirstGainLiteral(generated.header);
    CHECK_TRUE(std::isfinite(parsed));
    CHECK_FALSE(std::isnan(parsed));
    // A design whose first gain is an awkward decimal: it has to survive.
    CHECK_NEAR(parsed, design.gain(0, 0), 0.0);

    // And a gain chosen to be exactly representable but long in decimal.
    control::DesignResult awkward = design;
    awkward.gain(0, 0) = 0.1 + 0.2;  // 0.30000000000000004
    const control::CodegenResult again =
        control::GenerateController(discrete, awkward, Named("Awkward"));
    CHECK_NEAR(FirstGainLiteral(again.header), 0.1 + 0.2, 0.0);
}

TEST(Codegen, EmittedLiteralsAreCastSoAFloatTargetCompiles) {
    // `float k[]{0.30000000000000004}` is ill-formed: the value is not exactly
    // representable, so a braced initializer narrows and the build fails. The
    // emitted literals are therefore wrapped in static_cast<Real>(...), which
    // makes the conversion explicit. A float build is the normal case on an
    // MCU, so this is not a corner.
    const control::LtiSystem discrete = DiscretePlant();
    control::DesignResult design = Feedback(discrete);
    design.gain(0, 0) = 0.1 + 0.2;  // exactly the value that fails to narrow
    control::CodegenOptions float_options = Named("FloatController");
    float_options.real_type = "float";
    const control::CodegenResult generated =
        control::GenerateController(discrete, design, float_options);
    CHECK_TRUE(Contains(generated.header, "using Real = float;"));
    CHECK_TRUE(
        Contains(generated.header, "static_cast<Real>(0.30000000000000004)"));
    CHECK_FALSE(Contains(generated.header, "= {0.30000000000000004"));
    // The sample time is emitted at full precision too, not rounded to 0.05.
    CHECK_TRUE(Contains(generated.header, "static_cast<Real>(0.05"));
}

TEST(Codegen, NonIntegralVariantDoesNotEmitAnUnusedLocal) {
    // The emitted file has to be warning-free under -Werror on the target, so
    // an unused tracking error must not be emitted when there is no integrator.
    const control::LtiSystem discrete = DiscretePlant();
    const control::CodegenResult plain = control::GenerateController(
        discrete, Feedback(discrete), Named("Plain"));
    CHECK_FALSE(Contains(plain.header, "tracking_error"));
    CHECK_TRUE(Contains(plain.header, "static_cast<void>(setpoint)"));

    control::Matrix q(2, 2);
    q << 10.0, 0.0, 0.0, 1.0;
    control::Matrix qi(1, 1);
    qi << 5.0;
    control::Matrix r(1, 1);
    r << 1.0;
    const control::AugmentedDesign augmented =
        control::AddIntegralActionLqr(discrete, 0, q, qi, r);
    const control::CodegenResult tracking =
        control::GenerateControllerWithIntegral(discrete, augmented,
                                                Named("Tracking"));
    CHECK_TRUE(Contains(tracking.header, "tracking_error"));
    CHECK_FALSE(Contains(tracking.header, "static_cast<void>(setpoint)"));
}

TEST(Codegen, SaturationIsEmittedOnlyWhenAskedFor) {
    const control::LtiSystem discrete = DiscretePlant();
    const control::DesignResult design = Feedback(discrete);

    control::CodegenOptions plain = Named("Plain");
    CHECK_FALSE(plain.saturate);
    const control::CodegenResult without =
        control::GenerateController(discrete, design, plain);
    CHECK_FALSE(Contains(without.header, "Limit(command[i]"));

    control::CodegenOptions limited = Named("Limited");
    limited.saturate = true;
    limited.input_min = -2.5;
    limited.input_max = 2.5;
    const control::CodegenResult with =
        control::GenerateController(discrete, design, limited);
    CHECK_TRUE(Contains(with.header, "Limit(command[i]"));
    CHECK_TRUE(Contains(with.header, "-2.5"));
    CHECK_TRUE(Contains(with.header, "2.5"));
    // The limit is reported at run time so a caller can tell that the loop was
    // commanding more than the actuator could deliver.
    CHECK_TRUE(Contains(with.header, "bool saturated() const"));
}

TEST(Codegen, ObserverVariantTakesMeasurementsInsteadOfState) {
    const control::LtiSystem discrete = DiscretePlant();
    const control::DesignResult feedback = Feedback(discrete);
    const control::DesignResult observer =
        control::PlaceObserverPoles(discrete, {{-8.0, 0.0}, {-10.0, 0.0}});

    control::CodegenOptions options = Named("Observed");
    const control::CodegenResult generated =
        control::GenerateController(discrete, feedback, options, &observer);
    CHECK_TRUE(generated.has_observer);
    CHECK_TRUE(Contains(generated.header,
                        "static constexpr Real kL[kStates][kOutputs]"));
    // The observer prediction uses the plant matrices, so they must be present.
    CHECK_TRUE(Contains(generated.header, "kAd[i][j] * x_hat_[j]"));
    CHECK_TRUE(Contains(generated.header, "kBd[i][j] * u_prev_[j]"));
    CHECK_TRUE(Contains(generated.header, "kCd[i][j] * predicted[j]"));
    // The measurement is now an output, so the usage example says so.
    CHECK_TRUE(Contains(generated.usage_example, "measurement[1]"));
    CHECK_TRUE(
        Contains(generated.header, "measured outputs, kOutputs entries"));
}

TEST(Codegen, IntegralVariantEmitsAnIntegratorThatTheDesignAssumed) {
    // The augmented gain has n+1 columns; emitting it as plain state feedback
    // would silently drop the integral gain, which is why the plain entry point
    // refuses a gain of that shape.
    const control::LtiSystem discrete = DiscretePlant();
    control::Matrix q(2, 2);
    q << 10.0, 0.0, 0.0, 1.0;
    control::Matrix qi(1, 1);
    qi << 5.0;
    control::Matrix r(1, 1);
    r << 1.0;
    const control::AugmentedDesign augmented =
        control::AddIntegralActionLqr(discrete, 0, q, qi, r);

    CHECK_THROWS(control::GenerateController(discrete, augmented.design,
                                             Named("Wrong")));

    control::CodegenResult generated = control::GenerateControllerWithIntegral(
        discrete, augmented, Named("Tracking"));
    CHECK_TRUE(generated.has_integral_action);
    CHECK_TRUE(
        Contains(generated.header, "static constexpr Real kKi[kInputs]"));
    // The integrator update uses the sample time, which is the same convention
    // the augmented design used.
    CHECK_TRUE(Contains(generated.header,
                        "integral_[0] += kSampleTime * tracking_error;"));
    // Anti-windup: the integrator is held while the command is clipped.
    CHECK_TRUE(Contains(generated.header, "winding_up"));
    // Both gain blocks are present and the state block has n columns, not n+1.
    CHECK_TRUE(Contains(generated.header, "kK[i][j] * x_hat_[j]"));
    CHECK_TRUE(Contains(generated.header, "kKi[i] * integral_[0]"));
    CHECK_EQ(generated.state_feedback_gain.cols(), 2);
    CHECK_EQ(generated.integral_gain.rows(), 1);
    CHECK_NEAR(generated.integral_gain(0, 0), augmented.design.gain(0, 2), 0.0);
}

TEST(Codegen, RefusesDesignsItCannotEmitFaithfully) {
    const control::LtiSystem discrete = DiscretePlant();
    const control::DesignResult design = Feedback(discrete);

    // A continuous model: the emitted observer would use the wrong matrices.
    control::Matrix a(2, 2);
    a << 0.0, 1.0, -2.0, -3.0;
    control::Matrix b(2, 1);
    b << 0.0, 1.0;
    control::Matrix c(1, 2);
    c << 1.0, 0.0;
    const control::LtiSystem continuous =
        control::LtiSystem::Continuous(a, b, c);
    CHECK_THROWS(
        control::GenerateController(continuous, design, Named("Continuous")));

    // A gain of the wrong shape.
    control::DesignResult wrong = design;
    wrong.gain = control::Matrix::Zero(1, 3);
    CHECK_THROWS(
        control::GenerateController(discrete, wrong, Named("WrongShape")));

    // A non-finite gain: there is nothing to write.
    control::DesignResult not_finite = design;
    not_finite.gain(0, 0) = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(
        control::GenerateController(discrete, not_finite, Named("Nan")));

    // An observer gain of the wrong shape.
    control::DesignResult bad_observer = design;
    bad_observer.gain = control::Matrix::Zero(2, 2);
    CHECK_THROWS(control::GenerateController(
        discrete, design, Named("BadObserver"), &bad_observer));
}

TEST(Codegen, DesignSummaryCarriesTheNumbersAndTheWarnings) {
    const control::LtiSystem discrete = DiscretePlant();
    const control::DesignResult design = Feedback(discrete);
    const std::string summary =
        control::FormatDesignSummary(design, "test design");
    CHECK_TRUE(Contains(summary, "test design"));
    CHECK_TRUE(Contains(summary, "gain:"));
    CHECK_TRUE(Contains(summary, "closed-loop poles:"));
    CHECK_TRUE(Contains(summary, "damping:"));
    CHECK_TRUE(Contains(summary, "largest gain:"));
    CHECK_TRUE(Contains(summary, design.notes));
}
