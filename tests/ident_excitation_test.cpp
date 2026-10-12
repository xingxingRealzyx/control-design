// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#include <Eigen/Core>
#include <cmath>
#include <set>
#include <stdexcept>

#include "control/ident/excitation.h"
#include "tests/testing.h"

TEST(Excitation, PrbsIsBinaryAndHasTheMaximalPeriod) {
    control::ident::PrbsOptions options;
    options.order = 5;  // period 31
    options.samples_per_bit = 1;
    const Eigen::VectorXd signal = control::ident::GeneratePrbs(options);
    CHECK_EQ(signal.size(), 31);
    // Binary: exactly two distinct levels, symmetric about zero.
    std::set<double> levels;
    for (Eigen::Index i = 0; i < signal.size(); ++i) {
        levels.insert(signal(i));
    }
    CHECK_EQ(levels.size(), 2u);
    // A maximal-length sequence has one more of one level than the other; which
    // one depends on the tap convention, so the sum is asserted to be a single
    // level either way rather than assumed to be negative.
    CHECK_NEAR(std::abs(signal.sum()), 1.0, 1e-12);
    // A maximal-length sequence has no repeated window of `order` bits, which
    // is what gives it a flat spectrum.
    std::set<std::string> windows;
    for (int start = 0; start < 31; ++start) {
        std::string window;
        for (int k = 0; k < 5; ++k) {
            window += signal((start + k) % 31) > 0.0 ? '1' : '0';
        }
        windows.insert(window);
    }
    CHECK_EQ(windows.size(), 31u);
}

TEST(Excitation, PrbsAmplitudeAndHoldAreApplied) {
    control::ident::PrbsOptions options;
    options.order = 4;
    options.amplitude = 2.5;
    options.offset = 1.0;
    options.samples_per_bit = 4;
    const Eigen::VectorXd signal = control::ident::GeneratePrbs(options);
    CHECK_EQ(signal.size(), 15 * 4);
    for (Eigen::Index i = 0; i < signal.size(); ++i) {
        CHECK_TRUE(std::abs(std::abs(signal(i) - 1.0) - 2.5) < 1e-12);
    }
    // Each level is held for exactly samples_per_bit samples.
    for (int bit = 0; bit < 15; ++bit) {
        for (int k = 1; k < 4; ++k) {
            CHECK_NEAR(signal(bit * 4 + k), signal(bit * 4), 0.0);
        }
    }
}

TEST(Excitation, PersistencyDistinguishesNoExcitationFromAPrbs) {
    // The diagnosis this module exists for, and the reason to run it before the
    // experiment. A constant input carries no information about the dynamics:
    // its lagged copies are identical, the regressor is rank one, and no
    // estimator can recover a parameter from it. A pseudo-random sequence moves
    // both directions and supplies the independent directions the parameters
    // need.
    //
    // A note on what is deliberately *not* asserted: a single step transition
    // does make the lagged regressor full rank, because the shifted copies
    // differ in where the transition sits. The classical statement "a step is
    // persistently exciting of order one" is about the expected
    // autocorrelation, not about one finite record. What a step genuinely lacks
    // is frequency content - all of its energy is at DC - and that is a
    // statement about the spectrum, not about the condition number of this
    // matrix. Asserting otherwise would be asserting a property of
    // finite-sample linear algebra that does not hold.
    const Eigen::VectorXd constant = Eigen::VectorXd::Ones(600);
    control::ident::PrbsOptions prbs_options;
    prbs_options.order = 9;
    prbs_options.samples_per_bit = 2;
    prbs_options.bits = 300;
    const Eigen::VectorXd prbs = control::ident::GeneratePrbs(prbs_options);

    const control::ident::PersistencyResult constant_result =
        control::ident::CheckPersistency(constant, 3, 3, 1);
    const control::ident::PersistencyResult prbs_result =
        control::ident::CheckPersistency(prbs, 3, 3, 1);
    std::printf(
        "  constant: condition %.3e (exciting: %d)   prbs: condition %.3e\n",
        constant_result.condition_number,
        static_cast<int>(constant_result.persistently_exciting),
        prbs_result.condition_number);
    CHECK_FALSE(constant_result.persistently_exciting);
    CHECK_TRUE(prbs_result.persistently_exciting);
    // And the actionable part: how much order the input *can* support.
    CHECK_TRUE(constant_result.identifiable_order < 6);
    CHECK_TRUE(prbs_result.identifiable_order >= 4);
    CHECK_TRUE(constant_result.notes.find("does NOT support") !=
               std::string::npos);
}

TEST(Excitation, ChirpAndMultisineCoverTheirBandAndStayInRange) {
    control::ident::ChirpOptions chirp_options;
    chirp_options.start_frequency = 0.1;
    chirp_options.end_frequency = 5.0;
    chirp_options.amplitude = 1.0;
    chirp_options.duration = 10.0;
    chirp_options.sample_time = 0.01;
    const Eigen::VectorXd chirp = control::ident::GenerateChirp(chirp_options);
    CHECK_EQ(chirp.size(), 1000);
    // The peak of a continuous sinusoid generally falls between sample
    // instants, so the largest sample is at most the amplitude and not exactly
    // it. The bound is the worst case at this sample rate rather than an
    // arbitrary slack.
    CHECK_TRUE(chirp.cwiseAbs().maxCoeff() <= 1.0);
    CHECK_TRUE(chirp.cwiseAbs().maxCoeff() > 1.0 - 2.0 * M_PI * 5.0 * 0.01);
    // A logarithmic sweep passes through each decade in equal time, so the
    // number of zero crossings in the second half far exceeds the first.
    int early_crossings = 0;
    int late_crossings = 0;
    for (int k = 1; k < 500; ++k) {
        if ((chirp(k) > 0.0) != (chirp(k - 1) > 0.0)) {
            ++early_crossings;
        }
    }
    for (int k = 501; k < 1000; ++k) {
        if ((chirp(k) > 0.0) != (chirp(k - 1) > 0.0)) {
            ++late_crossings;
        }
    }
    CHECK_TRUE(late_crossings > 3 * early_crossings);

    control::ident::MultisineOptions multisine_options;
    multisine_options.components = 8;
    multisine_options.start_frequency = 0.5;
    multisine_options.end_frequency = 8.0;
    multisine_options.amplitude = 1.0;
    multisine_options.duration = 4.0;
    multisine_options.sample_time = 0.005;
    const Eigen::VectorXd multisine =
        control::ident::GenerateMultisine(multisine_options);
    // Schroeder phasing keeps the crest factor down: without it the peak would
    // approach the sum of the component amplitudes and saturate the actuator.
    const double crest = multisine.cwiseAbs().maxCoeff() / multisine.norm() *
                         std::sqrt(static_cast<double>(multisine.size()));
    std::printf("  multisine crest factor %.3f\n", crest);
    CHECK_TRUE(crest < 3.0);
}

TEST(Excitation, AdviceScalesWithNoiseAndWarnsAboutSaturation) {
    // Ten times the noise floor is the working figure: below it the fit is
    // meaningless, above it there is nothing to gain.
    const control::ident::ExcitationAdvice advice =
        control::ident::AdviseExcitation(2.0, 0.01, 100.0, 10.0, 1.0, 0.01);
    CHECK_NEAR(advice.recommended_amplitude, 10.0 * 0.01 / 2.0, 1e-9);
    CHECK_NEAR(advice.expected_output_swing, 0.1, 1e-9);
    CHECK_TRUE(advice.suggested_samples >= 2000);
    CHECK_EQ(advice.warnings.size(), 0u);

    // A noisy plant with a small actuator cannot clear the noise: the advice
    // says so rather than recommending an amplitude the actuator cannot
    // deliver.
    const control::ident::ExcitationAdvice limited =
        control::ident::AdviseExcitation(0.01, 0.5, 100.0, 1.0, 1.0, 0.01);
    CHECK_TRUE(limited.warnings.size() > 0);
    CHECK_TRUE(limited.recommended_amplitude <= 1.0);
    CHECK_TRUE(limited.warnings[0].find("noise") != std::string::npos);
}
