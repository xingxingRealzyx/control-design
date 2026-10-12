// Copyright 2024 The Control Design Library Authors.
// SPDX-License-Identifier: Apache-2.0
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>

#include "control/ident/dataset.h"
#include "control/ident/excitation.h"
#include "control/ident/ls.h"
#include "control/ident/subspace.h"
#include "control/ident/validate.h"
#include "ident_test_systems.h"
#include "tests/testing.h"

using ident_test::KnownSystem;

TEST(Ls, RecoversTheCoefficientsOfAKnownSystem) {
    // The central test of the module: data from a system whose coefficients are
    // known exactly, identified back, compared against those coefficients.
    const KnownSystem known;
    const control::ident::Dataset data =
        ident_test::MakeTrainingData(known, 2000, 0.0);
    // Without noise the recovery has to be exact to round-off, not "close".
    const control::ident::ArxFitReport report =
        control::ident::FitArx(data, 2, 2, 1);
    CHECK_TRUE(report.ok);
    CHECK_NEAR(report.model.a(0), known.a1, 1e-8);
    CHECK_NEAR(report.model.a(1), known.a2, 1e-8);
    CHECK_NEAR(report.model.b(0), known.b1, 1e-8);
    CHECK_NEAR(report.model.b(1), known.b2, 1e-8);
    // A clean, richly excited experiment is well conditioned.
    CHECK_TRUE(report.regressor_condition_number < 100.0);
    CHECK_EQ(report.warnings.size(), 0u);
}

TEST(Ls, RecoversThePolesDespiteNoise) {
    // With noise the coefficients move - that is what noise does - so the claim
    // is about the poles, which are what a design uses, and about the DC gain.
    const KnownSystem known;
    const control::ident::Dataset data =
        ident_test::MakeTrainingData(known, 4000, 0.05);
    const control::ident::ArxFitReport report =
        control::ident::FitArx(data, 2, 2, 1);

    const Eigen::VectorXcd truth =
        control::ident::ArxPoles(ident_test::Truth(known));
    const Eigen::VectorXcd estimated = control::ident::ArxPoles(report.model);
    CHECK_EQ(estimated.size(), 2);
    for (Eigen::Index i = 0; i < 2; ++i) {
        CHECK_NEAR(std::abs(estimated(i)), std::abs(truth(i)), 0.05);
        CHECK_NEAR(estimated(i).real(), truth(i).real(), 0.05);
    }
    // The DC gain is the most-used number in the model and the one a short
    // record gets wrong first.
    double sum_a = 1.0;
    double sum_b = 0.0;
    for (int i = 0; i < report.model.DenominatorOrder(); ++i) {
        sum_a += report.model.a(i);
    }
    for (int j = 0; j < report.model.NumeratorOrder(); ++j) {
        sum_b += report.model.b(j);
    }
    CHECK_NEAR(sum_b / sum_a,
               (known.b1 + known.b2) / (1.0 + known.a1 + known.a2), 0.15);
}

TEST(Ls, StateSpaceConversionReproducesTheDifferenceEquation) {
    // The state-space form is built by hand in observable canonical form, and a
    // sign or index error there produces a model that is stable, plausible and
    // wrong. The check is that its impulse response equals the one computed
    // from the difference equation the ARX model *is*.
    const KnownSystem known;
    const control::ident::ArxModel model = ident_test::Truth(known);
    const control::ident::StateSpaceModel state_space =
        control::ident::ToStateSpace(model);
    CHECK_EQ(state_space.Order(), 2);

    const int samples = 50;
    const Eigen::VectorXd from_arx =
        control::ident::ArxImpulseResponse(model, samples);
    const Eigen::VectorXd from_state_space =
        control::ident::StateSpaceImpulseResponse(state_space, samples);
    CHECK_NEAR((from_arx - from_state_space).cwiseAbs().maxCoeff(), 0.0, 1e-12);

    // And the poles of A are the roots of the denominator.
    const Eigen::VectorXcd poles = control::ident::ArxPoles(model);
    const Eigen::VectorXcd state_space_poles =
        control::ident::StateSpacePoles(state_space);
    for (Eigen::Index i = 0; i < poles.size(); ++i) {
        CHECK_NEAR(std::abs(poles(i) - state_space_poles(i)), 0.0, 1e-12);
    }
}

TEST(Ls, FrequencyResponseMatchesTheDifferenceEquation) {
    // Evaluated two ways: from the z-transform of the difference equation, and
    // by driving the state-space form with a sinusoid and measuring the
    // amplitude and phase at steady state. The second is what the first claims
    // to predict.
    const KnownSystem known;
    const control::ident::ArxModel model = ident_test::Truth(known);
    const control::ident::StateSpaceModel state_space =
        control::ident::ToStateSpace(model);
    const double omega = 3.0;  // rad/s
    const std::complex<double> analytic =
        control::ident::ArxFrequencyResponse(model, omega);

    const int samples = 4000;
    Eigen::VectorXd input(samples);
    for (int k = 0; k < samples; ++k) {
        input(k) = std::sin(omega * k * model.sample_time);
    }
    Eigen::VectorXd state = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd output(samples);
    for (int k = 0; k < samples; ++k) {
        output(k) =
            (state_space.c * state)(0, 0) + state_space.d(0, 0) * input(k);
        state = state_space.a * state + state_space.b * input(k);
    }
    // Least-squares fit of the last quarter to a sinusoid at the same
    // frequency. The two coefficients are solved jointly rather than by
    // projecting onto each basis function separately: the window does not
    // contain an integer number of periods, so sin and cos are not orthogonal
    // over it and separate projections are biased by about 0.3 percent - larger
    // than the tolerance this test wants, and a bias in the test rather than in
    // the code.
    double sum_ss = 0.0, sum_sc = 0.0, sum_c2 = 0.0, sum_sy = 0.0, sum_cy = 0.0;
    const int begin = 3 * samples / 4;
    for (int k = begin; k < samples; ++k) {
        const double s = std::sin(omega * k * model.sample_time);
        const double c = std::cos(omega * k * model.sample_time);
        sum_ss += s * s;
        sum_sc += s * c;
        sum_c2 += c * c;
        sum_sy += s * output(k);
        sum_cy += c * output(k);
    }
    const double determinant = sum_ss * sum_c2 - sum_sc * sum_sc;
    const double sine_coefficient =
        (sum_c2 * sum_sy - sum_sc * sum_cy) / determinant;
    const double cosine_coefficient =
        (sum_ss * sum_cy - sum_sc * sum_sy) / determinant;
    // With the input sin(w t), the steady-state output is A sin + B cos and the
    // complex gain is A + jB.
    const std::complex<double> measured(sine_coefficient, cosine_coefficient);
    std::printf("  |G| analytic %.6f, measured from a sinusoid %.6f\n",
                std::abs(analytic), std::abs(measured));
    CHECK_NEAR(std::abs(measured), std::abs(analytic), 1e-6);
    CHECK_NEAR(std::arg(measured), std::arg(analytic), 1e-6);
}

TEST(Ls, FreeRunSimulationIsWorseThanOneStepPredictionAndThatIsThePoint) {
    // The one-step prediction uses the measured past; the free-run simulation
    // does not. Reporting the one-step number as "the fit" is the most common
    // way an identification result is overstated, so the difference is asserted
    // to exist.
    const KnownSystem known;
    const control::ident::Dataset data =
        ident_test::MakeTrainingData(known, 1500, 0.1);
    const control::ident::ArxFitReport report =
        control::ident::FitArx(data, 2, 2, 1);
    const Eigen::VectorXd one_step =
        control::ident::PredictArxOneStep(report.model, data);
    const Eigen::VectorXd free_run = control::ident::SimulateArx(
        report.model, data.input, data.sample_time,
        data.output.head(report.model.DenominatorOrder()));
    const double one_step_fit =
        control::ident::ComputeFit(data.output.tail(one_step.size()), one_step)
            .fit_percent;
    const double free_run_fit =
        control::ident::ComputeFit(data.output, free_run).fit_percent;
    std::printf("  one-step fit %.2f %%, free-run fit %.2f %%\n", one_step_fit,
                free_run_fit);
    CHECK_TRUE(one_step_fit > free_run_fit);
    CHECK_TRUE(one_step_fit > 90.0);
}

TEST(Ls, RefusesOrdersAndDelaysItCannotFit) {
    const KnownSystem known;
    const control::ident::Dataset data =
        ident_test::MakeTrainingData(known, 200);
    CHECK_THROWS(control::ident::FitArx(data, 0, 2, 1));
    CHECK_THROWS(control::ident::FitArx(data, 2, 0, 1));
    CHECK_THROWS(control::ident::FitArx(data, 2, 2, 0));
    // More parameters than samples is not a fit, it is an interpolation of
    // noise. 160 parameters for 120 usable rows: this is the case that is
    // genuinely underdetermined. na = nb = 50 leaves 150 rows for 100
    // parameters, which is a legitimate (if ill-conditioned) fit, so the
    // earlier expectation that it would throw was wrong.
    CHECK_THROWS(control::ident::FitArx(data, 80, 80, 1));
}
