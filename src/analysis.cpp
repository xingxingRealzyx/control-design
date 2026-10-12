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

#include "control/analysis.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "control/lti.h"

namespace control {
namespace {

// Numerical rank with the threshold taken relative to the largest singular
// value. The absolute alternative is what makes a rank test pass on a model
// scaled by 1e-6 and fail on the same model scaled by 1e6.
RankResult RankOf(const Matrix& matrix, double relative_tolerance) {
    RankResult result;
    result.dimension = static_cast<int>(std::min(matrix.rows(), matrix.cols()));
    if (matrix.rows() == 0 || matrix.cols() == 0) {
        return result;
    }
    const Eigen::JacobiSVD<Matrix> svd(matrix);
    const Vector& singular = svd.singularValues();
    if (singular.size() == 0) {
        return result;
    }
    result.largest_singular_value = singular(0);
    result.smallest_singular_value = singular(singular.size() - 1);
    const double threshold = relative_tolerance * singular(0);
    int rank = 0;
    for (Eigen::Index i = 0; i < singular.size(); ++i) {
        if (singular(i) > threshold) {
            ++rank;
        }
    }
    result.rank = rank;
    return result;
}

// Orthonormal basis of the range of a matrix, with the numerical rank decided
// by the same relative threshold. Used for the Kalman subspaces: the columns of
// the controllability matrix span the reachable subspace but are horribly
// conditioned, so they are replaced by an orthonormal basis of the same space.
struct OrthonormalBasis {
    Matrix basis;  // rows() = ambient dimension, cols() = subspace dimension
    int dimension = 0;
    double smallest_kept = 0.0;
    double largest = 0.0;
};

OrthonormalBasis RangeBasis(const Matrix& matrix, double relative_tolerance) {
    OrthonormalBasis result;
    if (matrix.rows() == 0 || matrix.cols() == 0) {
        // An empty subspace is a legitimate answer here, not an error: it means
        // "no state is in this category".
        result.basis = Matrix::Zero(matrix.rows(), 0);
        return result;
    }
    const Eigen::JacobiSVD<Matrix> svd(
        matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Vector& singular = svd.singularValues();
    if (singular.size() == 0) {
        result.basis = Matrix::Zero(matrix.rows(), 0);
        return result;
    }
    result.largest = singular(0);
    const double threshold = relative_tolerance * singular(0);
    int count = 0;
    for (Eigen::Index i = 0; i < singular.size(); ++i) {
        if (singular(i) > threshold) {
            ++count;
        }
    }
    result.dimension = count;
    result.smallest_kept = count > 0 ? singular(count - 1) : 0.0;
    // Slice to exactly `count` columns. Eigen's thin U can carry more columns
    // than the requested count (for a rank-deficient input it still returns
    // min(rows, cols) of them), and a basis whose column count disagrees with
    // its declared dimension silently corrupts every later projection built
    // from it.
    result.basis = Matrix(svd.matrixU().leftCols(count));
    if (result.basis.cols() != result.dimension) {
        result.basis.conservativeResize(result.basis.rows(), result.dimension);
    }
    return result;
}

// Orthonormal basis of range(first) intersect range(second).
//
// The method is the obvious one: project the second basis onto the first, and
// keep the directions whose component *outside* the first subspace is
// negligible. An earlier version intersected by stacking annihilators and
// taking the null space, which is numerically delicate and returned a
// one-dimensional intersection for two subspaces that meet only at the origin.
OrthonormalBasis IntersectRanges(const OrthonormalBasis& first,
                                 const OrthonormalBasis& second,
                                 int ambient_dimension,
                                 double relative_tolerance) {
    OrthonormalBasis result;
    if (first.dimension == 0 || second.dimension == 0) {
        result.basis = Matrix::Zero(ambient_dimension, 0);
        return result;
    }
    const Matrix projector = first.basis * first.basis.transpose();
    const Matrix projected = projector * second.basis;
    // Guard first: a relative rank test on a nearly zero matrix always finds a
    // unit direction in it (its own largest singular value is the scale), so
    // "the projection is negligible" has to be decided *before* asking for a
    // basis, against the size of the subspace being projected. Without this the
    // intersection of two subspaces that meet only at the origin comes back
    // one-dimensional.
    const double projected_scale = projected.cwiseAbs().maxCoeff();
    const double reference_scale =
        std::max(1.0, second.basis.cwiseAbs().maxCoeff());
    if (projected_scale <= relative_tolerance * reference_scale) {
        result.basis = Matrix::Zero(ambient_dimension, 0);
        return result;
    }
    const OrthonormalBasis candidates =
        RangeBasis(projected, relative_tolerance);

    Matrix kept(ambient_dimension, candidates.dimension);
    int kept_count = 0;
    for (int i = 0; i < candidates.dimension; ++i) {
        // Both the test and the copy use column i: an earlier version tested
        // column i but copied column j, which kept all of them and produced
        // four subspaces whose dimensions summed to more than n.
        const Vector direction = candidates.basis.col(i);
        const Vector residual = direction - projector * direction;
        if (residual.norm() <= 1e-7 * std::max(1.0, direction.norm())) {
            kept.col(kept_count) = direction;
            ++kept_count;
        }
    }
    result.basis = kept.leftCols(kept_count);
    result.dimension = kept_count;
    return result;
}

// Orthonormal basis of the orthogonal complement of `basis` inside R^n. Built
// from a full SVD of the basis: the left singular vectors beyond the first
// `basis.cols()` columns span the complement.
Matrix OrthogonalComplement(const Matrix& basis, int ambient_dimension) {
    const int kept = static_cast<int>(basis.cols());
    if (kept >= ambient_dimension || basis.rows() == 0) {
        return Matrix::Zero(ambient_dimension, 0);
    }
    Eigen::JacobiSVD<Matrix> svd(basis, Eigen::ComputeFullU);
    return svd.matrixU().rightCols(ambient_dimension - kept);
}

bool ModeIsStable(const std::complex<double>& pole, bool continuous) {
    if (continuous) {
        return pole.real() < 0.0;
    }
    return std::abs(pole) < 1.0;
}

std::string FormatMode(const std::complex<double>& pole) {
    std::ostringstream text;
    text.precision(4);
    text << std::fixed << pole.real();
    if (std::abs(pole.imag()) > 1e-12) {
        text << (pole.imag() >= 0 ? " + " : " - ") << std::abs(pole.imag())
             << "j";
    }
    return text.str();
}

}  // namespace

Matrix ControllabilityMatrix(const LtiSystem& system) {
    system.Validate("ControllabilityMatrix");
    const int n = system.state_dimension();
    Matrix result(n, n * system.input_dimension());
    Matrix power = Matrix::Identity(n, n);
    for (int i = 0; i < n; ++i) {
        result.block(0, i * system.input_dimension(), n,
                     system.input_dimension()) = power * system.b();
        power = power * system.a();
    }
    return result;
}

Matrix ObservabilityMatrix(const LtiSystem& system) {
    system.Validate("ObservabilityMatrix");
    const int n = system.state_dimension();
    Matrix result(n * system.output_dimension(), n);
    Matrix power = Matrix::Identity(n, n);
    for (int i = 0; i < n; ++i) {
        result.block(i * system.output_dimension(), 0,
                     system.output_dimension(), n) = system.c() * power;
        power = power * system.a();
    }
    return result;
}

double PbhControllabilityMeasure(const LtiSystem& system,
                                 std::complex<double> pole) {
    const int n = system.state_dimension();
    Eigen::MatrixXcd shifted = pole * Eigen::MatrixXcd::Identity(n, n) -
                               system.a().cast<std::complex<double>>();
    Eigen::MatrixXcd augmented(n, n + system.input_dimension());
    augmented.leftCols(n) = shifted;
    augmented.rightCols(system.input_dimension()) =
        system.b().cast<std::complex<double>>();
    const Eigen::JacobiSVD<Eigen::MatrixXcd> svd(augmented);
    const Vector& singular = svd.singularValues();
    return singular.size() == 0 ? 0.0 : singular(singular.size() - 1);
}

double PbhObservabilityMeasure(const LtiSystem& system,
                               std::complex<double> pole) {
    const int n = system.state_dimension();
    Eigen::MatrixXcd shifted = pole * Eigen::MatrixXcd::Identity(n, n) -
                               system.a().cast<std::complex<double>>();
    Eigen::MatrixXcd augmented(n + system.output_dimension(), n);
    augmented.topRows(n) = shifted;
    augmented.bottomRows(system.output_dimension()) =
        system.c().cast<std::complex<double>>();
    const Eigen::JacobiSVD<Eigen::MatrixXcd> svd(augmented);
    const Vector& singular = svd.singularValues();
    return singular.size() == 0 ? 0.0 : singular(singular.size() - 1);
}

ControllabilityResult AnalyseControllability(const LtiSystem& system,
                                             const AnalysisOptions& options) {
    system.Validate("AnalyseControllability");
    ControllabilityResult result;

    const Matrix reachability = ControllabilityMatrix(system);
    result.rank = RankOf(reachability, options.relative_tolerance);
    result.rank.dimension = system.state_dimension();  // rank is capped at n
    result.fully_controllable = result.rank.rank == system.state_dimension();
    result.marginal =
        result.fully_controllable &&
        result.rank.smallest_singular_value <
            options.marginal_ratio * result.rank.largest_singular_value;

    // The PBH test, per mode, decides which eigenvalues are unreachable. Going
    // through the modes rather than the rank is what makes the answer
    // actionable: it points at the eigenvalue that cannot be moved.
    const ComplexVector poles = Poles(system);
    std::vector<std::pair<std::complex<double>, double>> classified;
    for (Eigen::Index i = 0; i < poles.size(); ++i) {
        const double measure = PbhControllabilityMeasure(system, poles(i));
        result.pbh_measures.conservativeResize(result.pbh_measures.size() + 1);
        result.pbh_measures(result.pbh_measures.size() - 1) = measure;
        const double scale = std::max(1.0, system.a().cwiseAbs().maxCoeff());
        if (measure <= options.relative_tolerance * scale) {
            classified.emplace_back(poles(i), measure);
        }
    }

    bool has_unstable_unreachable = false;
    for (const auto& entry : classified) {
        result.uncontrollable_modes.conservativeResize(
            result.uncontrollable_modes.size() + 1);
        result.uncontrollable_modes(result.uncontrollable_modes.size() - 1) =
            entry.first;
        if (!ModeIsStable(entry.first, system.is_continuous())) {
            has_unstable_unreachable = true;
        }
    }
    // Sorted by descending real part (continuous) / magnitude (discrete) so the
    // dangerous mode is reported first.
    if (result.uncontrollable_modes.size() > 1) {
        std::sort(result.uncontrollable_modes.data(),
                  result.uncontrollable_modes.data() +
                      result.uncontrollable_modes.size(),
                  [&system](const std::complex<double>& left,
                            const std::complex<double>& right) {
                      return system.is_continuous()
                                 ? left.real() > right.real()
                                 : std::abs(left) > std::abs(right);
                  });
    }
    result.stabilizable = !has_unstable_unreachable;

    if (result.fully_controllable && !result.marginal) {
        result.notes =
            "fully controllable: arbitrary pole placement is available";
    } else if (result.fully_controllable) {
        result.notes = "rank is full but only marginally (condition number " +
                       std::to_string(result.rank.condition_number()) +
                       "): pole placement will ask for very large gains";
    } else if (result.stabilizable) {
        result.notes =
            "not fully controllable, but every unreachable mode is stable: a "
            "stabilising controller exists; the unreachable states cannot be "
            "moved, only left to decay";
    } else {
        result.notes =
            "an unstable mode is unreachable by the input: no controller fixes "
            "this. The actuator cannot influence that state; change the "
            "actuator "
            "or the model";
    }
    return result;
}

ObservabilityResult AnalyseObservability(const LtiSystem& system,
                                         const AnalysisOptions& options) {
    system.Validate("AnalyseObservability");
    ObservabilityResult result;

    const Matrix observability = ObservabilityMatrix(system);
    result.rank = RankOf(observability, options.relative_tolerance);
    result.rank.dimension = system.state_dimension();
    result.fully_observable = result.rank.rank == system.state_dimension();
    result.marginal =
        result.fully_observable &&
        result.rank.smallest_singular_value <
            options.marginal_ratio * result.rank.largest_singular_value;

    const ComplexVector poles = Poles(system);
    const double scale = std::max(1.0, system.a().cwiseAbs().maxCoeff());
    bool has_unstable_unobservable = false;
    for (Eigen::Index i = 0; i < poles.size(); ++i) {
        const double measure = PbhObservabilityMeasure(system, poles(i));
        result.pbh_measures.conservativeResize(result.pbh_measures.size() + 1);
        result.pbh_measures(result.pbh_measures.size() - 1) = measure;
        if (measure <= options.relative_tolerance * scale) {
            result.unobservable_modes.conservativeResize(
                result.unobservable_modes.size() + 1);
            result.unobservable_modes(result.unobservable_modes.size() - 1) =
                poles(i);
            if (!ModeIsStable(poles(i), system.is_continuous())) {
                has_unstable_unobservable = true;
            }
        }
    }
    if (result.unobservable_modes.size() > 1) {
        std::sort(
            result.unobservable_modes.data(),
            result.unobservable_modes.data() + result.unobservable_modes.size(),
            [&system](const std::complex<double>& left,
                      const std::complex<double>& right) {
                return system.is_continuous()
                           ? left.real() > right.real()
                           : std::abs(left) > std::abs(right);
            });
    }
    result.detectable = !has_unstable_unobservable;

    if (result.fully_observable && !result.marginal) {
        result.notes =
            "fully observable: an observer with arbitrary dynamics exists";
    } else if (result.fully_observable) {
        result.notes = "rank is full but only marginally (condition number " +
                       std::to_string(result.rank.condition_number()) +
                       "): observer gains will be very large";
    } else if (result.detectable) {
        result.notes =
            "not fully observable, but every unobservable mode is stable: a "
            "detectable system, so an observer exists and the unobservable "
            "states "
            "decay on their own";
    } else {
        result.notes =
            "an unstable mode is unobservable: it cannot be seen, so no "
            "observer "
            "can estimate it. Add a sensor or change the model";
    }
    return result;
}

KalmanDecomposition DecomposeKalman(const LtiSystem& system,
                                    const AnalysisOptions& options) {
    system.Validate("DecomposeKalman");
    const int n = system.state_dimension();

    // Two subspaces generate the whole decomposition:
    //   R = range of the controllability matrix   (reachable states)
    //   O = range of the observability matrix     (visible states)
    // The four Kalman subspaces are R and its complement crossed with O and its
    // complement, and each is returned as an orthonormal basis so that the four
    // together form a similarity transformation a design can be carried out in.
    const OrthonormalBasis reachable =
        RangeBasis(ControllabilityMatrix(system), options.relative_tolerance);
    const OrthonormalBasis visible = RangeBasis(
        ObservabilityMatrix(system).transpose(), options.relative_tolerance);
    const Matrix unreachable_basis = OrthogonalComplement(reachable.basis, n);
    const Matrix invisible_basis = OrthogonalComplement(visible.basis, n);
    const OrthonormalBasis unreachable{
        unreachable_basis, static_cast<int>(unreachable_basis.cols()), 0.0,
        0.0};
    const OrthonormalBasis invisible{
        invisible_basis, static_cast<int>(invisible_basis.cols()), 0.0, 0.0};

    const OrthonormalBasis controllable_observable =
        IntersectRanges(reachable, visible, n, options.relative_tolerance);
    const OrthonormalBasis controllable_unobservable =
        IntersectRanges(reachable, invisible, n, options.relative_tolerance);
    const OrthonormalBasis uncontrollable_observable =
        IntersectRanges(unreachable, visible, n, options.relative_tolerance);
    const OrthonormalBasis uncontrollable_unobservable =
        IntersectRanges(unreachable, invisible, n, options.relative_tolerance);

    KalmanDecomposition decomposition;
    decomposition.controllable_observable = controllable_observable.basis;
    decomposition.controllable_unobservable = controllable_unobservable.basis;
    decomposition.uncontrollable_observable = uncontrollable_observable.basis;
    decomposition.uncontrollable_unobservable =
        uncontrollable_unobservable.basis;
    decomposition.controllable_observable_dimension =
        controllable_observable.dimension;
    decomposition.controllable_unobservable_dimension =
        controllable_unobservable.dimension;
    decomposition.uncontrollable_observable_dimension =
        uncontrollable_observable.dimension;
    decomposition.uncontrollable_unobservable_dimension =
        uncontrollable_unobservable.dimension;
    decomposition.spans_state_space =
        controllable_observable.dimension +
            controllable_unobservable.dimension +
            uncontrollable_observable.dimension +
            uncontrollable_unobservable.dimension ==
        n;
    return decomposition;
}

LtiSystem MinimalRealisation(const LtiSystem& system,
                             const AnalysisOptions& options) {
    system.Validate("MinimalRealisation");
    const KalmanDecomposition decomposition = DecomposeKalman(system, options);
    const int order = decomposition.minimal_order();
    if (order == 0) {
        throw std::invalid_argument(
            "MinimalRealisation: the model has no controllable and observable "
            "state, so its input-output behaviour is identically zero");
    }
    if (order == system.state_dimension()) {
        return system;  // already minimal
    }

    // Restrict to the reachable-and-observable subspace. Qco has orthonormal
    // columns, so the projection is a similarity transformation and the
    // input-output behaviour is unchanged.
    const Matrix projected_a =
        decomposition.controllable_observable.transpose() * system.a() *
        decomposition.controllable_observable;
    const Matrix projected_b =
        decomposition.controllable_observable.transpose() * system.b();
    const Matrix projected_c =
        system.c() * decomposition.controllable_observable;
    const LtiSystem projected(projected_a, projected_b, projected_c, system.d(),
                              system.sample_time());

    // Orthonormalise once more inside the projected model: the observable
    // directions within it can still be badly conditioned even though the
    // ambient frame is orthonormal.
    const OrthonormalBasis minimal = RangeBasis(
        ObservabilityMatrix(projected).transpose(), options.relative_tolerance);
    const int minimal_order = minimal.dimension;
    if (minimal_order == 0) {
        throw std::invalid_argument(
            "MinimalRealisation: the projected model has no observable state");
    }
    const Matrix a_reduced =
        minimal.basis.transpose() * projected_a * minimal.basis;
    const Matrix b_reduced = minimal.basis.transpose() * projected_b;
    const Matrix c_reduced = projected_c * minimal.basis;
    return system.is_continuous()
               ? LtiSystem(a_reduced, b_reduced, c_reduced, system.d(), 0.0)
               : LtiSystem(a_reduced, b_reduced, c_reduced, system.d(),
                           system.sample_time());
}

std::string DescribeControllability(const ControllabilityResult& result) {
    std::ostringstream text;
    text << "rank " << result.rank.rank << "/" << result.rank.dimension;
    if (result.rank.largest_singular_value > 0.0) {
        text << " (condition number " << result.rank.condition_number() << ")";
    }
    text << ": " << result.notes;
    if (result.uncontrollable_modes.size() > 0) {
        text << "\n  unreachable modes:";
        for (Eigen::Index i = 0; i < result.uncontrollable_modes.size(); ++i) {
            text << " " << FormatMode(result.uncontrollable_modes(i));
        }
    }
    return text.str();
}

std::string DescribeObservability(const ObservabilityResult& result) {
    std::ostringstream text;
    text << "rank " << result.rank.rank << "/" << result.rank.dimension;
    if (result.rank.largest_singular_value > 0.0) {
        text << " (condition number " << result.rank.condition_number() << ")";
    }
    text << ": " << result.notes;
    if (result.unobservable_modes.size() > 0) {
        text << "\n  invisible modes:";
        for (Eigen::Index i = 0; i < result.unobservable_modes.size(); ++i) {
            text << " " << FormatMode(result.unobservable_modes(i));
        }
    }
    return text.str();
}

}  // namespace control
