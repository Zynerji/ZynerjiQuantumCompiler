/**
 * pybind11 bindings for the ZynerjiQC C++ spectral engine.
 *
 * Exposes:
 * - build_sparse_laplacian() -> scipy-compatible sparse data
 * - compute_eigenvectors() -> (eigenvalues, eigenvectors) numpy arrays
 * - solve_lap() -> (row_indices, col_indices) assignment
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <tuple>

namespace py = pybind11;

// Forward declarations from spectral_engine.cpp
namespace zynerji_qc {
    Eigen::SparseMatrix<double> build_sparse_laplacian(
        const std::vector<int>& adj_rows,
        const std::vector<int>& adj_cols,
        const std::vector<double>& adj_vals,
        int n,
        double handedness,
        double omega,
        double c_log,
        double twist_frac
    );

    struct SpectralResult {
        Eigen::MatrixXd eigenvectors;
        Eigen::VectorXd eigenvalues;
    };

    SpectralResult compute_eigenvectors(
        const Eigen::SparseMatrix<double>& L,
        int k
    );

    std::pair<std::vector<int>, std::vector<int>> solve_lap(
        const Eigen::MatrixXd& cost
    );
}

PYBIND11_MODULE(_zynerji_qc_cpp, m) {
    m.doc() = "ZynerjiQC C++ spectral engine";

    m.def("build_sparse_laplacian",
        [](py::array_t<int> rows, py::array_t<int> cols, py::array_t<double> vals,
           int n, double handedness, double omega, double c_log, double twist_frac)
        -> std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>, int>
        {
            auto r = rows.unchecked<1>();
            auto c = cols.unchecked<1>();
            auto v = vals.unchecked<1>();

            std::vector<int> vr(r.data(0), r.data(0) + r.shape(0));
            std::vector<int> vc(c.data(0), c.data(0) + c.shape(0));
            std::vector<double> vv(v.data(0), v.data(0) + v.shape(0));

            auto L = zynerji_qc::build_sparse_laplacian(
                vr, vc, vv, n, handedness, omega, c_log, twist_frac
            );

            // Convert to COO for scipy
            std::vector<int> out_rows, out_cols;
            std::vector<double> out_vals;
            for (int k = 0; k < L.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
                    out_rows.push_back(it.row());
                    out_cols.push_back(it.col());
                    out_vals.push_back(it.value());
                }
            }

            return std::make_tuple(
                py::array_t<int>(out_rows.size(), out_rows.data()),
                py::array_t<int>(out_cols.size(), out_cols.data()),
                py::array_t<double>(out_vals.size(), out_vals.data()),
                n
            );
        },
        "Build sparse helical Laplacian. Returns (rows, cols, vals, n) for scipy.",
        py::arg("rows"), py::arg("cols"), py::arg("vals"),
        py::arg("n"), py::arg("handedness"),
        py::arg("omega") = 0.3, py::arg("c_log") = 1.0, py::arg("twist_frac") = 0.33
    );

    m.def("compute_eigenvectors",
        [](py::array_t<int> rows, py::array_t<int> cols, py::array_t<double> vals,
           int n, int k)
        -> std::tuple<py::array_t<double>, py::array_t<double>>
        {
            // Reconstruct sparse matrix
            auto r = rows.unchecked<1>();
            auto c = cols.unchecked<1>();
            auto v = vals.unchecked<1>();

            std::vector<Eigen::Triplet<double>> triplets;
            for (py::ssize_t i = 0; i < r.shape(0); ++i) {
                triplets.emplace_back(r(i), c(i), v(i));
            }

            Eigen::SparseMatrix<double> L(n, n);
            L.setFromTriplets(triplets.begin(), triplets.end());

            auto result = zynerji_qc::compute_eigenvectors(L, k);

            return std::make_tuple(
                py::cast(result.eigenvalues),
                py::cast(result.eigenvectors)
            );
        },
        "Compute k smallest non-trivial eigenvectors. Returns (eigenvalues, eigenvectors).",
        py::arg("rows"), py::arg("cols"), py::arg("vals"),
        py::arg("n"), py::arg("k") = 8
    );

    m.def("solve_lap",
        [](py::array_t<double> cost_matrix)
        -> std::tuple<py::array_t<int>, py::array_t<int>>
        {
            auto buf = cost_matrix.unchecked<2>();
            int nr = buf.shape(0);
            int nc = buf.shape(1);

            Eigen::MatrixXd C(nr, nc);
            for (int i = 0; i < nr; ++i)
                for (int j = 0; j < nc; ++j)
                    C(i, j) = buf(i, j);

            auto [row_ind, col_ind] = zynerji_qc::solve_lap(C);

            return std::make_tuple(
                py::array_t<int>(row_ind.size(), row_ind.data()),
                py::array_t<int>(col_ind.size(), col_ind.data())
            );
        },
        "Solve linear assignment problem. Returns (row_indices, col_indices).",
        py::arg("cost_matrix")
    );
}
