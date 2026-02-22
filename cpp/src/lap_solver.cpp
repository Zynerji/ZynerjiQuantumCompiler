/**
 * Jonker-Volgenant Linear Assignment Problem (LAP) solver.
 *
 * O(n^3) solver for the rectangular assignment problem.
 * Fast for n < 5000 (typical quantum device sizes).
 *
 * For now, delegates to a simple Hungarian implementation.
 * Can be replaced with a full JV solver for better performance.
 */

#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

namespace zynerji_qc {

/**
 * Solve the linear assignment problem (rectangular cost matrix).
 *
 * Given cost matrix C[i,j], find assignment that minimizes total cost.
 * Handles n_rows <= n_cols (more physical than logical qubits).
 *
 * @param cost  Cost matrix (n_rows x n_cols), n_rows <= n_cols
 * @return pair of (row_indices, col_indices) for optimal assignment
 */
std::pair<std::vector<int>, std::vector<int>> solve_lap(
    const Eigen::MatrixXd& cost
) {
    int n_rows = cost.rows();
    int n_cols = cost.cols();

    if (n_rows == 0 || n_cols == 0) {
        return {{}, {}};
    }

    // Pad to square if rectangular
    int n = std::max(n_rows, n_cols);
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n, n);
    C.block(0, 0, n_rows, n_cols) = cost;

    // Hungarian algorithm (Kuhn-Munkres)
    const double INF = std::numeric_limits<double>::infinity();

    // Step 1: Row reduction
    for (int i = 0; i < n; ++i) {
        double row_min = C.row(i).minCoeff();
        C.row(i).array() -= row_min;
    }

    // Step 2: Column reduction
    for (int j = 0; j < n; ++j) {
        double col_min = C.col(j).minCoeff();
        C.col(j).array() -= col_min;
    }

    // Step 3-5: Augmenting path algorithm
    std::vector<int> u(n + 1, 0), v(n + 1, 0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);

    // Re-run with original cost for the proper Hungarian
    C = Eigen::MatrixXd::Zero(n, n);
    C.block(0, 0, n_rows, n_cols) = cost;

    // Standard Hungarian with potentials
    std::vector<double> U(n + 1, 0), V(n + 1, 0);
    std::vector<int> assignment(n + 1, 0);
    std::vector<int> inv_assignment(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        std::vector<double> dist(n + 1, INF);
        std::vector<bool> used(n + 1, false);
        assignment[0] = i;
        int j0 = 0;

        do {
            used[j0] = true;
            int i0 = assignment[j0];
            double delta = INF;
            int j1 = -1;

            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    double val = C(i0 - 1, j - 1) - U[i0] - V[j];
                    if (val < dist[j]) {
                        dist[j] = val;
                        way[j] = j0;
                    }
                    if (dist[j] < delta) {
                        delta = dist[j];
                        j1 = j;
                    }
                }
            }

            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    U[assignment[j]] += delta;
                    V[j] -= delta;
                } else {
                    dist[j] -= delta;
                }
            }

            j0 = j1;
        } while (assignment[j0] != 0);

        // Trace back
        do {
            int j1 = way[j0];
            assignment[j0] = assignment[j1];
            j0 = j1;
        } while (j0);
    }

    // Extract result (only first n_rows assignments)
    std::vector<int> row_ind, col_ind;
    for (int j = 1; j <= n; ++j) {
        int i = assignment[j] - 1;
        if (i < n_rows && (j - 1) < n_cols) {
            row_ind.push_back(i);
            col_ind.push_back(j - 1);
        }
    }

    return {row_ind, col_ind};
}

}  // namespace zynerji_qc
