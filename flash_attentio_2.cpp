#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

using namespace std;

// Define a matrix type for convenience
typedef vector<vector<float>> Matrix;

// Initialize a matrix with zeros
Matrix initialize_matrix(size_t rows, size_t cols) {
    return Matrix(rows, vector<float>(cols, 0.0f));
}

// Display a matrix
void print_matrix(const Matrix& mat) {
    for (const auto& row : mat) {
        for (float val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

// Matrix multiplication
Matrix matmul(const Matrix& A, const Matrix& B) {
    size_t rows = A.size();
    size_t cols = B[0].size();
    size_t inner = A[0].size();
    Matrix result = initialize_matrix(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < inner; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Transpose a matrix
Matrix transpose(const Matrix& mat) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    Matrix transposed = initialize_matrix(cols, rows);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = mat[i][j];
        }
    }
    return transposed;
}

// Stable softmax function (row-wise)
Matrix softmax(const Matrix& mat) {
    Matrix result = mat;

    for (size_t i = 0; i < mat.size(); ++i) {
        float max_val = *max_element(mat[i].begin(), mat[i].end());
        float sum_exp = 0.0;

        for (float val : mat[i]) {
            sum_exp += exp(val - max_val);
        }

        for (size_t j = 0; j < mat[i].size(); ++j) {
            result[i][j] = exp(mat[i][j] - max_val) / sum_exp;
        }
    }
    return result;
}

// Flash Attention v2: Attention mechanism with chunking
Matrix flash_attention(const Matrix& Q, const Matrix& K, const Matrix& V, size_t chunk_size) {
    size_t seq_len = Q.size();
    size_t head_dim = Q[0].size();

    Matrix output = initialize_matrix(seq_len, head_dim);

    for (size_t chunk_start = 0; chunk_start < seq_len; chunk_start += chunk_size) {
        size_t chunk_end = min(chunk_start + chunk_size, seq_len);

        // Extract chunks
        Matrix Q_chunk(Q.begin() + chunk_start, Q.begin() + chunk_end);
        Matrix K_chunk(K.begin(), K.begin() + chunk_end); // Use keys up to the chunk end
        Matrix V_chunk(V.begin(), V.begin() + chunk_end);

        // Compute scaled attention scores
        Matrix K_transposed = transpose(K_chunk);
        Matrix scores = matmul(Q_chunk, K_transposed);

        for (auto& row : scores) {
            for (float& val : row) {
                val /= sqrt(head_dim);
            }
        }

        // Apply softmax to scores
        Matrix weights = softmax(scores);

        // Compute output for this chunk
        Matrix chunk_output = matmul(weights, V_chunk);

        // Add chunk output to the corresponding positions
        for (size_t i = chunk_start, local_idx = 0; i < chunk_end; ++i, ++local_idx) {
            output[i] = chunk_output[local_idx];
        }
    }
    return output;
}

// Main function for testing
int main() {
    size_t seq_len = 8;    // Sequence length
    size_t head_dim = 4;   // Dimension of each head
    size_t chunk_size = 4; // Chunk size for Flash Attention

    // Example Q, K, V matrices
    Matrix Q = {
        {0.1, 0.2, 0.3, 0.4},
        {0.2, 0.3, 0.4, 0.5},
        {0.3, 0.4, 0.5, 0.6},
        {0.4, 0.5, 0.6, 0.7},
        {0.5, 0.6, 0.7, 0.8},
        {0.6, 0.7, 0.8, 0.9},
        {0.7, 0.8, 0.9, 1.0},
        {0.8, 0.9, 1.0, 1.1}
    };
    Matrix K = Q; // Symmetric keys
    Matrix V = Q; // Symmetric values

    cout << "Input Q Matrix:" << endl;
    print_matrix(Q);
    cout << endl;

    // Perform Flash Attention
    Matrix output = flash_attention(Q, K, V, chunk_size);

    cout << "Flash Attention Output:" << endl;
    print_matrix(output);

    return 0;
}
