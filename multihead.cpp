#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

// Matrix type
typedef vector<vector<float>> Matrix;

// Utility functions for matrix operations
Matrix transpose(const Matrix& mat) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    Matrix transposed(cols, vector<float>(rows));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            transposed[j][i] = mat[i][j];
    return transposed;
}

Matrix matmul(const Matrix& A, const Matrix& B) {
    size_t rows = A.size();
    size_t cols = B[0].size();
    size_t inner = B.size();
    Matrix result(rows, vector<float>(cols, 0));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            for (size_t k = 0; k < inner; ++k)
                result[i][j] += A[i][k] * B[k][j];
    return result;
}

Matrix softmax(const Matrix& mat) {
    Matrix result = mat;
    for (auto& row : result) {
        float sum = 0.0;
        for (float val : row) {
            sum += exp(val);
        }
        for (float& val : row) {
            val = exp(val) / sum;
        }
    }
    return result;
}

// Attention mechanism
Matrix scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V, float scale) {
    Matrix K_transposed = transpose(K);
    Matrix scores = matmul(Q, K_transposed);
    for (auto& row : scores) {
        for (float& val : row) {
            val /= scale;
        }
    }
    Matrix attention_weights = softmax(scores);
    return matmul(attention_weights, V);
}

// Linear layer (for simplicity, use random weights)
class Linear {
    Matrix weights;
    vector<float> biases;

public:
    Linear(size_t input_dim, size_t output_dim) {
        weights = Matrix(input_dim, vector<float>(output_dim));
        biases = vector<float>(output_dim, 0);
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> dist(0, 1);
        for (size_t i = 0; i < input_dim; ++i)
            for (size_t j = 0; j < output_dim; ++j)
                weights[i][j] = dist(gen);
    }

    Matrix forward(const Matrix& input) {
        Matrix output = matmul(input, weights);
        for (size_t i = 0; i < output.size(); ++i)
            for (size_t j = 0; j < output[0].size(); ++j)
                output[i][j] += biases[j];
        return output;
    }
};

// Multihead Attention
class MultiheadAttention {
    size_t num_heads;
    size_t head_dim;
    Linear query_layer, key_layer, value_layer, output_layer;

public:
    MultiheadAttention(size_t input_dim, size_t num_heads)
        : num_heads(num_heads), 
          head_dim(input_dim / num_heads), 
          query_layer(input_dim, input_dim), 
          key_layer(input_dim, input_dim), 
          value_layer(input_dim, input_dim), 
          output_layer(input_dim, input_dim) {}

    Matrix forward(const Matrix& queries, const Matrix& keys, const Matrix& values) {
        // Linear transformations
        Matrix Q = query_layer.forward(queries);
        Matrix K = key_layer.forward(keys);
        Matrix V = value_layer.forward(values);

        // Split into heads
        vector<Matrix> Q_heads(num_heads), K_heads(num_heads), V_heads(num_heads);
        for (size_t h = 0; h < num_heads; ++h) {
            Q_heads[h] = Matrix(Q.size(), vector<float>(head_dim));
            K_heads[h] = Matrix(K.size(), vector<float>(head_dim));
            V_heads[h] = Matrix(V.size(), vector<float>(head_dim));
            for (size_t i = 0; i < Q.size(); ++i) {
                copy(Q[i].begin() + h * head_dim, Q[i].begin() + (h + 1) * head_dim, Q_heads[h][i].begin());
                copy(K[i].begin() + h * head_dim, K[i].begin() + (h + 1) * head_dim, K_heads[h][i].begin());
                copy(V[i].begin() + h * head_dim, V[i].begin() + (h + 1) * head_dim, V_heads[h][i].begin());
            }
        }

        // Attention per head
        vector<Matrix> attention_outputs(num_heads);
        float scale = sqrt(head_dim);
        for (size_t h = 0; h < num_heads; ++h) {
            attention_outputs[h] = scaled_dot_product_attention(Q_heads[h], K_heads[h], V_heads[h], scale);
        }

        // Concatenate heads
        Matrix concatenated(Q.size(), vector<float>(head_dim * num_heads));
        for (size_t i = 0; i < concatenated.size(); ++i) {
            for (size_t h = 0; h < num_heads; ++h) {
                copy(attention_outputs[h][i].begin(), attention_outputs[h][i].end(), concatenated[i].begin() + h * head_dim);
            }
        }

        // Final linear layer
        return output_layer.forward(concatenated);
    }
};

// Main function for testing
int main() {
    size_t input_dim = 8;
    size_t num_heads = 2;
    size_t seq_len = 4;

    // Example input
    Matrix queries(seq_len, vector<float>(input_dim, 0.5));
    Matrix keys(seq_len, vector<float>(input_dim, 0.3));
    Matrix values(seq_len, vector<float>(input_dim, 0.7));

    MultiheadAttention mha(input_dim, num_heads);
    Matrix output = mha.forward(queries, keys, values);

    cout << "Multihead Attention Output:\n";
    for (const auto& row : output) {
        for (float val : row)
            cout << val << " ";
        cout << "\n";
    }

    return 0;
}
