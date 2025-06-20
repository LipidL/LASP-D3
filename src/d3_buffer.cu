#include <cmath>

#include "d3_buffer.cuh"

// Calculate inverse of a 3x3 matrix
void matrix_inverse(const real_t mat[3][3], real_t inv[3][3]) {
    // Calculate determinant
    real_t det = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
              - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
              + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
    
              real_t inv_det = 1.0 / det;
    
    // Calculate cofactor matrix (transposed)
    inv[0][0] = (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) * inv_det;
    inv[0][1] = (mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2]) * inv_det;
    inv[0][2] = (mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1]) * inv_det;
    
    inv[1][0] = (mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2]) * inv_det;
    inv[1][1] = (mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0]) * inv_det;
    inv[1][2] = (mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) * inv_det;
    
    inv[2][0] = (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]) * inv_det;
    inv[2][1] = (mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1]) * inv_det;
    inv[2][2] = (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]) * inv_det;
}

// Transpose a 3x3 matrix
void matrix_transpose(const real_t mat[3][3], real_t trans[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            trans[j][i] = mat[i][j];
        }
    }
}

// Calculate row-wise norm of a 3x3 matrix
void row_norms(const real_t mat[3][3], real_t norms[3]) {
    for (int i = 0; i < 3; i++) {
        norms[i] = sqrt(mat[i][0] * mat[i][0] + 
                         mat[i][1] * mat[i][1] + 
                         mat[i][2] * mat[i][2]);
    }
}

// Equivalent to torch.ceil(cutoff * inv_distances).long()
void calculate_cell_repeats(real_t cell[3][3], real_t cutoff, size_t max_cell_bias[3]) {
    real_t inv[3][3];
    real_t trans[3][3];
    real_t norms[3];
    
    // Calculate inverse of cell matrix
    matrix_inverse(cell, inv);
    
    // Transpose the inverse matrix
    matrix_transpose(inv, trans);
    
    // Calculate norms of each row
    row_norms(trans, norms);
      // Multiply by cutoff and round up to nearest integer
    for (int i = 0; i < 3; i++) {
        max_cell_bias[i] = ((int)(cutoff * norms[i]) + 1) * 2 + 1; // the number of repeats need to be timed by 2 due to two directions, and add 1 due to the central unit (no translation at all)
    }
}

