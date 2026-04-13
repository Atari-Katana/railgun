#ifndef ISOQUANT_UTILS_CUH
#define ISOQUANT_UTILS_CUH

#include <cuda_runtime.h>

/*
 * IsoQuant Math Utilities
 * 
 * SO(4) rotations can be decomposed into a left-isoclinic and a right-isoclinic 
 * rotation, both represented by unit quaternions.
 * 
 * For a 4D vector v treated as a quaternion, the SO(4) transform is:
 * T(v) = qL * v * conj(qR)
 */

// Multiply two quaternions q1 and q2.
// Layout: float4(x, y, z, w) where w is the real part.
inline __device__ float4 q_mul(float4 q1, float4 q2) {
    float4 res;
    res.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    res.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    res.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    res.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    return res;
}

// Compute the conjugate of a quaternion.
inline __device__ float4 q_conj(float4 q) {
    return make_float4(-q.x, -q.y, -q.z, q.w);
}

// Apply SO(4) isoclinic transform to a 4D vector v.
// T(v) = qL * v * conj(qR)
inline __device__ float4 apply_isoquant(float4 v, float4 qL, float4 qR) {
    // Left-isoclinic: qL * v
    float4 v_rot_left = q_mul(qL, v);
    // Right-isoclinic: (qL * v) * conj(qR)
    float4 v_rot = q_mul(v_rot_left, q_conj(qR));
    return v_rot;
}

// Derive IsoQuant rotations from a 4D vector v.
// For now, returns identity rotations (no rotation).
inline __device__ void derive_isoquant(float4 v, float4& qL, float4& qR) {
    // Identity quaternion (0, 0, 0, 1)
    qL = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    qR = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
}

#endif // ISOQUANT_UTILS_CUH
