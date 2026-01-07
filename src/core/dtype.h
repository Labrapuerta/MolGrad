#pragma once

#include <cstddef>
#include <string>

enum class Dtype {
    Float32, // 32-bit floating point
    Float64, // 64-bit floating point
    Int32,   // 32-bit integer
    Int64    // 64-bit integer
};

inline size_t dtype_size(Dtype dt) {
    switch (dt) {
        case Dtype::Float32: return 4; 
        case Dtype::Float64: return 8;
        case Dtype::Int32:   return 4;
        case Dtype::Int64:   return 8;
        default:             return 0;
    }
}

inline std::string dtype_to_string(Dtype dt) {
    switch (dt) {
        case Dtype::Float32: return "Float32";
        case Dtype::Float64: return "Float64";
        case Dtype::Int32:   return "Int32";
        case Dtype::Int64:   return "Int64";
        default:             return "Unknown";
    }
}

template<typename T>
constexpr Dtype dtype_of() {
    if constexpr (std::is_same_v<T, float>) {
        return Dtype::Float32;
    } else if constexpr (std::is_same_v<T, double>) {
        return Dtype::Float64;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return Dtype::Int32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return Dtype::Int64;
    } else {
        static_assert(!sizeof(T), "Unsupported dtype");
    }
}


