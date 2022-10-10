//
// Implementation for Yocto/Mesh
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2021 Fabio Pellacini
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

// TODO(fabio): remove asserts
// TODO(fabio): better name for v2t
// TODO(fabio): better name for adjacencies

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------

#include "yocto_mesh.h"

#include <cassert>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <yocto/yocto_geometry.h>
#include <yocto/yocto_modelio.h>

// #define TESTS_MAY_FAIL

#define YOCTO_BEZIER_PRECISE 1
#define YOCTO_BEZIER_DOUBLE 1
#define UNFOLD_INTERSECTING_CIRCLES 0

// -----------------------------------------------------------------------------
// USING DIRECTIVES
// -----------------------------------------------------------------------------
namespace yocto {

// using directives
using std::deque;
using std::pair;
using std::unordered_set;
using namespace std::string_literals;

}  // namespace yocto

// -----------------------------------------------------------------------------
// MATH FUNCTIONS IN DOUBLE
// -----------------------------------------------------------------------------
namespace yocto {

constexpr auto dbl_max = std::numeric_limits<double>::max();
constexpr auto dbl_min = std::numeric_limits<double>::lowest();
constexpr auto dbl_eps = std::numeric_limits<double>::epsilon();

inline double abs(double a);
inline double min(double a, double b);
inline double max(double a, double b);
inline double clamp(double a, double min, double max);
inline double sign(double a);
inline double sqr(double a);
inline double sqrt(double a);
inline double sin(double a);
inline double cos(double a);
inline double tan(double a);
inline double asin(double a);
inline double acos(double a);
inline double atan(double a);
inline double log(double a);
inline double exp(double a);
inline double log2(double a);
inline double exp2(double a);
inline double pow(double a, double b);
inline bool   isfinite(double a);
inline double atan2(double a, double b);
inline double fmod(double a, double b);
inline double radians(double a);
inline double degrees(double a);
inline double lerp(double a, double b, double u);
inline void   swap(double& a, double& b);
inline double smoothstep(double a, double b, double u);
inline double bias(double a, double bias);
inline double gain(double a, double gain);

}  // namespace yocto

// -----------------------------------------------------------------------------
// VECTORS
// -----------------------------------------------------------------------------
namespace yocto {

struct vec2d {
  double x = 0;
  double y = 0;

  double&       operator[](int i);
  const double& operator[](int i) const;
};

struct vec3d {
  double x = 0;
  double y = 0;
  double z = 0;

  double&       operator[](int i);
  const double& operator[](int i) const;
};

struct vec4d {
  double x = 0;
  double y = 0;
  double z = 0;
  double w = 0;

  double&       operator[](int i);
  const double& operator[](int i) const;
};

// Zero vector constants.
[[deprecated]] constexpr auto zero2d = vec2d{0, 0};
[[deprecated]] constexpr auto zero3d = vec3d{0, 0, 0};
[[deprecated]] constexpr auto zero4d = vec4d{0, 0, 0, 0};

// One vector constants.
[[deprecated]] constexpr auto one2d = vec2d{1, 1};
[[deprecated]] constexpr auto one3d = vec3d{1, 1, 1};
[[deprecated]] constexpr auto one4d = vec4d{1, 1, 1, 1};

// Element access
inline vec3d xyz(const vec4d& a);

// Vector sequence operations.
inline int           size(const vec2d& a);
inline const double* begin(const vec2d& a);
inline const double* end(const vec2d& a);
inline double*       begin(vec2d& a);
inline double*       end(vec2d& a);
inline const double* data(const vec2d& a);
inline double*       data(vec2d& a);

// Vector comparison operations.
inline bool operator==(const vec2d& a, const vec2d& b);
inline bool operator!=(const vec2d& a, const vec2d& b);

// Vector operations.
inline vec2d operator+(const vec2d& a);
inline vec2d operator-(const vec2d& a);
inline vec2d operator+(const vec2d& a, const vec2d& b);
inline vec2d operator+(const vec2d& a, double b);
inline vec2d operator+(double a, const vec2d& b);
inline vec2d operator-(const vec2d& a, const vec2d& b);
inline vec2d operator-(const vec2d& a, double b);
inline vec2d operator-(double a, const vec2d& b);
inline vec2d operator*(const vec2d& a, const vec2d& b);
inline vec2d operator*(const vec2d& a, double b);
inline vec2d operator*(double a, const vec2d& b);
inline vec2d operator/(const vec2d& a, const vec2d& b);
inline vec2d operator/(const vec2d& a, double b);
inline vec2d operator/(double a, const vec2d& b);

// Vector assignments
inline vec2d& operator+=(vec2d& a, const vec2d& b);
inline vec2d& operator+=(vec2d& a, double b);
inline vec2d& operator-=(vec2d& a, const vec2d& b);
inline vec2d& operator-=(vec2d& a, double b);
inline vec2d& operator*=(vec2d& a, const vec2d& b);
inline vec2d& operator*=(vec2d& a, double b);
inline vec2d& operator/=(vec2d& a, const vec2d& b);
inline vec2d& operator/=(vec2d& a, double b);

// Vector products and lengths.
inline double dot(const vec2d& a, const vec2d& b);
inline double cross(const vec2d& a, const vec2d& b);

inline double length(const vec2d& a);
inline double length_squared(const vec2d& a);
inline vec2d  normalize(const vec2d& a);
inline double distance(const vec2d& a, const vec2d& b);
inline double distance_squared(const vec2d& a, const vec2d& b);
inline double angle(const vec2d& a, const vec2d& b);

// Max element and clamp.
inline vec2d max(const vec2d& a, double b);
inline vec2d min(const vec2d& a, double b);
inline vec2d max(const vec2d& a, const vec2d& b);
inline vec2d min(const vec2d& a, const vec2d& b);
inline vec2d clamp(const vec2d& x, double min, double max);
inline vec2d lerp(const vec2d& a, const vec2d& b, double u);
inline vec2d lerp(const vec2d& a, const vec2d& b, const vec2d& u);

inline double max(const vec2d& a);
inline double min(const vec2d& a);
inline double sum(const vec2d& a);
inline double mean(const vec2d& a);

// Functions applied to vector elements
inline vec2d abs(const vec2d& a);
inline vec2d sqr(const vec2d& a);
inline vec2d sqrt(const vec2d& a);
inline vec2d exp(const vec2d& a);
inline vec2d log(const vec2d& a);
inline vec2d exp2(const vec2d& a);
inline vec2d log2(const vec2d& a);
inline bool  isfinite(const vec2d& a);
inline vec2d pow(const vec2d& a, double b);
inline vec2d pow(const vec2d& a, const vec2d& b);
inline vec2d gain(const vec2d& a, double b);
inline void  swap(vec2d& a, vec2d& b);

// Vector sequence operations.
inline int           size(const vec3d& a);
inline const double* begin(const vec3d& a);
inline const double* end(const vec3d& a);
inline double*       begin(vec3d& a);
inline double*       end(vec3d& a);
inline const double* data(const vec3d& a);
inline double*       data(vec3d& a);

// Vector comparison operations.
inline bool operator==(const vec3d& a, const vec3d& b);
inline bool operator!=(const vec3d& a, const vec3d& b);

// Vector operations.
inline vec3d operator+(const vec3d& a);
inline vec3d operator-(const vec3d& a);
inline vec3d operator+(const vec3d& a, const vec3d& b);
inline vec3d operator+(const vec3d& a, double b);
inline vec3d operator+(double a, const vec3d& b);
inline vec3d operator-(const vec3d& a, const vec3d& b);
inline vec3d operator-(const vec3d& a, double b);
inline vec3d operator-(double a, const vec3d& b);
inline vec3d operator*(const vec3d& a, const vec3d& b);
inline vec3d operator*(const vec3d& a, double b);
inline vec3d operator*(double a, const vec3d& b);
inline vec3d operator/(const vec3d& a, const vec3d& b);
inline vec3d operator/(const vec3d& a, double b);
inline vec3d operator/(double a, const vec3d& b);

// Vector assignments
inline vec3d& operator+=(vec3d& a, const vec3d& b);
inline vec3d& operator+=(vec3d& a, double b);
inline vec3d& operator-=(vec3d& a, const vec3d& b);
inline vec3d& operator-=(vec3d& a, double b);
inline vec3d& operator*=(vec3d& a, const vec3d& b);
inline vec3d& operator*=(vec3d& a, double b);
inline vec3d& operator/=(vec3d& a, const vec3d& b);
inline vec3d& operator/=(vec3d& a, double b);

// Vector products and lengths.
inline double dot(const vec3d& a, const vec3d& b);
inline vec3d  cross(const vec3d& a, const vec3d& b);

inline double length(const vec3d& a);
inline double length_squared(const vec3d& a);
inline vec3d  normalize(const vec3d& a);
inline double distance(const vec3d& a, const vec3d& b);
inline double distance_squared(const vec3d& a, const vec3d& b);
inline double angle(const vec3d& a, const vec3d& b);

// Orthogonal vectors.
inline vec3d orthogonal(const vec3d& v);
inline vec3d orthonormalize(const vec3d& a, const vec3d& b);

// Reflected and refracted vector.
inline vec3d reflect(const vec3d& w, const vec3d& n);
inline vec3d refract(const vec3d& w, const vec3d& n, double inv_eta);

// Max element and clamp.
inline vec3d max(const vec3d& a, double b);
inline vec3d min(const vec3d& a, double b);
inline vec3d max(const vec3d& a, const vec3d& b);
inline vec3d min(const vec3d& a, const vec3d& b);
inline vec3d clamp(const vec3d& x, double min, double max);
inline vec3d lerp(const vec3d& a, const vec3d& b, double u);
inline vec3d lerp(const vec3d& a, const vec3d& b, const vec3d& u);

inline double max(const vec3d& a);
inline double min(const vec3d& a);
inline double sum(const vec3d& a);
inline double mean(const vec3d& a);

// Functions applied to vector elements
inline vec3d abs(const vec3d& a);
inline vec3d sqr(const vec3d& a);
inline vec3d sqrt(const vec3d& a);
inline vec3d exp(const vec3d& a);
inline vec3d log(const vec3d& a);
inline vec3d exp2(const vec3d& a);
inline vec3d log2(const vec3d& a);
inline vec3d pow(const vec3d& a, double b);
inline vec3d pow(const vec3d& a, const vec3d& b);
inline vec3d gain(const vec3d& a, double b);
inline bool  isfinite(const vec3d& a);
inline void  swap(vec3d& a, vec3d& b);

// Vector sequence operations.
inline int           size(const vec4d& a);
inline const double* begin(const vec4d& a);
inline const double* end(const vec4d& a);
inline double*       begin(vec4d& a);
inline double*       end(vec4d& a);
inline const double* data(const vec4d& a);
inline double*       data(vec4d& a);

// Vector comparison operations.
inline bool operator==(const vec4d& a, const vec4d& b);
inline bool operator!=(const vec4d& a, const vec4d& b);

// Vector operations.
inline vec4d operator+(const vec4d& a);
inline vec4d operator-(const vec4d& a);
inline vec4d operator+(const vec4d& a, const vec4d& b);
inline vec4d operator+(const vec4d& a, double b);
inline vec4d operator+(double a, const vec4d& b);
inline vec4d operator-(const vec4d& a, const vec4d& b);
inline vec4d operator-(const vec4d& a, double b);
inline vec4d operator-(double a, const vec4d& b);
inline vec4d operator*(const vec4d& a, const vec4d& b);
inline vec4d operator*(const vec4d& a, double b);
inline vec4d operator*(double a, const vec4d& b);
inline vec4d operator/(const vec4d& a, const vec4d& b);
inline vec4d operator/(const vec4d& a, double b);
inline vec4d operator/(double a, const vec4d& b);

// Vector assignments
inline vec4d& operator+=(vec4d& a, const vec4d& b);
inline vec4d& operator+=(vec4d& a, double b);
inline vec4d& operator-=(vec4d& a, const vec4d& b);
inline vec4d& operator-=(vec4d& a, double b);
inline vec4d& operator*=(vec4d& a, const vec4d& b);
inline vec4d& operator*=(vec4d& a, double b);
inline vec4d& operator/=(vec4d& a, const vec4d& b);
inline vec4d& operator/=(vec4d& a, double b);

// Vector products and lengths.
inline double dot(const vec4d& a, const vec4d& b);
inline double length(const vec4d& a);
inline double length_squared(const vec4d& a);
inline vec4d  normalize(const vec4d& a);
inline double distance(const vec4d& a, const vec4d& b);
inline double distance_squared(const vec4d& a, const vec4d& b);
inline double angle(const vec4d& a, const vec4d& b);

inline vec4d slerp(const vec4d& a, const vec4d& b, double u);

// Max element and clamp.
inline vec4d max(const vec4d& a, double b);
inline vec4d min(const vec4d& a, double b);
inline vec4d max(const vec4d& a, const vec4d& b);
inline vec4d min(const vec4d& a, const vec4d& b);
inline vec4d clamp(const vec4d& x, double min, double max);
inline vec4d lerp(const vec4d& a, const vec4d& b, double u);
inline vec4d lerp(const vec4d& a, const vec4d& b, const vec4d& u);

inline double max(const vec4d& a);
inline double min(const vec4d& a);
inline double sum(const vec4d& a);
inline double mean(const vec4d& a);

// Functions applied to vector elements
inline vec4d abs(const vec4d& a);
inline vec4d sqr(const vec4d& a);
inline vec4d sqrt(const vec4d& a);
inline vec4d exp(const vec4d& a);
inline vec4d log(const vec4d& a);
inline vec4d exp2(const vec4d& a);
inline vec4d log2(const vec4d& a);
inline vec4d pow(const vec4d& a, double b);
inline vec4d pow(const vec4d& a, const vec4d& b);
inline vec4d gain(const vec4d& a, double b);
inline bool  isfinite(const vec4d& a);
inline void  swap(vec4d& a, vec4d& b);

// Quaternion operatons represented as xi + yj + zk + w
// const auto identity_quat4f = vec4d{0, 0, 0, 1};
inline vec4d quat_mul(const vec4d& a, double b);
inline vec4d quat_mul(const vec4d& a, const vec4d& b);
inline vec4d quat_conjugate(const vec4d& a);
inline vec4d quat_inverse(const vec4d& a);

}  // namespace yocto

// -----------------------------------------------------------------------------
// MATH CONSTANTS AND FUNCTIONS IN DOUBLE
// -----------------------------------------------------------------------------
namespace yocto {

inline double abs(double a) { return a < 0 ? -a : a; }
inline double min(double a, double b) { return (a < b) ? a : b; }
inline double max(double a, double b) { return (a > b) ? a : b; }
inline double clamp(double a, double min_, double max_) {
  return min(max(a, min_), max_);
}
inline double sign(double a) { return a < 0 ? -1 : 1; }
inline double sqr(double a) { return a * a; }
inline double sqrt(double a) { return std::sqrt(a); }
inline double sin(double a) { return std::sin(a); }
inline double cos(double a) { return std::cos(a); }
inline double tan(double a) { return std::tan(a); }
inline double asin(double a) { return std::asin(a); }
inline double acos(double a) { return std::acos(a); }
inline double atan(double a) { return std::atan(a); }
inline double log(double a) { return std::log(a); }
inline double exp(double a) { return std::exp(a); }
inline double log2(double a) { return std::log2(a); }
inline double exp2(double a) { return std::exp2(a); }
inline double pow(double a, double b) { return std::pow(a, b); }
inline bool   isfinite(double a) { return std::isfinite(a); }
inline double atan2(double a, double b) { return std::atan2(a, b); }
inline double fmod(double a, double b) { return std::fmod(a, b); }
inline void   swap(double& a, double& b) { std::swap(a, b); }
inline double radians(double a) { return a * pif / 180; }
inline double degrees(double a) { return a * 180 / pif; }
inline double lerp(double a, double b, double u) { return a * (1 - u) + b * u; }
inline double step(double a, double u) { return u < a ? 0 : 1; }
inline double smoothstep(double a, double b, double u) {
  auto t = clamp((u - a) / (b - a), 0.0, 1.0);
  return t * t * (3 - 2 * t);
}
inline double bias(double a, double bias) {
  return a / ((1 / bias - 2) * (1 - a) + 1);
}
inline double gain(double a, double gain) {
  return (a < 0.5) ? bias(a * 2, gain) / 2
                   : bias(a * 2 - 1, 1 - gain) / 2 + 0.5;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// DOUBLE VECTORS
// -----------------------------------------------------------------------------
namespace yocto {

// Vec2
inline double& vec2d::operator[](int i) { return (&x)[i]; }
inline const double& vec2d::operator[](int i) const { return (&x)[i]; }

// Vec3
inline double& vec3d::operator[](int i) { return (&x)[i]; }
inline const double& vec3d::operator[](int i) const { return (&x)[i]; }

// Vec4
inline double& vec4d::operator[](int i) { return (&x)[i]; }
inline const double& vec4d::operator[](int i) const { return (&x)[i]; }

// Element access
inline vec3d xyz(const vec4d& a) { return {a.x, a.y, a.z}; }

// Vector sequence operations.
inline int           size(const vec2d& a) { return 2; }
inline const double* begin(const vec2d& a) { return &a.x; }
inline const double* end(const vec2d& a) { return &a.x + 2; }
inline double*       begin(vec2d& a) { return &a.x; }
inline double*       end(vec2d& a) { return &a.x + 2; }
inline const double* data(const vec2d& a) { return &a.x; }
inline double*       data(vec2d& a) { return &a.x; }

// Vector comparison operations.
inline bool operator==(const vec2d& a, const vec2d& b) {
  return a.x == b.x && a.y == b.y;
}
inline bool operator!=(const vec2d& a, const vec2d& b) {
  return a.x != b.x || a.y != b.y;
}

// Vector operations.
inline vec2d operator+(const vec2d& a) { return a; }
inline vec2d operator-(const vec2d& a) { return {-a.x, -a.y}; }
inline vec2d operator+(const vec2d& a, const vec2d& b) {
  return {a.x + b.x, a.y + b.y};
}
inline vec2d operator+(const vec2d& a, double b) { return {a.x + b, a.y + b}; }
inline vec2d operator+(double a, const vec2d& b) { return {a + b.x, a + b.y}; }
inline vec2d operator-(const vec2d& a, const vec2d& b) {
  return {a.x - b.x, a.y - b.y};
}
inline vec2d operator-(const vec2d& a, double b) { return {a.x - b, a.y - b}; }
inline vec2d operator-(double a, const vec2d& b) { return {a - b.x, a - b.y}; }
inline vec2d operator*(const vec2d& a, const vec2d& b) {
  return {a.x * b.x, a.y * b.y};
}
inline vec2d operator*(const vec2d& a, double b) { return {a.x * b, a.y * b}; }
inline vec2d operator*(double a, const vec2d& b) { return {a * b.x, a * b.y}; }
inline vec2d operator/(const vec2d& a, const vec2d& b) {
  return {a.x / b.x, a.y / b.y};
}
inline vec2d operator/(const vec2d& a, double b) { return {a.x / b, a.y / b}; }
inline vec2d operator/(double a, const vec2d& b) { return {a / b.x, a / b.y}; }

// Vector assignments
inline vec2d& operator+=(vec2d& a, const vec2d& b) { return a = a + b; }
inline vec2d& operator+=(vec2d& a, double b) { return a = a + b; }
inline vec2d& operator-=(vec2d& a, const vec2d& b) { return a = a - b; }
inline vec2d& operator-=(vec2d& a, double b) { return a = a - b; }
inline vec2d& operator*=(vec2d& a, const vec2d& b) { return a = a * b; }
inline vec2d& operator*=(vec2d& a, double b) { return a = a * b; }
inline vec2d& operator/=(vec2d& a, const vec2d& b) { return a = a / b; }
inline vec2d& operator/=(vec2d& a, double b) { return a = a / b; }

// Vector products and lengths.
inline double dot(const vec2d& a, const vec2d& b) {
  return a.x * b.x + a.y * b.y;
}
inline double cross(const vec2d& a, const vec2d& b) {
  return a.x * b.y - a.y * b.x;
}

inline double length(const vec2d& a) { return sqrt(dot(a, a)); }
inline double length_squared(const vec2d& a) { return dot(a, a); }
inline vec2d  normalize(const vec2d& a) {
  auto l = length(a);
  return (l != 0) ? a / l : a;
}
inline double distance(const vec2d& a, const vec2d& b) { return length(a - b); }
inline double distance_squared(const vec2d& a, const vec2d& b) {
  return dot(a - b, a - b);
}
inline double angle(const vec2d& a, const vec2d& b) {
  return acos(clamp(dot(normalize(a), normalize(b)), (double)-1, (double)1));
}

// Max element and clamp.
inline vec2d max(const vec2d& a, double b) {
  return {max(a.x, b), max(a.y, b)};
}
inline vec2d min(const vec2d& a, double b) {
  return {min(a.x, b), min(a.y, b)};
}
inline vec2d max(const vec2d& a, const vec2d& b) {
  return {max(a.x, b.x), max(a.y, b.y)};
}
inline vec2d min(const vec2d& a, const vec2d& b) {
  return {min(a.x, b.x), min(a.y, b.y)};
}
inline vec2d clamp(const vec2d& x, double min, double max) {
  return {clamp(x.x, min, max), clamp(x.y, min, max)};
}
inline vec2d lerp(const vec2d& a, const vec2d& b, double u) {
  return a * (1 - u) + b * u;
}
inline vec2d lerp(const vec2d& a, const vec2d& b, const vec2d& u) {
  return a * (1 - u) + b * u;
}

inline double max(const vec2d& a) { return max(a.x, a.y); }
inline double min(const vec2d& a) { return min(a.x, a.y); }
inline double sum(const vec2d& a) { return a.x + a.y; }
inline double mean(const vec2d& a) { return sum(a) / 2; }

// Functions applied to vector elements
inline vec2d abs(const vec2d& a) { return {abs(a.x), abs(a.y)}; }
inline vec2d sqr(const vec2d& a) { return {sqr(a.x), sqr(a.y)}; }
inline vec2d sqrt(const vec2d& a) { return {sqrt(a.x), sqrt(a.y)}; }
inline vec2d exp(const vec2d& a) { return {exp(a.x), exp(a.y)}; }
inline vec2d log(const vec2d& a) { return {log(a.x), log(a.y)}; }
inline vec2d exp2(const vec2d& a) { return {exp2(a.x), exp2(a.y)}; }
inline vec2d log2(const vec2d& a) { return {log2(a.x), log2(a.y)}; }
inline bool  isfinite(const vec2d& a) { return isfinite(a.x) && isfinite(a.y); }
inline vec2d pow(const vec2d& a, double b) {
  return {pow(a.x, b), pow(a.y, b)};
}
inline vec2d pow(const vec2d& a, const vec2d& b) {
  return {pow(a.x, b.x), pow(a.y, b.y)};
}
inline vec2d gain(const vec2d& a, double b) {
  return {gain(a.x, b), gain(a.y, b)};
}
inline void swap(vec2d& a, vec2d& b) { std::swap(a, b); }

// Vector sequence operations.
inline int           size(const vec3d& a) { return 3; }
inline const double* begin(const vec3d& a) { return &a.x; }
inline const double* end(const vec3d& a) { return &a.x + 3; }
inline double*       begin(vec3d& a) { return &a.x; }
inline double*       end(vec3d& a) { return &a.x + 3; }
inline const double* data(const vec3d& a) { return &a.x; }
inline double*       data(vec3d& a) { return &a.x; }

// Vector comparison operations.
inline bool operator==(const vec3d& a, const vec3d& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
inline bool operator!=(const vec3d& a, const vec3d& b) {
  return a.x != b.x || a.y != b.y || a.z != b.z;
}

// Vector operations.
inline vec3d operator+(const vec3d& a) { return a; }
inline vec3d operator-(const vec3d& a) { return {-a.x, -a.y, -a.z}; }
inline vec3d operator+(const vec3d& a, const vec3d& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline vec3d operator+(const vec3d& a, double b) {
  return {a.x + b, a.y + b, a.z + b};
}
inline vec3d operator+(double a, const vec3d& b) {
  return {a + b.x, a + b.y, a + b.z};
}
inline vec3d operator-(const vec3d& a, const vec3d& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline vec3d operator-(const vec3d& a, double b) {
  return {a.x - b, a.y - b, a.z - b};
}
inline vec3d operator-(double a, const vec3d& b) {
  return {a - b.x, a - b.y, a - b.z};
}
inline vec3d operator*(const vec3d& a, const vec3d& b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}
inline vec3d operator*(const vec3d& a, double b) {
  return {a.x * b, a.y * b, a.z * b};
}
inline vec3d operator*(double a, const vec3d& b) {
  return {a * b.x, a * b.y, a * b.z};
}
inline vec3d operator/(const vec3d& a, const vec3d& b) {
  return {a.x / b.x, a.y / b.y, a.z / b.z};
}
inline vec3d operator/(const vec3d& a, double b) {
  return {a.x / b, a.y / b, a.z / b};
}
inline vec3d operator/(double a, const vec3d& b) {
  return {a / b.x, a / b.y, a / b.z};
}

// Vector assignments
inline vec3d& operator+=(vec3d& a, const vec3d& b) { return a = a + b; }
inline vec3d& operator+=(vec3d& a, double b) { return a = a + b; }
inline vec3d& operator-=(vec3d& a, const vec3d& b) { return a = a - b; }
inline vec3d& operator-=(vec3d& a, double b) { return a = a - b; }
inline vec3d& operator*=(vec3d& a, const vec3d& b) { return a = a * b; }
inline vec3d& operator*=(vec3d& a, double b) { return a = a * b; }
inline vec3d& operator/=(vec3d& a, const vec3d& b) { return a = a / b; }
inline vec3d& operator/=(vec3d& a, double b) { return a = a / b; }

// Vector products and lengths.
inline double dot(const vec3d& a, const vec3d& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline vec3d cross(const vec3d& a, const vec3d& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline double length(const vec3d& a) { return sqrt(dot(a, a)); }
inline double length_squared(const vec3d& a) { return dot(a, a); }
inline vec3d  normalize(const vec3d& a) {
  auto l = length(a);
  return (l != 0) ? a / l : a;
}
inline double distance(const vec3d& a, const vec3d& b) { return length(a - b); }
inline double distance_squared(const vec3d& a, const vec3d& b) {
  return dot(a - b, a - b);
}
inline double angle(const vec3d& a, const vec3d& b) {
  return acos(clamp(dot(normalize(a), normalize(b)), (double)-1, (double)1));
}

// Orthogonal vectors.
inline vec3d orthogonal(const vec3d& v) {
  // http://lolengine.net/blog/2013/09/21/picking-orthogonal-vector-combing-coconuts)
  return abs(v.x) > abs(v.z) ? vec3d{-v.y, v.x, 0} : vec3d{0, -v.z, v.y};
}
inline vec3d orthonormalize(const vec3d& a, const vec3d& b) {
  return normalize(a - b * dot(a, b));
}

// Reflected and refracted vector.
inline vec3d reflect(const vec3d& w, const vec3d& n) {
  return -w + 2 * dot(n, w) * n;
}
inline vec3d refract(const vec3d& w, const vec3d& n, double inv_eta) {
  auto cosine = dot(n, w);
  auto k      = 1 + inv_eta * inv_eta * (cosine * cosine - 1);
  if (k < 0) return {0, 0, 0};  // tir
  return -w * inv_eta + (inv_eta * cosine - sqrt(k)) * n;
}

// Max element and clamp.
inline vec3d max(const vec3d& a, double b) {
  return {max(a.x, b), max(a.y, b), max(a.z, b)};
}
inline vec3d min(const vec3d& a, double b) {
  return {min(a.x, b), min(a.y, b), min(a.z, b)};
}
inline vec3d max(const vec3d& a, const vec3d& b) {
  return {max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)};
}
inline vec3d min(const vec3d& a, const vec3d& b) {
  return {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
}
inline vec3d clamp(const vec3d& x, double min, double max) {
  return {clamp(x.x, min, max), clamp(x.y, min, max), clamp(x.z, min, max)};
}
inline vec3d lerp(const vec3d& a, const vec3d& b, double u) {
  return a * (1 - u) + b * u;
}
inline vec3d lerp(const vec3d& a, const vec3d& b, const vec3d& u) {
  return a * (1 - u) + b * u;
}

inline double max(const vec3d& a) { return max(max(a.x, a.y), a.z); }
inline double min(const vec3d& a) { return min(min(a.x, a.y), a.z); }
inline double sum(const vec3d& a) { return a.x + a.y + a.z; }
inline double mean(const vec3d& a) { return sum(a) / 3; }

// Functions applied to vector elements
inline vec3d abs(const vec3d& a) { return {abs(a.x), abs(a.y), abs(a.z)}; }
inline vec3d sqr(const vec3d& a) { return {sqr(a.x), sqr(a.y), sqr(a.z)}; }
inline vec3d sqrt(const vec3d& a) { return {sqrt(a.x), sqrt(a.y), sqrt(a.z)}; }
inline vec3d exp(const vec3d& a) { return {exp(a.x), exp(a.y), exp(a.z)}; }
inline vec3d log(const vec3d& a) { return {log(a.x), log(a.y), log(a.z)}; }
inline vec3d exp2(const vec3d& a) { return {exp2(a.x), exp2(a.y), exp2(a.z)}; }
inline vec3d log2(const vec3d& a) { return {log2(a.x), log2(a.y), log2(a.z)}; }
inline vec3d pow(const vec3d& a, double b) {
  return {pow(a.x, b), pow(a.y, b), pow(a.z, b)};
}
inline vec3d pow(const vec3d& a, const vec3d& b) {
  return {pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z)};
}
inline vec3d gain(const vec3d& a, double b) {
  return {gain(a.x, b), gain(a.y, b), gain(a.z, b)};
}
inline bool isfinite(const vec3d& a) {
  return isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
}
inline void swap(vec3d& a, vec3d& b) { std::swap(a, b); }

// Vector sequence operations.
inline int           size(const vec4d& a) { return 4; }
inline const double* begin(const vec4d& a) { return &a.x; }
inline const double* end(const vec4d& a) { return &a.x + 4; }
inline double*       begin(vec4d& a) { return &a.x; }
inline double*       end(vec4d& a) { return &a.x + 4; }
inline const double* data(const vec4d& a) { return &a.x; }
inline double*       data(vec4d& a) { return &a.x; }

// Vector comparison operations.
inline bool operator==(const vec4d& a, const vec4d& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
inline bool operator!=(const vec4d& a, const vec4d& b) {
  return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}

// Vector operations.
inline vec4d operator+(const vec4d& a) { return a; }
inline vec4d operator-(const vec4d& a) { return {-a.x, -a.y, -a.z, -a.w}; }
inline vec4d operator+(const vec4d& a, const vec4d& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
inline vec4d operator+(const vec4d& a, double b) {
  return {a.x + b, a.y + b, a.z + b, a.w + b};
}
inline vec4d operator+(double a, const vec4d& b) {
  return {a + b.x, a + b.y, a + b.z, a + b.w};
}
inline vec4d operator-(const vec4d& a, const vec4d& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
inline vec4d operator-(const vec4d& a, double b) {
  return {a.x - b, a.y - b, a.z - b, a.w - b};
}
inline vec4d operator-(double a, const vec4d& b) {
  return {a - b.x, a - b.y, a - b.z, a - b.w};
}
inline vec4d operator*(const vec4d& a, const vec4d& b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}
inline vec4d operator*(const vec4d& a, double b) {
  return {a.x * b, a.y * b, a.z * b, a.w * b};
}
inline vec4d operator*(double a, const vec4d& b) {
  return {a * b.x, a * b.y, a * b.z, a * b.w};
}
inline vec4d operator/(const vec4d& a, const vec4d& b) {
  return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}
inline vec4d operator/(const vec4d& a, double b) {
  return {a.x / b, a.y / b, a.z / b, a.w / b};
}
inline vec4d operator/(double a, const vec4d& b) {
  return {a / b.x, a / b.y, a / b.z, a / b.w};
}

// Vector assignments
inline vec4d& operator+=(vec4d& a, const vec4d& b) { return a = a + b; }
inline vec4d& operator+=(vec4d& a, double b) { return a = a + b; }
inline vec4d& operator-=(vec4d& a, const vec4d& b) { return a = a - b; }
inline vec4d& operator-=(vec4d& a, double b) { return a = a - b; }
inline vec4d& operator*=(vec4d& a, const vec4d& b) { return a = a * b; }
inline vec4d& operator*=(vec4d& a, double b) { return a = a * b; }
inline vec4d& operator/=(vec4d& a, const vec4d& b) { return a = a / b; }
inline vec4d& operator/=(vec4d& a, double b) { return a = a / b; }

// Vector products and lengths.
inline double dot(const vec4d& a, const vec4d& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline double length(const vec4d& a) { return sqrt(dot(a, a)); }
inline double length_squared(const vec4d& a) { return dot(a, a); }
inline vec4d  normalize(const vec4d& a) {
  auto l = length(a);
  return (l != 0) ? a / l : a;
}
inline double distance(const vec4d& a, const vec4d& b) { return length(a - b); }
inline double distance_squared(const vec4d& a, const vec4d& b) {
  return dot(a - b, a - b);
}
inline double angle(const vec4d& a, const vec4d& b) {
  return acos(clamp(dot(normalize(a), normalize(b)), (double)-1, (double)1));
}

inline vec4d slerp(const vec4d& a, const vec4d& b, double u) {
  // https://en.wikipedia.org/wiki/Slerp
  auto an = normalize(a), bn = normalize(b);
  auto d = dot(an, bn);
  if (d < 0) {
    bn = -bn;
    d  = -d;
  }
  if (d > (double)0.9995) return normalize(an + u * (bn - an));
  auto th = acos(clamp(d, (double)-1, (double)1));
  if (th == 0) return an;
  return an * (sin(th * (1 - u)) / sin(th)) + bn * (sin(th * u) / sin(th));
}

// Max element and clamp.
inline vec4d max(const vec4d& a, double b) {
  return {max(a.x, b), max(a.y, b), max(a.z, b), max(a.w, b)};
}
inline vec4d min(const vec4d& a, double b) {
  return {min(a.x, b), min(a.y, b), min(a.z, b), min(a.w, b)};
}
inline vec4d max(const vec4d& a, const vec4d& b) {
  return {max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)};
}
inline vec4d min(const vec4d& a, const vec4d& b) {
  return {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w)};
}
inline vec4d clamp(const vec4d& x, double min, double max) {
  return {clamp(x.x, min, max), clamp(x.y, min, max), clamp(x.z, min, max),
      clamp(x.w, min, max)};
}
inline vec4d lerp(const vec4d& a, const vec4d& b, double u) {
  return a * (1 - u) + b * u;
}
inline vec4d lerp(const vec4d& a, const vec4d& b, const vec4d& u) {
  return a * (1 - u) + b * u;
}

inline double max(const vec4d& a) { return max(max(max(a.x, a.y), a.z), a.w); }
inline double min(const vec4d& a) { return min(min(min(a.x, a.y), a.z), a.w); }
inline double sum(const vec4d& a) { return a.x + a.y + a.z + a.w; }
inline double mean(const vec4d& a) { return sum(a) / 4; }

// Functions applied to vector elements
inline vec4d abs(const vec4d& a) {
  return {abs(a.x), abs(a.y), abs(a.z), abs(a.w)};
}
inline vec4d sqr(const vec4d& a) {
  return {sqr(a.x), sqr(a.y), sqr(a.z), sqr(a.w)};
}
inline vec4d sqrt(const vec4d& a) {
  return {sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w)};
}
inline vec4d exp(const vec4d& a) {
  return {exp(a.x), exp(a.y), exp(a.z), exp(a.w)};
}
inline vec4d log(const vec4d& a) {
  return {log(a.x), log(a.y), log(a.z), log(a.w)};
}
inline vec4d exp2(const vec4d& a) {
  return {exp2(a.x), exp2(a.y), exp2(a.z), exp2(a.w)};
}
inline vec4d log2(const vec4d& a) {
  return {log2(a.x), log2(a.y), log2(a.z), log2(a.w)};
}
inline vec4d pow(const vec4d& a, double b) {
  return {pow(a.x, b), pow(a.y, b), pow(a.z, b), pow(a.w, b)};
}
inline vec4d pow(const vec4d& a, const vec4d& b) {
  return {pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z), pow(a.w, b.w)};
}
inline vec4d gain(const vec4d& a, double b) {
  return {gain(a.x, b), gain(a.y, b), gain(a.z, b), gain(a.w, b)};
}
inline bool isfinite(const vec4d& a) {
  return isfinite(a.x) && isfinite(a.y) && isfinite(a.z) && isfinite(a.w);
}
inline void swap(vec4d& a, vec4d& b) { std::swap(a, b); }

// Quaternion operatons represented as xi + yj + zk + w
// const auto identity_quat4f = vec4d{0, 0, 0, 1};
inline vec4d quat_mul(const vec4d& a, double b) {
  return {a.x * b, a.y * b, a.z * b, a.w * b};
}
inline vec4d quat_mul(const vec4d& a, const vec4d& b) {
  return {a.x * b.w + a.w * b.x + a.y * b.w - a.z * b.y,
      a.y * b.w + a.w * b.y + a.z * b.x - a.x * b.z,
      a.z * b.w + a.w * b.z + a.x * b.y - a.y * b.x,
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
inline vec4d quat_conjugate(const vec4d& a) { return {-a.x, -a.y, -a.z, a.w}; }
inline vec4d quat_inverse(const vec4d& a) {
  return quat_conjugate(a) / dot(a, a);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// VECTOR HASHING
// -----------------------------------------------------------------------------
namespace std {

// Hash functor for vector for use with hash_map
template <>
struct hash<yocto::vec2i> {
  size_t operator()(const yocto::vec2i& v) const {
    static const auto hasher = std::hash<int>();
    auto              h      = (size_t)0;
    h ^= hasher(v.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= hasher(v.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

}  // namespace std

// -----------------------------------------------------------------------------
// UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

#define report_floating_point(x) \
  if (!isfinite(x))              \
    printf("%s, line %d: nan/infinity detected\n", __FILE__, __LINE__);

template <typename T>
inline int find_in_vector(const T& vec, int x) {
  for (auto i = 0; i < size(vec); i++)
    if (vec[i] == x) return i;
  return -1;
}

inline int mod3(int i) { return (i > 2) ? i - 3 : i; }

[[maybe_unused]] static vec2d to_double(const vec2f& v) {
  return vec2d{v.x, v.y};
}
[[maybe_unused]] static vec3d to_double(const vec3f& v) {
  return vec3d{v.x, v.y, v.z};
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// UTILITIES
// -----------------------------------------------------------------------------
namespace yocto {

vec3f eval_position(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const mesh_point& sample) {
  auto [x, y, z] = triangles[sample.face];
  return interpolate_triangle(
      positions[x], positions[y], positions[z], sample.uv);
}

vec3f eval_normal(const vector<vec3i>& triangles, const vector<vec3f>& normals,
    const mesh_point& point) {
  auto [x, y, z] = triangles[point.face];
  return normalize(
      interpolate_triangle(normals[x], normals[y], normals[z], point.uv));
}

static int find_adjacent_triangle(
    const vec3i& triangle, const vec3i& adjacent) {
  for (auto i : range(3)) {
    auto k = find_in_vector(adjacent, triangle[i]);
    if (k != -1) {
      if (find_in_vector(adjacent, triangle[mod3(i + 1)]) != -1) {
        return i;
      } else {
        return mod3(i + 2);
      }
    }
  }
  assert(0 && "input triangles are not adjacent");
  return -1;
}

static int find_adjacent_triangle(
    const vector<vec3i>& triangles, int face, int neighbor) {
  return find_adjacent_triangle(triangles[face], triangles[neighbor]);
}

static vec2f unfold_point(const vec3f& pa, const vec3f& pb,
    const vec3f& pv, const vec2f& ca, const vec2f& cb) {
  // Unfold position of vertex v
  auto ex = normalize(ca - cb);
  auto ey = vec2f{-ex.y, ex.x};

  auto result = vec2f{};

  // Ortogonal projection
  auto pv_pb = pv - pb;
  auto x     = dot(pa - pb, pv_pb) / length(pa - pb);
  result     = x * ex;

  // Pythagorean theorem
  auto y = dot(pv_pb, pv_pb) - x * x;
  if (y > 0) {
    y = sqrt(y);
    result += y * ey;
  } else {
    assert(0);
  }

  return result;
}

using unfold_triangle  = std::array<vec2f, 3>;

// given the 2D coordinates in tanget space of a triangle, find the coordinates
// of the k-th neighbor triangle
unfold_triangle unfold_face(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const unfold_triangle& tr, int face,
    int neighbor) {
  auto k = find_adjacent_triangle(triangles, face, neighbor);
  auto j = find_adjacent_triangle(triangles, neighbor, face);
  assert(j != -1);
  assert(k != -1);
  auto v = triangles[neighbor][mod3(j + 2)];
  auto a = triangles[face][k];
  auto b = triangles[face][mod3(k + 1)];

  auto result         = unfold_triangle{};
  result[j]           = tr[mod3(k + 1)];
  result[mod3(j + 1)] = tr[k];

  auto& pa    = positions[a];
  auto& pb    = positions[b];
  auto& pv    = positions[v];
  auto  point = unfold_point(pa, pb, pv, result[mod3(j + 1)], result[j]);
  result[mod3(j + 2)] = result[j] + point;

  assert(result[0] != result[1]);
  assert(result[1] != result[2]);
  assert(result[2] != result[0]);
  return result;
}

static unfold_triangle unfold_face(
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const unfold_triangle& tr, int face,
    int k) {
  return unfold_face(triangles, positions, tr, face, adjacencies[face][k]);
}

// assign 2D coordinates to vertices of the triangle containing the mesh
// point, putting the point at (0, 0)
unfold_triangle triangle_coordinates(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const mesh_point& point) {
  auto result = unfold_triangle{};
  auto tr     = triangles[point.face];
  result[0]   = {0, 0};
  result[1]   = {0, length(positions[tr.x] - positions[tr.y])};
  result[2] = result[0] + unfold_point(positions[tr.y], positions[tr.x],
                              positions[tr.z], result[1], result[0]);

  // Transform coordinates such that point = (0, 0)
  auto point_coords = interpolate_triangle(
      result[0], result[1], result[2], point.uv);
  result[0] -= point_coords;
  result[1] -= point_coords;
  result[2] -= point_coords;

  assert(result[0] != result[1]);
  assert(result[1] != result[2]);
  assert(result[2] != result[0]);
  return result;
}

// assign 2D coordinates to a strip of triangles. point start is at (0, 0)
vector<unfold_triangle> unfold_strip(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<int>& strip,
    const mesh_point& start) {
  auto coords = vector<unfold_triangle>(strip.size());
  assert(start.face == strip[0]);
  coords[0] = triangle_coordinates(triangles, positions, start);

  for (auto i = 1; i < (int)strip.size(); i++) {
    assert(coords[i - 1][0] != coords[i - 1][1]);
    assert(coords[i - 1][1] != coords[i - 1][2]);
    assert(coords[i - 1][2] != coords[i - 1][0]);
    coords[i] = unfold_face(
        triangles, positions, coords[i - 1], strip[i - 1], strip[i]);
  }

  return coords;
}

// Create sequence of 2D segments (portals) needed for funneling.
static vector<pair<vec2f, vec2f>> make_funnel_portals(
    const vector<vec3i>& triangles, const vector<unfold_triangle>& coords,
    const vector<int>& strip, const mesh_point& to) {
  auto portals = vector<pair<vec2f, vec2f>>(strip.size());
  for (auto i = 0; i < (int)strip.size() - 1; i++) {
    auto curr = strip[i], next = strip[i + 1];
    auto k     = find_adjacent_triangle(triangles, curr, next);
    auto tr    = coords[i];
    portals[i] = {tr[k], tr[mod3(k + 1)]};
  }
  auto end = interpolate_triangle(
      coords.back()[0], coords.back()[1], coords.back()[2], to.uv);
  portals.back() = {end, end};
  return portals;
}


static vector<pair<vec2f, vec2f>> unfold_funnel_portals(
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<int>& strip, const mesh_point& start, const mesh_point& end) {
  auto coords = unfold_strip(triangles, positions, strip, start);
  return make_funnel_portals(triangles, coords, strip, end);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// ADJACENCIES
// -----------------------------------------------------------------------------
namespace yocto {

// Triangle fan starting from a face and going towards the k-th neighbor face.
vector<int> triangle_fan(
    const vector<vec3i>& adjacencies, int face, int k, bool clockwise) {
  auto result = vector<int>{};
  result.push_back(face);
  auto prev   = face;
  auto node   = adjacencies[face][k];
  auto offset = 2 - (int)clockwise;
  // for(auto i : range(256)) {
  while (true) {
    if (node == -1) break;
    if (node == face) break;
    result.push_back(node);
    auto kk = find_in_vector(adjacencies[node], prev);
    assert(kk != -1);
    kk   = mod3(kk + offset);
    prev = node;
    node = adjacencies[node][kk];
  }
  return result;
}

// returns the list of triangles incident at each vertex in ccw order
vector<vector<int>> vertex_to_triangles(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies) {
  auto v2t    = vector<vector<int>>{positions.size(), vector<int>{}};
  auto offset = 0;
  for (auto i = 0; i < (int)triangles.size(); ++i) {
    for (auto j = 0; j < 3; ++j) {
      auto curr = triangles[i][j];
      if (v2t[curr].size() == 0) {
        offset    = find_in_vector(triangles[i], curr);
        v2t[curr] = triangle_fan(adjacencies, i, mod3(offset + 2));
      }
    }
  }
  return v2t;
}

// Face adjacent to t and opposite to vertex vid
int opposite_face(const vector<vec3i>& triangles,
    const vector<vec3i>& adjacencies, int t, int vid) {
  auto triangle = adjacencies[t];
  for (auto i = 0; i < 3; ++i) {
    if (find_in_vector(triangles[triangle[i]], vid) < 0) return triangle[i];
  }
  return -1;
}

// Finds the opposite vertex of an edge
int opposite_vertex(const vec3i& triangle, const vec2i& edge) {
  for (auto i = 0; i < 3; ++i) {
    if (triangle[i] != edge.x && triangle[i] != edge.y) return triangle[i];
  }
  return -1;
}

int opposite_vertex(const vector<vec3i>& triangles,
    const vector<vec3i>& adjacencies, int face, int k) {
  int neighbor = adjacencies[face][k];
  int j        = find_in_vector(adjacencies[neighbor], face);
  assert(j != -1);
  auto tt = triangles[neighbor];
  return tt[mod3(j + 2)];
}

// Finds common edge between triangles
vec2i common_edge(const vec3i& triangle0, const vec3i& triangle1) {
  for (auto i : range(3)) {
    for (auto k : range(3)) {
      if (triangle0[i] == triangle1[k] &&
          triangle0[mod3(i + 1)] == triangle1[mod3(k + 2)])
        return {triangle0[i], triangle0[mod3(i + 1)]};
    }
  }
  return {-1, -1};
}

vec2i opposite_edge(const vec3i& t, int vid) {
  auto offset = find_in_vector(t, vid);
  auto v0     = t[mod3(offset + 1)];
  auto v1     = t[mod3(offset + 2)];
  return vec2i{v0, v1};
}

// TODO: cleanup
vec2i common_edge(const vector<vec3i>& triangles, int pid0, int pid1) {
  auto& poly0 = triangles[pid0];
  auto& poly1 = triangles[pid1];
  for (auto i = 0; i < 3; ++i) {
    auto& vid    = poly0[i];
    auto  offset = find_in_vector(poly1, vid);
    if (offset < 0) {
      offset  = find_in_vector(poly0, vid);
      auto e0 = poly0[(offset + 1) % 3];
      auto e1 = poly0[(offset + 2) % 3];
      if (find_in_vector(poly1, e0) != -1 && find_in_vector(poly1, e1) != -1)
        return {e0, e1};
      else
        return {-1, -1};
    }
  }
  return {-1, -1};
}

// TODO: cleanup
int common_vertex(const vector<vec3i>& triangles, int pid0, int pid1) {
  auto& poly0 = triangles[pid0];
  auto& poly1 = triangles[pid1];
  for (auto i = 0; i < 3; ++i) {
    auto& vid    = poly0[i];
    auto  offset = find_in_vector(poly1, vid);
    if (offset != -1) return vid;
  }
  return -1;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// PROCEDURAL MODELING
// -----------------------------------------------------------------------------
namespace yocto {

// Extract isoline from surface scalar field.
void meandering_triangles(const vector<float>& field, float isoline,
    int selected_tag, int t0, int t1, vector<vec3i>& triangles,
    vector<int>& tags, vector<vec3f>& positions, vector<vec3f>& normals) {
  auto num_triangles = (int)triangles.size();

  // Edgemap to keep track of the added vertex on each splitted edge.
  // key: edge (ordered std::pair), value: vertex index
  auto emap = unordered_map<vec2i, int>();

  // Helper procedures.
  auto make_edge = [](int a, int b) -> vec2i {
    return a < b ? vec2i{a, b} : vec2i{b, a};
  };
  auto add_vertex = [&](int a, int b, float coeff) -> int {
    auto position = coeff * positions[a] + (1 - coeff) * positions[b];
    auto normal   = normalize(coeff * normals[a] + (1 - coeff) * normals[b]);
    auto index    = (int)positions.size();
    positions.push_back(position);
    normals.push_back(normal);
    return index;
  };
  auto get_tag = [&](int v) { return field[v] > isoline ? t1 : t0; };

  for (auto i = 0; i < num_triangles; ++i) {
    if (tags[i] != selected_tag) continue;

    auto tr = triangles[i];

    // Find which vertex has different tag, if any.
    auto j = -1;
    for (auto k : range(3)) {
      if (get_tag(tr[k]) != get_tag(tr[mod3(k + 1)]) &&
          get_tag(tr[k]) != get_tag(tr[mod3(k + 2)])) {
        j = k;
      }
    }

    // If all vertices have same tag, then tag the whole triangle and continue.
    if (j == -1) {
      tags[i] = get_tag(tr.x);
      continue;
    }

    // Reorder the triangle such that vertex with different tag is always z
    tr = {tr[mod3(j + 2)], tr[mod3(j + 1)], tr[j]};

    auto values = vec3f{field[tr.x], field[tr.y], field[tr.z]};
    values -= isoline;

    // Create or retrieve new vertices that split the edges
    auto new_verts = std::array<int, 2>{-1, -1};
    for (auto k : {0, 1}) {
      auto vert = tr[k];
      auto a    = values[k];
      auto b    = values.z;
      auto edge = make_edge(tr.z, vert);
      auto it   = emap.find(edge);

      if (it != emap.end()) {
        // Edge already processed.
        new_verts[k] = it->second;
      } else {
        // Compute new vertex via interpolation.
        auto alpha   = abs(a / (b - a));
        new_verts[k] = add_vertex(tr.z, vert, alpha);
        emap.insert(it, {edge, new_verts[k]});
      }
    }

    /*
                     tr.z
                       /\
                      /  \
                     /    \
                    /  i   \
                   /        \
    new_verts[1]  /..........\  new_verts[0]
                 /   . n_f[0] \
                /       .      \
               /           .    \
              / new_faces[1]  .  \
             /___________________.\
       tr.y                      tr.x

    */

    // Add two new faces.
    triangles.push_back({new_verts[0], new_verts[1], tr.x});
    tags.push_back(get_tag(tr.x));

    triangles.push_back({tr.y, tr.x, new_verts[1]});
    tags.push_back(get_tag(tr.y));

    // Edit old face.
    triangles[i] = vec3i{new_verts[0], tr.z, new_verts[1]};
    tags[i]      = get_tag(tr.z);
  }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// SHAPE GEODESICS
// -----------------------------------------------------------------------------
namespace yocto {

static void connect_nodes(geodesic_solver& solver, int a, int b, float length) {
  solver.graph[a].push_back({b, length});
  solver.graph[b].push_back({a, length});
}

static float opposite_nodes_arc_length(
    const vector<vec3f>& positions, int a, int c, const vec2i& edge) {
  // Triangles (a, b, d) and (b, d, c) are connected by (b, d) edge
  // Nodes a and c must be connected.

  auto b = edge.x, d = edge.y;
  auto ba = positions[a] - positions[b];
  auto bc = positions[c] - positions[b];
  auto bd = positions[d] - positions[b];

  auto cos_alpha = dot(normalize(ba), normalize(bd));
  auto cos_beta  = dot(normalize(bc), normalize(bd));
  auto sin_alpha = sqrt(max(0.0f, 1 - cos_alpha * cos_alpha));
  auto sin_beta  = sqrt(max(0.0f, 1 - cos_beta * cos_beta));

  // cos(alpha + beta)
  auto cos_alpha_beta = cos_alpha * cos_beta - sin_alpha * sin_beta;
  if (cos_alpha_beta <= -1) return flt_max;

  // law of cosines (generalized Pythagorean theorem)
  auto len = dot(ba, ba) + dot(bc, bc) -
             length(ba) * length(bc) * 2 * cos_alpha_beta;

  if (len <= 0)
    return flt_max;
  else
    return sqrt(len);
}

static void connect_opposite_nodes(geodesic_solver& solver,
    const vector<vec3f>& positions, const vec3i& tr0, const vec3i& tr1,
    const vec2i& edge) {
  auto opposite_vertex = [](const vec3i& tr, const vec2i& edge) -> int {
    for (auto i = 0; i < 3; ++i) {
      if (tr[i] != edge.x && tr[i] != edge.y) return tr[i];
    }
    return -1;
  };

  auto v0 = opposite_vertex(tr0, edge);
  auto v1 = opposite_vertex(tr1, edge);
  if (v0 == -1 || v1 == -1) return;
  auto length = opposite_nodes_arc_length(positions, v0, v1, edge);
  connect_nodes(solver, v0, v1, length);
}

geodesic_solver make_geodesic_solver(const vector<vec3i>& triangles,
    const vector<vec3i>& adjacencies, const vector<vec3f>& positions) {
  auto solver = geodesic_solver{};
  solver.graph.resize(positions.size());
  for (auto face = 0; face < (int)triangles.size(); face++) {
    for (auto k : range(3)) {
      auto a = triangles[face][k];
      auto b = triangles[face][mod3(k + 1)];

      // connect mesh edges
      auto len = length(positions[a] - positions[b]);
      if (a < b) connect_nodes(solver, a, b, len);

      // connect opposite nodes
      auto neighbor = adjacencies[face][k];
      if (face < neighbor) {
        connect_opposite_nodes(
            solver, positions, triangles[face], triangles[neighbor], {a, b});
      }
    }
  }
  return solver;
}

int find_small_edge(
    const vector<vec3i>& triangles, const vector<vec3f>& positions, int face) {
  int verts[3] = {triangles[face].x, triangles[face].y, triangles[face].z};

  vec3f pos[3] = {
      positions[verts[0]], positions[verts[1]], positions[verts[2]]};

  float lengths[3] = {length(pos[0] - pos[1]), length(pos[1] - pos[2]),
      length(pos[2] - pos[0])};

  int small = 0;
  if (lengths[0] < lengths[1] && lengths[0] < lengths[2]) small = 0;
  if (lengths[1] < lengths[0] && lengths[1] < lengths[2]) small = 1;
  if (lengths[2] < lengths[0] && lengths[2] < lengths[1]) small = 2;

  if (lengths[small] < 0.3 * lengths[(small + 1) % 3] &&
      lengths[small] < 0.3 * lengths[(small + 2) % 3]) {
    return small;
  }
  return -1;
}

// Construct a graph to compute geodesic distances
dual_geodesic_solver make_dual_geodesic_solver(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies) {
  auto solver = dual_geodesic_solver{};
  solver.centroids.resize(triangles.size());

  // Init centroids.
  for (int face = 0; face < (int)triangles.size(); face++) {
    vec3f pos[3] = {positions[triangles[face].x], positions[triangles[face].y],
        positions[triangles[face].z]};
    auto  l0     = length(pos[0] - pos[1]);
    auto  p0     = (pos[0] + pos[1]) / 2;
    auto  l1     = length(pos[1] - pos[2]);
    auto  p1     = (pos[1] + pos[2]) / 2;
    auto  l2     = length(pos[2] - pos[0]);
    auto  p2     = (pos[2] + pos[0]) / 2;
    solver.centroids[face] = (l0 * p0 + l1 * p1 + l2 * p2) / (l0 + l1 + l2);
  }

  solver.graph.resize(triangles.size());
  for (auto i = 0; i < (int)solver.graph.size(); ++i) {
    for (auto k = 0; k < 3; ++k) {
      solver.graph[i][k].node = adjacencies[i][k];

      if (adjacencies[i][k] == -1) {
        solver.graph[i][k].length = flt_max;
      } else {
        // TODO(splinesurf): check which solver is better/faster.
#if 1
        solver.graph[i][k].length = length(
            solver.centroids[i] - solver.centroids[adjacencies[i][k]]);
#else
        auto p0    = mesh_point{i, {1 / 3.f, 1 / 3.f}};
        auto p1    = mesh_point{adjacencies[i][k], {1 / 3.f, 1 / 3.f}};
        auto path  = geodesic_path{};
        path.strip = {i, adjacencies[i][k]};
        path.start = p0;
        path.end   = p1;
        straighten_path(path, triangles, positions, adjacencies);
        auto len = path_length(path, triangles, positions, adjacencies);
        solver.graph[i][k].length = len;
#endif
      }
    }
  }
  return solver;
}

// `update` is a function that is executed during expansion, every time a node
// is put into queue. `exit` is a function that tells whether to expand the
// current node or perform early exit.
template <typename Update, typename Stop, typename Exit>
void visit_geodesic_graph(vector<float>& field, const geodesic_solver& solver,
    const vector<int>& sources, Update&& update, Stop&& stop, Exit&& exit) {
  /*
     This algortithm uses the heuristic Small Label Fisrt and Large Label Last
     https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm

     Large Label Last (LLL): When extracting nodes from the queue, pick the
     front one. If it weights more than the average weight of the queue, put
     on the back and check the next node. Continue this way.
     Sometimes average_weight is less than every value due to floating point
     errors (doesn't happen with double precision).

     Small Label First (SLF): When adding a new node to queue, instead of
     always pushing it to the end of the queue, if it weights less than the
     front node of the queue, it is put on front. Otherwise the node is put at
     the end of the queue.
  */

  auto in_queue = vector<bool>(solver.graph.size(), false);

  // Cumulative weights of elements in queue. Used to keep track of the
  // average weight of the queue.
  auto cumulative_weight = 0.0;

  // setup queue
  auto queue = deque<int>();
  for (auto source : sources) {
    in_queue[source] = true;
    cumulative_weight += field[source];
    queue.push_back(source);
  }

  while (!queue.empty()) {
    auto node           = queue.front();
    auto average_weight = (float)(cumulative_weight / queue.size());

    // Large Label Last (see comment at the beginning)
    for (auto tries = 0; tries < (int)queue.size() + 1; tries++) {
      if (field[node] <= average_weight) break;
      queue.pop_front();
      queue.push_back(node);
      node = queue.front();
    }

    // Remove node from queue.
    queue.pop_front();
    in_queue[node] = false;
    cumulative_weight -= field[node];

    // Check early exit condition.
    if (exit(node)) break;
    if (stop(node)) continue;

    for (auto i = 0; i < (int)solver.graph[node].size(); i++) {
      // Distance of neighbor through this node
      auto new_distance = field[node] + solver.graph[node][i].length;
      auto neighbor     = solver.graph[node][i].node;

      auto old_distance = field[neighbor];
      if (new_distance >= old_distance) continue;

      if (in_queue[neighbor]) {
        // If neighbor already in queue, don't add it.
        // Just update cumulative weight.
        cumulative_weight += new_distance - old_distance;
      } else {
        // If neighbor not in queue, add node to queue using Small Label
        // First (see comment at the beginning).
        if (queue.empty() || (new_distance < field[queue.front()]))
          queue.push_front(neighbor);
        else
          queue.push_back(neighbor);

        // Update queue information.
        in_queue[neighbor] = true;
        cumulative_weight += new_distance;
      }

      // Update distance of neighbor.
      field[neighbor] = new_distance;
      update(node, neighbor, new_distance);
    }
  }
}

// Compute geodesic distances
void update_geodesic_distances(vector<float>& distances,
    const geodesic_solver& solver, const vector<int>& sources,
    float max_distance) {
  auto update = [](int node, int neighbor, float new_distance) {};
  auto stop   = [&](int node) { return distances[node] > max_distance; };
  auto exit   = [](int node) { return false; };
  visit_geodesic_graph(distances, solver, sources, update, stop, exit);
}

vector<float> compute_geodesic_distances(const geodesic_solver& solver,
    const vector<int>& sources, float max_distance) {
  auto distances = vector<float>(solver.graph.size(), flt_max);
  for (auto source : sources) distances[source] = 0.0f;
  update_geodesic_distances(distances, solver, sources, max_distance);
  return distances;
}

// Compute all shortest paths from source vertices to any other vertex.
// Paths are implicitly represented: each node is assigned its previous node
// in the path. Graph search early exits when reching end_vertex.
vector<int> compute_geodesic_parents(
    const geodesic_solver& solver, const vector<int>& sources, int end_vertex) {
  auto parents   = vector<int>(solver.graph.size(), -1);
  auto distances = vector<float>(solver.graph.size(), flt_max);
  auto update    = [&parents](int node, int neighbor, float new_distance) {
    parents[neighbor] = node;
  };
  auto stop = [end_vertex](int node) { return node == end_vertex; };
  auto exit = [](int node) { return false; };
  for (auto source : sources) distances[source] = 0.0f;
  visit_geodesic_graph(distances, solver, sources, update, stop, exit);
  return parents;
}

// Sample vertices with a Poisson distribution using geodesic distances
// Sampling strategy is farthest point sampling (FPS): at every step
// take the farthers point from current sampled set until done.
vector<int> sample_vertices_poisson(
    const geodesic_solver& solver, int num_samples) {
  auto verts = vector<int>{};
  verts.reserve(num_samples);
  auto distances = vector<float>(solver.graph.size(), flt_max);
  while (true) {
    auto max_index =
        (int)(std::max_element(distances.begin(), distances.end()) -
              distances.begin());
    verts.push_back(max_index);
    if ((int)verts.size() >= num_samples) break;
    distances[max_index] = 0;
    update_geodesic_distances(distances, solver, {max_index}, flt_max);
  }
  return verts;
}

// Compute the distance field needed to compute a voronoi diagram
vector<vector<float>> compute_voronoi_fields(
    const geodesic_solver& solver, const vector<int>& generators) {
  auto fields = vector<vector<float>>(generators.size());

  // Find max distance from a generator to set an early exit condition for the
  // following distance field computations. This optimization makes
  // computation time weakly dependant on the number of generators.
  auto total = compute_geodesic_distances(solver, generators);
  auto max   = *std::max_element(total.begin(), total.end());
  for (auto i = 0; i < (int)generators.size(); ++i) {
    fields[i]                = vector<float>(solver.graph.size(), flt_max);
    fields[i][generators[i]] = 0;
    fields[i] = compute_geodesic_distances(solver, {generators[i]}, max);
  }
  return fields;
}

vector<vec3f> colors_from_field(
    const vector<float>& field, float scale, const vec3f& c0, const vec3f& c1) {
  auto colors = vector<vec3f>{field.size()};
  for (auto i = 0; i < (int)colors.size(); i++) {
    colors[i] = ((int64_t)(field[i] * scale)) % 2 ? c0 : c1;
  }
  return colors;
}

// `update` is a function that is executed during expansion, every time a node
// is put into queue. `exit` is a function that tells whether to expand the
// current node or perform early exit.
// DO NOT TOUCH THIS FUNCTION!!!
template <typename Update, typename Stop, typename Exit>
void visit_geodesic_graph(vector<float>& field,
    const dual_geodesic_solver& solver, const vector<int>& sources,
    Update&& update, Stop&& stop, Exit&& exit) {
  // DO NOT TOUCH THIS FUNCTION!!!
  /*
     This algortithm uses the heuristic Small Label Fisrt and Large Label Last
     https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm

     Large Label Last (LLL): When extracting nodes from the queue, pick the
     front one. If it weights more than the average weight of the queue, put
     on the back and check the next node. Continue this way.
     Sometimes average_weight is less than every value due to floating point
     errors (doesn't happen with double precision).

     Small Label First (SLF): When adding a new node to queue, instead of
     always pushing it to the end of the queue, if it weights less than the
     front node of the queue, it is put on front. Otherwise the node is put at
     the end of the queue.
  */

  auto in_queue = vector<bool>(solver.graph.size(), false);

  // Cumulative weights of elements in queue. Used to keep track of the
  // average weight of the queue.
  auto cumulative_weight = 0.0;

  // setup queue
  auto queue = std::deque<int>();
  for (auto source : sources) {
    in_queue[source] = true;
    cumulative_weight += field[source];
    queue.push_back(source);
  }

  while (!queue.empty()) {
    auto node           = queue.front();
    auto average_weight = (float)(cumulative_weight / queue.size());

    // Large Label Last (see comment at the beginning)
    for (auto tries = 0; tries < (int)queue.size() + 1; tries++) {
      if (field[node] <= average_weight) break;
      queue.pop_front();
      queue.push_back(node);
      node = queue.front();
    }

    // Remove node from queue.
    queue.pop_front();
    in_queue[node] = false;
    cumulative_weight -= field[node];

    // Check early exit condition.
    if (exit(node)) break;
    if (stop(node)) continue;

    for (auto i = 0; i < solver.graph[node].size(); i++) {
      // Distance of neighbor through this node
      auto new_distance = field[node] + solver.graph[node][i].length;
      auto neighbor     = solver.graph[node][i].node;

      auto old_distance = field[neighbor];
      if (new_distance >= old_distance) continue;

      if (in_queue[neighbor]) {
        // If neighbor already in queue, don't add it.
        // Just update cumulative weight.
        cumulative_weight += new_distance - old_distance;
      } else {
        // If neighbor not in queue, add node to queue using Small Label
        // First (see comment at the beginning).
        if (queue.empty() || (new_distance < field[queue.front()]))
          queue.push_front(neighbor);
        else
          queue.push_back(neighbor);

        // Update queue information.
        in_queue[neighbor] = true;
        cumulative_weight += new_distance;
      }

      // Update distance of neighbor.
      field[neighbor] = new_distance;
      if (update(node, neighbor, new_distance)) return;
    }
  }
}

geodesic_path compute_shortest_path(const dual_geodesic_solver& graph,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const mesh_point& start,
    const mesh_point& end, strip_arena& arena) {
  auto path = geodesic_path{};
  if (start.face == end.face) {
    path.start = start;
    path.end   = end;
    path.strip = {start.face};
  } else {
    auto strip = compute_short_strip(graph, triangles, positions, end, start, arena);
    path = shortest_path(triangles, positions, adjacencies, start, end, strip);
  }

  return path;
};

// geodesic_path compute_shortest_path(const dual_geodesic_solver& graph,
//     const vector<vec3i>& triangles, const vector<vec3f>& positions,
//     const vector<vec3i>& adjacencies, const vector<mesh_point>& points) {
//   // geodesic path
//   auto path = vector<mesh_point>{};
//   for (auto idx = 0; idx < (int)points.size() - 1; idx++) {
//     auto segment = compute_shortest_path(
//         graph, triangles, positions, adjacencies, points[idx], points[idx + 1]);
//     path.insert(path.end(), segment.begin(), segment.end());
//   }
//   return path;
// }

// Compute visualizations for the shortest path connecting a set of points.
vector<vec3f> visualize_shortest_path(const dual_geodesic_solver& graph,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const mesh_point& start,
    const mesh_point& end, strip_arena& arena) {
  // TODO(giacomo): rename to strip_positions() and just return array of centroids.
  auto strip = vector<int>{};
  if (start.face == end.face) {
    strip.push_back(start.face);
    strip.push_back(end.face);
  } else {
    strip = compute_short_strip(graph, triangles, positions, end, start, arena);
  }
  
  auto path = vector<vec3f>{};
  for (auto face : strip)
    path.push_back(graph.centroids[face]);
  return path;
}

vector<vec3f> visualize_shortest_path(const dual_geodesic_solver& graph,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const vector<mesh_point>& points, strip_arena& arena) {
  // geodesic path
  auto path = vector<vec3f>{};
  for (auto idx = 0; idx < (int)points.size() - 1; idx++) {
    auto segment = visualize_shortest_path(graph, triangles, positions,
        adjacencies, points[idx], points[idx + 1], arena);
    path.insert(path.end(), segment.begin(), segment.end());
  }
  return path;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// IMPLEMENTATION OF SURFACE GRADIENT
// -----------------------------------------------------------------------------
namespace yocto {
static vec3f compute_gradient(const vec3i& triangle, const vector<vec3f>& positions,
    const vector<float>& field) {
  auto xy     = positions[triangle.y] - positions[triangle.x];
  auto yz     = positions[triangle.z] - positions[triangle.y];
  auto zx     = positions[triangle.x] - positions[triangle.z];
  auto normal = normalize(cross(zx, xy));
  auto result = vec3f{0, 0, 0};
  result += field[triangle.x] * cross(normal, yz);
  result += field[triangle.y] * cross(normal, zx);
  result += field[triangle.z] * cross(normal, xy);
  return result;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// STRIPS
// -----------------------------------------------------------------------------
namespace yocto {

// `update` is a function that is executed during expansion, every time a node
// is put into queue. `exit` is a function that tells whether to expand the
// current node or perform early exit.
// TODO(fabio): this needs a lof of cleaning
template <typename Update, typename Stop, typename Exit>
void heuristic_visit_geodesic_graph(vector<float>& field,
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, int start, int end, Update&& update,
    Stop&& stop, Exit&& exit) {
  auto destination_pos = eval_position(
      triangles, positions, {end, {1.0f / 3, 1.0f / 3}});

  auto estimate_dist = [&](int face) {
    auto p = eval_position(triangles, positions, {face, {1.0f / 3, 1.0f / 3}});
    return length(p - destination_pos);
  };
  field[start] = estimate_dist(start);

  auto in_queue = vector<bool>(solver.graph.size(), false);

  // Cumulative weights of elements in queue. Used to keep track of the
  // average weight of the queue.
  double cumulative_weight = 0.0;

  // setup queue
  auto queue      = std::deque<int>{};
  in_queue[start] = true;
  cumulative_weight += field[start];
  queue.push_back(start);

  while (!queue.empty()) {
    auto node           = queue.front();
    auto average_weight = (float)(cumulative_weight / queue.size());

    // Large Label Last (see comment at the beginning)
    for (auto tries = 0; tries < (int)queue.size() + 1; tries++) {
      if (field[node] <= average_weight) break;
      queue.pop_front();
      queue.push_back(node);
      node = queue.front();
    }

    // Remove node from queue.
    queue.pop_front();
    in_queue[node] = false;
    cumulative_weight -= field[node];

    // Check early exit condition.
    if (exit(node)) break;
    if (stop(node)) continue;

    for (auto i = 0; i < (int)solver.graph[node].size(); i++) {
      auto neighbor = solver.graph[node][i].node;
      if (neighbor == -1) continue;

      // Distance of neighbor through this node
      auto new_distance = field[node];
      new_distance += solver.graph[node][i].length;
      new_distance += estimate_dist(neighbor);
      new_distance -= estimate_dist(node);

      auto old_distance = field[neighbor];
      if (new_distance >= old_distance) continue;

      if (in_queue[neighbor]) {
        // If neighbor already in queue, don't add it.
        // Just update cumulative weight.
        cumulative_weight += new_distance - old_distance;
      } else {
        // If neighbor not in queue, add node to queue using Small Label
        // First (see comment at the beginning).
        if (queue.empty() || (new_distance < field[queue.front()]))
          queue.push_front(neighbor);
        else
          queue.push_back(neighbor);

        // Update queue information.
        in_queue[neighbor] = true;
        cumulative_weight += new_distance;
      }

      // Update distance of neighbor.
      field[neighbor] = new_distance;
#if YOCTO_BEZIER_PRECISE == 0
      if (update(node, neighbor, new_distance)) return;
#else
      update(node, neighbor, new_distance);
#endif
    }
  }
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// GEODESIC PATH
// -----------------------------------------------------------------------------
namespace yocto {

vector<vec3f> path_positions(const geodesic_path& path,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies) {
  auto get_edge = [&](int f0, int f1) {
    auto k = find_in_vector(adjacencies[f0], f1);
    if (k == -1) return vec2i{-1, -1};
    auto tr = triangles[f0];
    return vec2i{tr[k], tr[mod3(k + 1)]};
  };

  auto result = vector<vec3f>(path.lerps.size() + 2);
  result[0]   = eval_position(triangles, positions, path.start);
  for (auto i = 0; i < (int)path.lerps.size(); i++) {
    auto e = get_edge(path.strip[i], path.strip[i + 1]);
    if (e == vec2i{-1, -1}) continue;
    auto x        = path.lerps[i];
    auto p        = lerp(positions[e.x], positions[e.y], x);
    result[i + 1] = p;
  }
  result.back() = eval_position(triangles, positions, path.end);
  return result;
}

vector<vec3f> path_positions(const mesh_path& path,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies) {
  auto result = vector<vec3f>(path.size());
  for (int i = 0; i < result.size(); i++) {
      result[i] = eval_position(triangles, positions, path[i]);
  }
  return result;
}

vector<float> path_parameters(const geodesic_path& path,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies) {
  return path_parameters(
      path_positions(path, triangles, positions, adjacencies));
}

vector<float> path_parameters(const mesh_path& path,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies) {
  return path_parameters(
      path_positions(path, triangles, positions, adjacencies));
}

float path_length(const geodesic_path& path, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies) {
  return path_length(path_positions(path, triangles, positions, adjacencies));
}

float path_length(const mesh_path& path, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies) {
  return path_length(path_positions(path, triangles, positions, adjacencies));
}

vector<float> path_parameters(const vector<vec3f>& positions) {
  auto len         = 0.0f;
  auto parameter_t = vector<float>(positions.size());
  for (auto i = 0; i < (int)positions.size(); i++) {
    if (i) len += length(positions[i] - positions[i - 1]);
    parameter_t[i] = len;
  }
  for (auto& t : parameter_t) t /= len;
  return parameter_t;
}

float path_length(const vector<vec3f>& positions) {
  auto len = 0.0f;
  for (auto i = 0; i < (int)positions.size(); i++) {
    if (i) len += length(positions[i] - positions[i - 1]);
  }
  return len;
}

// Find barycentric coordinates of a point inside a triangle (a, b, c).
static vec2f barycentric_coordinates(
    const vec2f& point, const vec2f& a, const vec2f& b, const vec2f& c) {
  auto  v0 = b - a, v1 = c - a, v2 = point - a;
  float d00   = dot(v0, v0);
  float d01   = dot(v0, v1);
  float d11   = dot(v1, v1);
  float d20   = dot(v2, v0);
  float d21   = dot(v2, v1);
  float denom = d00 * d11 - d01 * d01;
  return vec2f{d11 * d20 - d01 * d21, d00 * d21 - d01 * d20} / denom;
}

// given a direction expressed in tangent space of the face start,
// continue the path as straight as possible.
geodesic_path compute_straightest_path(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const mesh_point& start, const vec2f& direction, float path_length) {
  auto path  = geodesic_path{};
  path.start = start;
  path.strip.push_back(start.face);

  auto coords    = triangle_coordinates(triangles, positions, start);
  auto prev_face = -2, face = start.face;
  auto len = 0.0f;

  // https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
  auto intersect = [](const vec2f& direction, const vec2f& left,
                       const vec2f& right) {
    auto v1 = -left;
    auto v2 = right - left;
    auto v3 = vec2f{-direction.y, direction.x};
    auto t0 = cross(v2, v1) / dot(v2, v3);
    auto t1 = -dot(left, v3) / dot(v2, v3);
    return pair<float, float>{t0, t1};
  };

  while (len < path_length) {
    // Given the triangle, find which edge is intersected by the line.
    for (auto k = 0; k < 3; ++k) {
      auto neighbor = adjacencies[face][k];
      if (neighbor == prev_face) continue;
      auto left     = coords[k];
      auto right    = coords[mod3(k + 1)];
      auto [t0, t1] = intersect(direction, left, right);
      if (t0 > 0 && t1 >= 0 && t1 <= 1) {
        len = t0;
        if (t0 < path_length) {
          path.lerps.push_back(t1);
          // Step to next face.
          prev_face = face;
          if (neighbor == -1) {
            path_length = len;
            break;
          }
          coords = unfold_face(triangles, positions, coords, face, neighbor);
          face   = adjacencies[face][k];
          path.strip.push_back(face);
        }
        break;
      }
    }
  }

  auto p   = direction * path_length;
  auto uv  = barycentric_coordinates(p, coords[0], coords[1], coords[2]);
  path.end = {face, uv};
  return path;
}

mat2f parallel_transport_rotation(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const geodesic_path& path) {
  if (path.start.face == path.end.face) return mat2f{{0, 0}, {0, 0}};

  auto coords   = unfold_strip(triangles, positions, path.strip, path.start);
  auto a        = coords.back()[0];
  auto b        = coords.back()[1];
  auto y_axis   = normalize(b - a);
  auto rotation = mat2f{};
  rotation.y    = y_axis;
  rotation.x    = {y_axis.y, -y_axis.x};
  rotation      = transpose(rotation);
  return rotation;
}

static float intersect_segments(const vec2f& start1, const vec2f& end1,
    const vec2f& start2, const vec2f& end2) {
  if (end1 == start2) return 0;
  if (end2 == start1) return 1;
  if (start2 == start1) return 0;
  if (end2 == end1) return 1;
  auto a   = end1 - start1;    // direction of line a
  auto b   = start2 - end2;    // direction of line b, reversed
  auto d   = start2 - start1;  // right-hand side
  auto det = a.x * b.y - a.y * b.x;
  assert(det);
  return (a.x * d.y - a.y * d.x) / det;
}

[[maybe_unused]] static bool check_point(const mesh_point& point) {
  assert(point.face != -1);
  assert(point.uv.x >= 0);
  assert(point.uv.y >= 0);
  assert(point.uv.x <= 1);
  assert(point.uv.y <= 1);
  return true;
}

[[maybe_unused]] static bool check_strip(
    const vector<vec3i>& adjacencies, const vector<int>& strip) {
  auto faces = unordered_set<int>{};
  faces.insert(strip[0]);
  for (auto i = 1; i < (int)strip.size(); ++i) {
    if (faces.count(strip[i]) != 0) {
      printf("strip[%d] (face: %d) appears twice\n", i, strip[i]);
    }
    faces.insert(strip[i]);
    assert(find_in_vector(adjacencies[strip[i - 1]], strip[i]) != -1);
    assert(find_in_vector(adjacencies[strip[i]], strip[i - 1]) != -1);
  }
  return true;
}

static void remove_loops_from_strip(vector<int>& strip) {
  auto faces      = unordered_map<int, int>{};
  faces[strip[0]] = 0;
  auto result     = vector<int>(strip.size());
  result[0]       = strip[0];
  auto index      = 1;
  for (auto i = 1; i < (int)strip.size(); ++i) {
    if (faces.count(strip[i]) != 0) {
      // printf("fixing %d (%d)\n", i, strip[i]);
      auto t = faces[strip[i]];
      index  = t + 1;
      continue;
    }
    faces[strip[i]] = i;
    result[index++] = strip[i];
  }
  result.resize(index);
  strip = result;
}

struct funnel_point {
  int   face = 0;
  vec2f pos  = {0, 0};
};

static int max_curvature_point(const vector<funnel_point>& path) {
  // Among vertices around which the path curves, find the vertex
  // with maximum angle. We are going to fix that vertex. Actually, max_index is
  // the index of the first face containing that vertex.
  auto max_index = -1;
  auto max_angle = 0.0f;
  for (auto i = 1; i < (int)path.size() - 1; ++i) {
    auto pos   = path[i].pos;
    auto prev  = path[i - 1].pos;
    auto next  = path[i + 1].pos;
    auto v0    = normalize(pos - prev);
    auto v1    = normalize(next - pos);
    auto angle = 1 - dot(v0, v1);
    if (angle > max_angle) {
      max_index = path[i].face;
      max_angle = angle;
    }
  }
  return max_index;
}

static vector<float> funnel(
    const vector<pair<vec2f, vec2f>>& portals, vector<funnel_point>& points) {
  // Find straight path.
  auto start       = vec2f{0, 0};
  auto apex_index  = 0;
  auto left_index  = 0;
  auto right_index = 0;
  auto apex        = start;
  auto left_bound  = portals[0].first;
  auto right_bound = portals[0].second;

  // Add start point.
  points = vector<funnel_point>{{apex_index, apex}};
  points.reserve(portals.size());

  // @Speed: is this slower than an inlined function?
  auto area = [](const vec2f a, const vec2f b, const vec2f c) {
    return cross(b - a, c - a);
  };

  for (auto i = 1; i < (int)portals.size(); ++i) {
    auto left = portals[i].first, right = portals[i].second;
    // Update right vertex.
    if (area(apex, right_bound, right) <= 0) {
      if (apex == right_bound || area(apex, left_bound, right) > 0) {
        // Tighten the funnel.
        right_bound = right;
        right_index = i;
      } else {
        // Right over left, insert left to path and restart scan from
        // portal left point.
        if (left_bound != apex) {
          points.push_back({left_index, left_bound});
          // Make current left the new apex.
          apex       = left_bound;
          apex_index = left_index;
          // Reset portal
          left_bound  = apex;
          right_bound = apex;
          left_index  = apex_index;
          right_index = apex_index;
          // Restart scan
          i = apex_index;
          continue;
        }
      }
    }

    // Update left vertex.
    if (area(apex, left_bound, left) >= 0) {
      if (apex == left_bound || area(apex, right_bound, left) < 0) {
        // Tighten the funnel.
        left_bound = left;
        left_index = i;
      } else {
        if (right_bound != apex) {
          points.push_back({right_index, right_bound});
          // Make current right the new apex.
          apex       = right_bound;
          apex_index = right_index;
          // Reset portal
          left_bound  = apex;
          right_bound = apex;
          left_index  = apex_index;
          right_index = apex_index;
          // Restart scan
          i = apex_index;
          continue;
        }
      }
    }
  }

  // This happens when we got an apex on the last edge of the strip
  if (points.back().pos != portals.back().first) {
    points.push_back({(int)portals.size() - 1, portals.back().first});
  }
  assert(points.back().pos == portals.back().first);
  assert(points.back().pos == portals.back().second);

  auto lerps = vector<float>();
  lerps.reserve(portals.size());
  for (auto i = 0; i < (int)points.size() - 1; i++) {
    auto a = points[i].pos;
    auto b = points[i + 1].pos;
    for (auto k = points[i].face; k < points[i + 1].face; k++) {
      auto& portal = portals[k];
        auto  s = intersect_segments(a, b, portal.first, portal.second);
      report_floating_point(s);
      auto p = clamp(s, 0.0f, 1.0f);
      lerps.push_back((float)p);
    }
  }

  auto index = 1;
  for (auto i = 1; i < (int)portals.size(); ++i) {
    if ((portals[i].first == points[index].pos) ||
        (portals[i].second == points[index].pos)) {
      points[index].face = i;
      index += 1;
    }
  }
  // max_index = max_curvature_point_double(points);
  assert(lerps.size() == portals.size() - 1);
  return lerps;
}

static vector<int> fix_strip(const vector<vec3i>& adjacencies,
    const vector<int>& strip, int index, int k, bool left) {
  assert(index < (int)strip.size() - 1);
  auto face = strip[index];
  if (!left) k = mod3(k + 2);

  // Create triangle fan that starts at face, walks backward along the strip for
  // a while, exits and then re-enters back.
  auto fan = triangle_fan(adjacencies, face, k, left);
  if (fan.empty()) return strip;

  // fan is a loop of faces which has partial
  // intersection with strip. We wan to remove the intersection from strip and
  // insert there the remaining part of fan, so that we have a new valid strip.
  auto first_strip_intersection = index;
  auto first_fan_intersection   = 0;
  for (auto i = 1; i < (int)fan.size(); i++) {
    auto fan_index   = i;
    auto strip_index = max(index - i, 0);
    if (strip_index < 0) break;
    if (fan[fan_index] == strip[strip_index]) {
      first_strip_intersection = strip_index;
      first_fan_intersection   = fan_index;
    } else {
      break;
    }
  }
  auto second_strip_intersection = index;
  auto second_fan_intersection   = 0;
  for (auto i = 0; i < (int)fan.size(); i++) {
    auto fan_index   = (int)fan.size() - 1 - i;
    auto strip_index = index + i + 1;
    if (strip_index >= (int)strip.size()) break;
    if (fan[fan_index] == strip[strip_index]) {
      second_strip_intersection = strip_index;
      second_fan_intersection   = fan_index;
    } else {
      break;
    }
  }

  if (first_strip_intersection >= second_strip_intersection) return strip;
  if (first_fan_intersection >= second_fan_intersection) return strip;

  auto result = vector<int>{};
  result.reserve(strip.size() + 12);

  // Initial part of original strip, up until intersection with fan.
  result.insert(
      result.end(), strip.begin(), strip.begin() + first_strip_intersection);

  // Append out-flanking part of fan.
  result.insert(result.end(), fan.begin() + first_fan_intersection,
      fan.begin() + second_fan_intersection);

  // Append remaining part of strip after intersection with fan.
  for (auto i = second_strip_intersection; i < (int)strip.size(); ++i)
    result.push_back(strip[i]);

  assert(check_strip(adjacencies, result));
  remove_loops_from_strip(result);
  assert(check_strip(adjacencies, result));
  return result;
}

static void straighten_path(geodesic_path& path, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies) {
  auto init_portals = unfold_funnel_portals(
      triangles, positions, path.strip, path.start, path.end);
  vector<funnel_point> points;
  path.lerps = funnel(init_portals, points);

  auto already_fixed_vertices = unordered_set<int>{};

  struct bend_info {
    int  index      = -1;     // Index of bend in the strip array.
    int  vertex     = -1;     // Vertex of the mesh.
    int  k          = -1;     // Index of the vertex in its containing triangle.
    bool flank_left = false;  // Where to flank the problematic vertex.
  };

  auto max_curvature_point = [&]() {
    // Among vertices around which the path bends, find the vertex
    // with maximum angle. We are going to fix that bend around that vertex.
    auto result    = bend_info{};
    auto max_angle = 0.0;
    for (auto i = 1; i < (int)points.size() - 1; ++i) {
      auto bend      = bend_info{};
      bend.index     = points[i].face;
      auto face      = path.strip[points[i].face];
      auto face_next = path.strip[points[i].face + 1];
      auto edge      = common_edge(triangles[face], triangles[face_next]);
      if (path.lerps[bend.index] == 0) {
        bend.vertex = edge.x;
      } else if (path.lerps[bend.index] == 1) {
        bend.vertex     = edge.y;
        bend.flank_left = true;
      }
      if (already_fixed_vertices.count(bend.vertex)) {
        continue;
      }
      bend.k      = find_in_vector(triangles[face], bend.vertex);
      auto& pos   = points[i].pos;
      auto& prev  = points[i - 1].pos;
      auto& next  = points[i + 1].pos;
      auto  v0    = normalize(pos - prev);
      auto  v1    = normalize(next - pos);
      auto  angle = 1 - dot(v0, v1);
      if (angle > max_angle) {
        max_angle = angle;
        result    = bend;
      }
    }
    if (result.vertex != -1) {
      already_fixed_vertices.insert(result.vertex);
    }
    return result;
  };

  auto bend = max_curvature_point();

#if YOCTO_BEZIER_PRECISE == 0
  auto max_iterations = path.strip.size() * 2;
  for (auto i = 0; i < max_iterations && bend.index != -1; i++) {
#else
  while (bend.index != -1) {
#endif
    path.strip = fix_strip(
        adjacencies, path.strip, bend.index, bend.k, bend.flank_left);

    auto portals = unfold_funnel_portals(
        triangles, positions, path.strip, path.start, path.end);
    path.lerps = funnel(portals, points);
    bend       = max_curvature_point();
  }
}

geodesic_path shortest_path(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const mesh_point& start, const mesh_point& end, const vector<int>& strip) {
  auto path  = geodesic_path{};
  path.start = start;
  path.end   = end;
  path.strip = strip;
  straighten_path(path, triangles, positions, adjacencies);
  return path;
}

mesh_path convert_mesh_path(const vector<vec3i>& triangles,
    const vector<vec3i>& adjacencies, const vector<int>& strip,
    const vector<float>& lerps, const mesh_point& start,
    const mesh_point& end) {
  auto result = mesh_path{};
  result.resize(lerps.size() + 2);
  result[0] = start;
  for (auto i = 0; i < (int)lerps.size(); ++i) {
    auto  k              = find_in_vector(adjacencies[strip[i]], strip[i + 1]);
    vec2f uvw[3]         = {{0, 0}, {1, 0}, {0, 1}};
    auto  a              = uvw[k];
    auto  b              = uvw[mod3(k + 1)];
    auto  uv             = lerp(a, b, lerps[i]);
    result[i + 1] = {strip[i], uv};
  }
  result.back() = end;
  return result;
}

mesh_point eval_path_point(const geodesic_path& path,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const vector<float>& parameter_t,
    float t) {
  if (t <= 0) return path.start;
  if (t >= 1) return path.end;

  // strip with 1 triangle are trivial, just average the uvs
  if (path.start.face == path.end.face) {
    return mesh_point{path.start.face, lerp(path.start.uv, path.end.uv, t)};
  }
  // util function
  auto rotate = [](const vec3f& v, int k) {
    if (mod3(k) == 0)
      return v;
    else if (mod3(k) == 1)
      return vec3f{v.z, v.x, v.y};
    else
      return vec3f{v.y, v.z, v.x};
  };

  // TODO(splinesurf): check which is better betwee linear saerch, binary search
  // and "linear fit" search.
#if 1
  // linear search
  auto i = 0;
  for (; i < (int)parameter_t.size() - 1; i++) {
    if (parameter_t[i + 1] >= t) break;
  }
#else
  // binary search
  // (but "linear fit search" should be faster here)
  // https://blog.demofox.org/2019/03/22/linear-fit-search/
  auto i      = -1;
  auto i_low  = 0;
  auto i_high = (int)parameter_t.size() - 1;
  while (true) {
    i           = (i_high + i_low) / 2;
    auto t_low  = parameter_t[i];
    auto t_high = parameter_t[i + 1];
    if (t_low <= t && t_high >= t) break;

    if (t_low > t) {
      i_high = i;
    } else {
      i_low = i;
    }
  }
#endif

  auto t_low  = parameter_t[i];
  auto t_high = parameter_t[i + 1];
  auto alpha  = (t - t_low) / (t_high - t_low);
  auto face   = path.strip[i];
  auto uv_low = vec2f{0, 0};
  if (i == 0) {
    uv_low = path.start.uv;
  } else {
    auto uvw = lerp(vec3f{1, 0, 0}, vec3f{0, 1, 0}, 1 - path.lerps[i - 1]);
    auto prev_face = path.strip[i - 1];
    auto k         = find_in_vector(adjacencies[face], prev_face);
    uvw            = rotate(uvw, k + 2);
    uv_low         = {uvw.x, uvw.y};
  }
  auto uv_high = vec2f{0, 0};
  if (i == (int)parameter_t.size() - 2) {
    uv_high = path.end.uv;
  } else {
    auto uvw       = lerp(vec3f{1, 0, 0}, vec3f{0, 1, 0}, path.lerps[i]);
    auto next_face = path.strip[i + 1];
    auto k         = find_in_vector(adjacencies[face], next_face);
    uvw            = rotate(uvw, k + 2);
    uv_high        = {uvw.x, uvw.y};
  }
  auto uv = lerp(uv_low, uv_high, alpha);
  return mesh_point{face, uv};
}

mesh_point eval_path_point(const geodesic_path& path,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, float t) {
  auto parameter_t = path_parameters(
      path_positions(path, triangles, positions, adjacencies));
  return eval_path_point(
      path, triangles, positions, adjacencies, parameter_t, t);
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// MESH BEZIER
// -----------------------------------------------------------------------------
namespace yocto {

static void search_strip(strip_arena& arena,
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const mesh_point& start,
    const mesh_point& end) {
  auto start_pos = eval_position(triangles, positions, start);
  auto end_pos   = eval_position(triangles, positions, end);

  auto& weight = arena.field;
  auto& in_queue = arena.in_queue;

  auto estimate_dist = [&](int face) {
    return length(solver.centroids[face] - end_pos);
  };
  weight[start.face] = length(start_pos - end_pos);

  // Cumulative weights of elements in queue. Used to keep track of the
  // average weight of the queue.
  double cumulative_weight = 0.0;

  // setup queue
  auto queue           = std::deque<int>{};
  in_queue[start.face] = true;
  arena.visited.push_back(start.face);

  cumulative_weight += weight[start.face];
  queue.push_back(start.face);


  while (!queue.empty()) {
    auto node           = queue.front();
    auto average_weight = (float)(cumulative_weight / queue.size());

    // Large Label Last (see comment at the beginning)
    for (auto tries = 0; tries < (int)queue.size() + 1; tries++) {
      if (weight[node] <= average_weight) break;
      queue.pop_front();
      queue.push_back(node);
      node = queue.front();
    }

    // Remove node from queue.
    queue.pop_front();
    in_queue[node] = false;
    cumulative_weight -= weight[node];

    for (auto i = 0; i < (int)solver.graph[node].size(); i++) {
      auto neighbor = solver.graph[node][i].node;
      if (neighbor == -1) continue;

      // Distance of neighbor through this node
      auto new_distance = weight[node];
      new_distance += solver.graph[node][i].length;
      new_distance += estimate_dist(neighbor);
      new_distance -= estimate_dist(node);

      auto old_distance = weight[neighbor];
      if (new_distance >= old_distance) continue;

      if (in_queue[neighbor]) {
        // If neighbor already in queue, don't add it.
        // Just update cumulative weight.
        cumulative_weight += new_distance - old_distance;
      } else {
        // If neighbor not in queue, add node to queue using Small Label
        // First (see comment at the beginning).
        if (queue.empty() || (new_distance < weight[queue.front()]))
          queue.push_front(neighbor);
        else
          queue.push_back(neighbor);

        // Update queue information.
        in_queue[neighbor] = true;
        arena.visited.push_back(neighbor);
        cumulative_weight += new_distance;
      }

      // Update distance of neighbor.
      weight[neighbor] = new_distance;
      arena.parents[neighbor] = node;
      if (neighbor == end.face) return;
    }
  }
}

strip_arena::strip_arena(size_t size) {
  parents.assign(size, -1);
  field.assign(size, flt_max);
  in_queue.assign(size, false);
  visited.clear();
}

void strip_arena::cleanup() {
  for (auto& v : visited) {
    parents[v]  = -1;
    field[v]    = flt_max;
    in_queue[v] = false;
  }
  visited.clear();
}

vector<int> compute_short_strip(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const mesh_point& start, const mesh_point& end, strip_arena& arena) {
  if (start.face == end.face) return {start.face};

  auto extract_strip = [](const strip_arena& arena, const mesh_point& end) {
    auto strip = vector<int>{};
    auto node  = end.face;
    strip.reserve((int)yocto::sqrt((float)arena.parents.size()));

    while (node != -1) {
      assert(find_in_vector(strip, node) != 1);
      assert(strip.size() < arena.parents.size());
      strip.push_back(node);
      node = arena.parents[node];
    }
    return strip;
  };

  if(arena.field.empty()) {
    auto temp_arena = strip_arena(solver.graph.size());
    search_strip(temp_arena, solver, triangles, positions, start, end);
    return extract_strip(temp_arena, end);
  } else {
    search_strip(arena, solver, triangles, positions, start, end);
    auto strip = extract_strip(arena, end);
    arena.cleanup();
    return strip;
  }
}

static mesh_point geodesic_lerp(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const mesh_point& start,
    const mesh_point& end, float t, strip_arena& arena) {
  if (start.face == end.face) {
    return mesh_point{start.face, lerp(start.uv, end.uv, t)};
  }
  auto path = compute_shortest_path(
      solver, triangles, positions, adjacencies, start, end, arena);
  auto point = eval_path_point(path, triangles, positions, adjacencies, t);
  assert(check_point(point));
  return point;
}

using spline_polygon = array<mesh_point, 4>;

static pair<spline_polygon, spline_polygon> subdivide_bezier_polygon(
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const spline_polygon& input, float t, strip_arena& arena) {
  auto Q0 = geodesic_lerp(
      solver, triangles, positions, adjacencies, input[0], input[1], t, arena);
  auto Q1 = geodesic_lerp(
      solver, triangles, positions, adjacencies, input[1], input[2], t, arena);
  auto Q2 = geodesic_lerp(
      solver, triangles, positions, adjacencies, input[2], input[3], t, arena);
  auto R0 = geodesic_lerp(solver, triangles, positions, adjacencies, Q0, Q1, t, arena);
  auto R1 = geodesic_lerp(solver, triangles, positions, adjacencies, Q1, Q2, t, arena);
  auto S  = geodesic_lerp(solver, triangles, positions, adjacencies, R0, R1, t, arena);
  return {{input[0], Q0, R0, S}, {S, R1, Q2, input[3]}};
}

vector<mesh_point> compute_bezier_uniform(
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const spline_polygon& control_points, int subdivisions, strip_arena& arena) {
  auto segments = vector<spline_polygon>{control_points};
  auto result   = vector<spline_polygon>();
  for (auto subdivision : range(subdivisions)) {
    result.resize(segments.size() * 2);
    for (auto i = 0; i < (int)segments.size(); i++) {
      auto [split0, split1] = subdivide_bezier_polygon(
          solver, triangles, positions, adjacencies, segments[i], 0.5, arena);
      result[2 * i]     = split0;
      result[2 * i + 1] = split1;
    }
    swap(segments, result);
  }
  return {(mesh_point*)segments.data(),
      (mesh_point*)segments.data() + segments.size() * 4};
}

vector<mesh_point> compute_bezier_uniform(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const vector<mesh_point>& control_points,
    int subdivisions, strip_arena& arena) {
  auto path = vector<mesh_point>{};
  for (auto idx = 0; idx < (int)control_points.size() - 3; idx += 3) {
    auto polygon = spline_polygon{control_points[idx + 0],
        control_points[idx + 1], control_points[idx + 2],
        control_points[idx + 3]};
    auto segment = compute_bezier_uniform(
        solver, triangles, positions, adjacencies, polygon, subdivisions, arena);
    path.insert(path.end(), segment.begin(), segment.end());
  }
  return path;
}

}  // namespace yocto

// -----------------------------------------------------------------------------
// MESH BEZIER
// -----------------------------------------------------------------------------
namespace yocto {

static mesh_point geodesic_lerp(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const mesh_point& a, const mesh_point& b,
    const mesh_point& c, float t0, float t1, strip_arena& arena) {
  // den := (1-t0-t1) + t0 = 1 - t1;
  auto t  = t0 / (1 - t1);
  auto ab = geodesic_lerp(solver, triangles, positions, adjacencies, a, b, t, arena);
  return geodesic_lerp(solver, triangles, positions, adjacencies, ab, c, t1, arena);
}

vec2f tangent_path_direction(const geodesic_path& path, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies, bool start) {
  auto find = [](const vec3i& vec, int x) {
    for (auto i = 0; i < size(vec); i++)
      if (vec[i] == x) return i;
    return -1;
  };

  auto direction = vec2f{};

  if (start) {
    auto start_tr = triangle_coordinates(triangles, positions, path.start);

    if (path.lerps.empty()) {
      direction = interpolate_triangle(
          start_tr[0], start_tr[1], start_tr[2], path.end.uv);
    } else {
      auto x    = path.lerps[0];
      auto k    = find(adjacencies[path.strip[0]], path.strip[1]);
      direction = lerp(start_tr[k], start_tr[(k + 1) % 3], x);
    }
  } else {
    auto end_tr = triangle_coordinates(triangles, positions, path.end);
    if (path.lerps.empty()) {
      direction = interpolate_triangle(
          end_tr[0], end_tr[1], end_tr[2], path.start.uv);
    } else {
      auto x = path.lerps.back();
      auto k = find(
          adjacencies[path.strip.rbegin()[0]], path.strip.rbegin()[1]);
      direction = lerp(end_tr[k], end_tr[(k + 1) % 3], 1 - x);
    }
  }
  return normalize(direction);
}

using spline_polygon = array<mesh_point, 4>;

static bool is_control_polygon_unfoldable(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const spline_polygon& segment) {
  if (segment[0].face != segment[1].face) return false;
  if (segment[1].face != segment[2].face) return false;
  if (segment[2].face != segment[3].face) return false;
  return true;
};

array<spline_polygon, 2> insert_bezier_point(
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const spline_polygon& polygon, float t0, strip_arena& arena) {
  auto t_start = 0.f;
  auto t_end   = 1.f;
  auto points  = polygon;
  while (true) {
    auto Q0 = geodesic_lerp(
        solver, triangles, positions, adjacencies, points[0], points[1], 0.5, arena);
    auto Q1 = geodesic_lerp(
        solver, triangles, positions, adjacencies, points[1], points[2], 0.5, arena);
    auto Q2 = geodesic_lerp(
        solver, triangles, positions, adjacencies, points[2], points[3], 0.5, arena);
    auto R0 = geodesic_lerp(
        solver, triangles, positions, adjacencies, Q0, Q1, 0.5, arena);
    auto R1 = geodesic_lerp(
        solver, triangles, positions, adjacencies, Q1, Q2, 0.5, arena);
    auto S = geodesic_lerp(
        solver, triangles, positions, adjacencies, R0, R1, 0.5, arena);
    auto mid_t = (t_start + t_end) / 2.f;
    if (t0 < mid_t) {
      points[1] = Q0;
      points[2] = R0;
      points[3] = S;
      t_end     = mid_t;
    } else {
      points[0] = S;
      points[1] = R1;
      points[2] = Q2;
      t_start   = mid_t;
    }
    if (is_control_polygon_unfoldable(
            solver, triangles, positions, adjacencies, points))
      break;
  }
  // Compute the parameter t local to the leaf control polygon.
  auto tP_local = (t0 - t_start) / (t_end - t_start);
  // Subdivide the leaf control with De Castljeau creating two new control
  // polygons. They are segment_left and segment_right.
  auto [segment_left, segment_right] = subdivide_bezier_polygon(
      solver, triangles, positions, adjacencies, points, tP_local, arena);
  auto left_side  = compute_shortest_path(solver, triangles, positions,
       adjacencies, segment_left.back(), segment_left[2], arena);
  auto right_side = compute_shortest_path(solver, triangles, positions,
      adjacencies, segment_right[0], segment_right[1], arena);
  // P is the inserted mesh point that sepraters segment_left and
  // segment_right.
  // assert(segment_left[3] == segment_right[0]);
  auto P = segment_right[0];
  // left part
  {
    auto Pp2_len = path_length(left_side, triangles, positions, adjacencies);
    auto Pp2_dir = tangent_path_direction(left_side, triangles, positions, adjacencies);
    //    assert(left_leaf.start == P);
    auto delta_len = t_start * Pp2_len / (t0 - t_start);
    auto path      = compute_straightest_path(
             triangles, positions, adjacencies, P, Pp2_dir, delta_len + Pp2_len);
    auto Pp2 = path.end;
    auto Pp1 = geodesic_lerp(
        solver, triangles, positions, adjacencies, polygon[0], polygon[1], t0, arena);
    segment_left = {polygon[0], Pp1, Pp2, P};
  }
  // right part
  {
    auto Pp1_len = path_length(right_side, triangles, positions, adjacencies);
    auto Pp1_dir = tangent_path_direction(right_side, triangles, positions, adjacencies);
    auto delta_len = (1 - t_end) / (t_end - t0) * Pp1_len;
    auto path      = compute_straightest_path(
             triangles, positions, adjacencies, P, Pp1_dir, delta_len + Pp1_len);
    auto Pp1 = path.end;
    auto Pp2 = geodesic_lerp(
        solver, triangles, positions, adjacencies, polygon[2], polygon[3], t0, arena);
    segment_right = {P, Pp1, Pp2, polygon[3]};
  }
  return {segment_left, segment_right};
}

static bool is_bezier_straight_enough(const geodesic_path& a,
    const geodesic_path& b, const geodesic_path& c,
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    float min_curve_size, float precision) {
  // TODO(giacomo): we don't need all positions!
  // auto a_positions = path_positions(solver, triangles, positions,
  // adjacencies,  a); auto b_positions = path_positions(solver, triangles,
  // positions, adjacencies,  b); auto c_positions = path_positions(solver,
  // triangles, positions, adjacencies,  c);

  {
    // On curve apex we may never reach straightess, so we check curve
    // length.
    auto pos  = array<vec3f, 4>{};
    pos[0]    = eval_position(triangles, positions, a.start);
    pos[1]    = eval_position(triangles, positions, b.start);
    pos[2]    = eval_position(triangles, positions, c.start);
    pos[3]    = eval_position(triangles, positions, c.end);
    float len = 0;
    for (auto i : range(3)) {
      len += length(pos[i] - pos[i + 1]);
    }
    if (len < min_curve_size) return true;
  }

  {
    auto dir0 = tangent_path_direction(a, triangles, positions, adjacencies, false);  // end
    auto dir1 = tangent_path_direction(b, triangles, positions, adjacencies, true);  // start
    auto angle1 = cross(dir0, dir1);
    if (fabs(angle1) > precision) {
      // printf("a1: %f > %f\n", angle1, params.precision);
      return false;
    }
  }

  {
    auto dir0 = tangent_path_direction(b, triangles, positions, adjacencies, false);  // end
    auto dir1 = tangent_path_direction(c, triangles, positions, adjacencies, true);  // start
    auto angle1 = cross(dir0, dir1);
    if (fabs(angle1) > precision) {
      // printf("a2: %f > %f\n", angle1, params.precision);
      return false;
    }
  }

  return true;
}

static void compute_bezier_adaptive_recursive(
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const spline_polygon& input, int max_depth, float min_curve_size, float precision,
    vector<mesh_point>& result, strip_arena& arena, int depth = 0) {
  // resulting beziers: (P0, Q0, R0, S) (S, R1, Q2, P3)

  if (depth > max_depth) {
    return;
  }
  auto [P0, P1, P2, P3] = input;

  auto P0_P1 = compute_shortest_path(
      solver, triangles, positions, adjacencies, P0, P1, arena);
  auto P1_P2 = compute_shortest_path(
      solver, triangles, positions, adjacencies, P1, P2, arena);
  auto P2_P3 = compute_shortest_path(
      solver, triangles, positions, adjacencies, P2, P3, arena);

  if (is_bezier_straight_enough(P0_P1, P1_P2, P2_P3, solver, triangles,
          positions, adjacencies, min_curve_size, precision)) {
    result.push_back(P0);
    result.push_back(P1);
    result.push_back(P2);
    result.push_back(P3);
    return;
  }

  auto Q0    = eval_path_point(P0_P1, triangles, positions, adjacencies, 0.5);
  auto Q1    = eval_path_point(P1_P2, triangles, positions, adjacencies, 0.5);
  auto Q2    = eval_path_point(P2_P3, triangles, positions, adjacencies, 0.5);
  auto Q0_Q1 = compute_shortest_path(
      solver, triangles, positions, adjacencies, Q0, Q1, arena);
  auto Q1_Q2 = compute_shortest_path(
      solver, triangles, positions, adjacencies, Q1, Q2, arena);

  auto R0    = eval_path_point(Q0_Q1, triangles, positions, adjacencies, 0.5);
  auto R1    = eval_path_point(Q1_Q2, triangles, positions, adjacencies, 0.5);
  auto R0_R1 = compute_shortest_path(
      solver, triangles, positions, adjacencies, R0, R1, arena);

  auto S = eval_path_point(R0_R1, triangles, positions, adjacencies, 0.5);

  compute_bezier_adaptive_recursive(solver, triangles, positions,
      adjacencies, {P0, Q0, R0, S}, max_depth, min_curve_size, precision, result, arena, depth + 1);
  compute_bezier_adaptive_recursive(solver, triangles, positions,
      adjacencies, {S, R1, Q2, P3}, max_depth, min_curve_size, precision, result, arena, depth + 1);
}

vector<mesh_point> compute_bezier_adaptive(
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const spline_polygon& control_points, int max_depth, float min_curve_size, float precision, strip_arena& arena) {
  auto result = vector<mesh_point>{};
  compute_bezier_adaptive_recursive(solver, triangles, positions,
      adjacencies, control_points, max_depth, min_curve_size, precision, result, arena);
  return result;
}

vector<mesh_point> compute_bezier_path(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const spline_polygon& control_points,
    const spline_params& params, strip_arena& arena) {
  switch (params.algorithm) {
    case spline_algorithm::de_casteljau_uniform: {
      return compute_bezier_uniform(
          solver, triangles, positions, adjacencies, control_points, params.subdivisions, arena);
    }
    case spline_algorithm::de_casteljau_adaptive: {
      return compute_bezier_adaptive(
          solver, triangles, positions, adjacencies, control_points, params.max_depth,
          params.min_curve_size, params.precision, arena);
    }
    case spline_algorithm::lane_riesenfeld_uniform: {
      return lane_riesenfeld_uniform(solver, triangles, positions, adjacencies,
          control_points, params.subdivisions, arena);
    }
    case spline_algorithm::lane_riesenfeld_adaptive: {
      return lane_riesenfeld_adaptive(
          solver, triangles, positions, adjacencies, control_points, params, arena);
    }
    default: return {};
  }
}

vector<mesh_point> compute_bezier_path(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const vector<mesh_point>& control_points,
    const spline_params& params, strip_arena& arena) {
  auto path = vector<mesh_point>{};
  for (auto idx = 0; idx < (int)control_points.size() - 3; idx += 3) {
    auto polygon = spline_polygon{control_points[idx + 0],
        control_points[idx + 1], control_points[idx + 2],
        control_points[idx + 3]};
    auto segment = compute_bezier_path(
        solver, triangles, positions, adjacencies, polygon, params, arena);
    path.insert(path.end(), segment.begin(), segment.end());
  }
  return path;
}

}  // namespace yocto


// -----------------------------------------------------------------------------
// LANE-RIESENFELD
// -----------------------------------------------------------------------------

namespace yocto {

vector<mesh_point> lane_riesenfeld_uniform(
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const array<mesh_point, 4>& control_points, int num_subdivisions, strip_arena& arena) {
  auto size = 7;
  struct parametric_path {
    geodesic_path path = {};
    vector<float> t    = {};
  };
  parametric_path curr_path = {};
  parametric_path gamma01   = {};
  parametric_path gamma32   = {};
  auto            prev      = mesh_point{};
  auto            curr      = mesh_point{};
  auto            q         = vector<mesh_point>(size);

  {
    auto& p      = control_points;
    gamma01.path = compute_shortest_path(
        solver, triangles, positions, adjacencies, p[0], p[1], arena);
    gamma01.t = path_parameters(
        gamma01.path, triangles, positions, adjacencies);
    gamma32.path = compute_shortest_path(
        solver, triangles, positions, adjacencies, p[3], p[2], arena);
    gamma32.t = path_parameters(
        gamma32.path, triangles, positions, adjacencies);
    curr_path.path = compute_shortest_path(
        solver, triangles, positions, adjacencies, p[1], p[2], arena);
    curr_path.t = path_parameters(
        curr_path.path, triangles, positions, adjacencies);
    q[0]           = p[0];
    q[1]           = eval_path_point(gamma01.path, triangles, positions, adjacencies,
                   gamma01.t, 0.25);
    auto p0p1      = eval_path_point(gamma01.path, triangles, positions, adjacencies,
              gamma01.t, 0.5);
    auto p1p2      = eval_path_point(curr_path.path, triangles, positions, adjacencies,
                                     curr_path.t, 0.5);
    curr_path.path = compute_shortest_path(
        solver, triangles, positions, adjacencies, p0p1, p1p2, arena);
    curr_path.t = path_parameters(
        curr_path.path, triangles, positions, adjacencies);
    q[2]           = eval_path_point(curr_path.path, triangles, positions, adjacencies,
                  curr_path.t, 0.25);
    auto p2p3      = eval_path_point(gamma32.path, triangles, positions, adjacencies,
              gamma32.t, 0.5);
    prev           = eval_path_point(curr_path.path, triangles, positions, adjacencies,
                  curr_path.t, 5 / 8.f);
    curr_path.path = compute_shortest_path(
        solver, triangles, positions, adjacencies, p1p2, p2p3);
    curr_path.t = path_parameters(
        curr_path.path, triangles, positions, adjacencies);
    curr = eval_path_point(curr_path.path, triangles, positions, adjacencies,
         curr_path.t, 3 / 8.f);
    q[3] = geodesic_lerp(
        solver, triangles, positions, adjacencies, prev, curr, 0.5, arena);
    q[4] = eval_path_point(curr_path.path, triangles, positions, adjacencies,
         curr_path.t, 0.75);
    q[5] = eval_path_point(gamma32.path, triangles, positions, adjacencies,
         gamma32.t, 0.25);
    q[6] = p[3];
  }

  auto p = vector<mesh_point>{};

  for (int subdiv = 0; subdiv < num_subdivisions; subdiv++) {
    std::swap(p, q);

    auto new_size = 2 * size - 3;
    q.resize(new_size);
    q[0]           = p[0];
    q[1]           = eval_path_point(gamma01.path, triangles, positions, adjacencies,
                   gamma01.t, 1.f / pow((float)2, (float)3 + subdiv));
    curr_path.path = compute_shortest_path(
        solver, triangles, positions, adjacencies, p[1], p[2]);
    curr_path.t = path_parameters(
        curr_path.path, triangles, positions, adjacencies);
    q[2]           = eval_path_point(curr_path.path, triangles, positions, adjacencies,
                   curr_path.t, 0.25);
    prev           = eval_path_point(curr_path.path, triangles, positions, adjacencies,
                   curr_path.t, 5 / 8.f);
    curr_path.path = compute_shortest_path(
        solver, triangles, positions, adjacencies, p[2], p[3], arena);
    curr_path.t = path_parameters(
        curr_path.path, triangles, positions, adjacencies);
    curr = eval_path_point(curr_path.path, triangles, positions, adjacencies,
        curr_path.t, 0.25);
    q[3] = geodesic_lerp(
        solver, triangles, positions, adjacencies, prev, curr, 0.5, arena);
    prev = eval_path_point(curr_path.path, triangles, positions, adjacencies,
        curr_path.t, 0.5);
    for (auto j = 4; j < 2 * size - 8; j += 2) {
      q[j] = prev;
      prev = eval_path_point(curr_path.path, triangles, positions, adjacencies,
          curr_path.t, 0.75);
      curr_path.path = compute_shortest_path(solver, triangles, positions,
          adjacencies, p[j / 2 + 1], p[j / 2 + 2]);
      curr_path.t    = path_parameters(
             curr_path.path, triangles, positions, adjacencies);
      curr     = eval_path_point(curr_path.path, triangles, positions, adjacencies,
               curr_path.t, 0.25);
      q[j + 1] = geodesic_lerp(
          solver, triangles, positions, adjacencies, prev, curr, 1 / 2.f, arena);
      prev = eval_path_point(curr_path.path, triangles, positions, adjacencies,
          curr_path.t, 0.5);
    }
    q[2 * size - 8] = prev;
    {
      auto qq = &q[new_size - 4];
      auto pp = &p[size - 4];
      prev    = eval_path_point(curr_path.path, triangles, positions, adjacencies,
              curr_path.t, 0.75);
      curr_path.path = compute_shortest_path(
          solver, triangles, positions, adjacencies, pp[1], pp[2]);
      curr_path.t = path_parameters(
          curr_path.path, triangles, positions, adjacencies);
      curr  = eval_path_point(curr_path.path, triangles, positions, adjacencies,
           curr_path.t, 3 / 8.f);
      qq[0] = geodesic_lerp(
          solver, triangles, positions, adjacencies, prev, curr, 0.5, arena);
      qq[1] = eval_path_point(curr_path.path, triangles, positions, adjacencies,
          curr_path.t, 0.75);
      qq[2] = eval_path_point(gamma32.path, triangles, positions, adjacencies,
          gamma32.t, 1.f / pow((float)2, (float)3 + subdiv));
      qq[3] = pp[3];
    }
    size = new_size;
  }
  return q;
}

static bool is_bezier_straight_enough(const geodesic_path& a,
    const geodesic_path& b, const geodesic_path& c,
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const spline_params& params) {
  // TODO(giacomo): we don't need all positions!
  // auto a_positions = path_positions(solver, triangles, positions,
  // adjacencies,  a); auto b_positions = path_positions(solver, triangles,
  // positions, adjacencies,  b); auto c_positions = path_positions(solver,
  // triangles, positions, adjacencies,  c);

  {
    // On curve apex we may never reach straightess, so we check curve
    // length.
    auto pos  = array<vec3f, 4>{};
    pos[0]    = eval_position(triangles, positions, a.start);
    pos[1]    = eval_position(triangles, positions, b.start);
    pos[2]    = eval_position(triangles, positions, c.start);
    pos[3]    = eval_position(triangles, positions, c.end);
    float len = 0;
    for (auto i : range(3)) {
      len += length(pos[i] - pos[i + 1]);
    }
    if (len < params.min_curve_size) return true;
  }

  {
    auto dir0 = tangent_path_direction(a, triangles, positions, adjacencies, false);  // end
    auto dir1 = tangent_path_direction(b, triangles, positions, adjacencies, true);  // start
    auto angle1 = cross(dir0, dir1);
    if (fabs(angle1) > params.precision) {
      // printf("a1: %f > %f\n", angle1, params.precision);
      return false;
    }
  }

  {
    auto dir0 = tangent_path_direction(b, triangles, positions, adjacencies, false);  // end
    auto dir1 = tangent_path_direction(c, triangles, positions, adjacencies, true);  // start
    auto angle1 = cross(dir0, dir1);
    if (fabs(angle1) > params.precision) {
      // printf("a2: %f > %f\n", angle1, params.precision);
      return false;
    }
  }

  return true;
}

static mesh_point lane_riesenfeld_init(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const mesh_point& a, const mesh_point& b,
    const mesh_point& c, strip_arena& arena) {
  auto Q0 = geodesic_lerp(
      solver, triangles, positions, adjacencies, a, b, 5 / 8.f, arena);
  auto Q1 = geodesic_lerp(
      solver, triangles, positions, adjacencies, b, c, 3 / 8.f, arena);
  return geodesic_lerp(solver, triangles, positions, adjacencies, Q0, Q1, 0.5, arena);
}

struct spline_node {
  std::array<mesh_point, 4>    points  = {};
  std::array<geodesic_path, 3> lines   = {};
  vec2f                        t       = {};
  int                          depth   = 0;
  bool                         is_good = false;
};

static mesh_point eval_path_point(const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies, const
    geodesic_path& path, float t) {
  return eval_path_point(path, triangles, positions, adjacencies, t);
}

static mesh_point lane_riesenfeld_boundary(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const mesh_point& a, const mesh_point& b,
    const mesh_point& c, const bool& left, strip_arena& arena) {
  auto Q0 = mesh_point{};
  auto Q1 = mesh_point{};
  if (left) {
    Q0 = geodesic_lerp(
        solver, triangles, positions, adjacencies, a, b, 5 / 8.f, arena);
    Q1 = geodesic_lerp(solver, triangles, positions, adjacencies, b, c, 0.25, arena);

  } else {
    Q0 = geodesic_lerp(solver, triangles, positions, adjacencies, a, b, 0.75, arena);
    Q1 = geodesic_lerp(
        solver, triangles, positions, adjacencies, b, c, 3 / 8.f, arena);
  }
  return geodesic_lerp(solver, triangles, positions, adjacencies, Q0, Q1, 0.5, arena);
}

static mesh_point lane_riesenfeld_boundary(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const geodesic_path& l0,
    const geodesic_path& l1, const bool& left, strip_arena& arena) {
  auto Q0 = mesh_point{};
  auto Q1 = mesh_point{};
  if (left) {
    Q0 = eval_path_point(
        solver, triangles, positions, adjacencies, l0, 5 / 8.f);
    Q1 = eval_path_point(solver, triangles, positions, adjacencies, l1, 0.25);

  } else {
    Q0 = eval_path_point(solver, triangles, positions, adjacencies, l0, 0.75);
    Q1 = eval_path_point(
        solver, triangles, positions, adjacencies, l1, 3 / 8.f);
  }
  return geodesic_lerp(solver, triangles, positions, adjacencies, Q0, Q1, 0.5, arena);
}

static mesh_point lane_riesenfeld_regular(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const mesh_point& a, const mesh_point& b,
    const mesh_point& c, strip_arena& arena) {
  auto Q0 = geodesic_lerp(
      solver, triangles, positions, adjacencies, a, b, 0.75, arena);
  auto Q1 = geodesic_lerp(
      solver, triangles, positions, adjacencies, b, c, 0.25, arena);
  return geodesic_lerp(solver, triangles, positions, adjacencies, Q0, Q1, 0.5, arena);
}

static mesh_point lane_riesenfeld_regular(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const geodesic_path& l0,
    const geodesic_path& l1, strip_arena& arena) {
  auto Q0 = eval_path_point(
      solver, triangles, positions, adjacencies, l0, 0.75);
  auto Q1 = eval_path_point(
      solver, triangles, positions, adjacencies, l1, 0.25);
  return geodesic_lerp(solver, triangles, positions, adjacencies, Q0, Q1, 0.5, arena);
}

static pair<bool, vector<mesh_point>> handle_boundary_node(
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const spline_node& leaf, const vector<int>& new_ones_entries, strip_arena& arena) {
  vector<mesh_point> new_ones(5);
  auto               k = leaf.depth + 1;
  if (new_ones_entries[0] > 3 &&
      new_ones_entries.back() < pow((float)2, (float)k) - 1)
    return {false, new_ones};
  else if (new_ones_entries[0] <= 3) {
    if (new_ones_entries[0] == 0) {
      new_ones[0] = leaf.points[0];
      new_ones[1] = eval_path_point(
          solver, triangles, positions, adjacencies, leaf.lines[0], 0.5);
      new_ones[2] = eval_path_point(
          solver, triangles, positions, adjacencies, leaf.lines[1], 0.25);
      new_ones[3] = lane_riesenfeld_boundary(solver, triangles, positions,
          adjacencies, leaf.lines[1], leaf.lines[2], true, arena);
      new_ones[4] = eval_path_point(
          solver, triangles, positions, adjacencies, leaf.lines[2], 0.5);
    } else if (new_ones_entries[0] == 2) {
      new_ones[0] = eval_path_point(
          solver, triangles, positions, adjacencies, leaf.lines[0], 0.25);
      new_ones[1] = lane_riesenfeld_boundary(solver, triangles, positions,
          adjacencies, leaf.lines[0], leaf.lines[1], true, arena);
      new_ones[2] = eval_path_point(
          solver, triangles, positions, adjacencies, leaf.lines[1], 0.5);
      new_ones[3] = lane_riesenfeld_regular(solver, triangles, positions,
          adjacencies, leaf.lines[1], leaf.lines[2], arena);
      new_ones[4] = eval_path_point(
          solver, triangles, positions, adjacencies, leaf.lines[2], 0.5);
    } else
      assert(false);
  } else {
    if (new_ones_entries.back() == pow((float)2, (float)k) + 2) {
      new_ones[0] = eval_path_point(
          solver, triangles, positions, adjacencies, leaf.lines[0], 0.5);
      new_ones[1] = lane_riesenfeld_boundary(solver, triangles, positions,
          adjacencies, leaf.lines[0], leaf.lines[1], false, arena);
      new_ones[2] = eval_path_point(
          solver, triangles, positions, adjacencies, leaf.lines[1], 0.75);
      new_ones[3] = eval_path_point(
          solver, triangles, positions, adjacencies, leaf.lines[2], 0.5);
      new_ones[4] = leaf.points[3];
    } else if (new_ones_entries.back() == pow((float)2, (float)k)) {
      new_ones[0] = eval_path_point(
          solver, triangles, positions, adjacencies, leaf.lines[0], 0.5);
      new_ones[1] = lane_riesenfeld_regular(solver, triangles, positions,
          adjacencies, leaf.lines[0], leaf.lines[1], arena);
      new_ones[2] = eval_path_point(
          solver, triangles, positions, adjacencies, leaf.lines[1], 0.5);
      new_ones[3] = lane_riesenfeld_boundary(solver, triangles, positions,
          adjacencies, leaf.lines[1], leaf.lines[2], false, arena);
      new_ones[4] = eval_path_point(
          solver, triangles, positions, adjacencies, leaf.lines[2], 0.75);
    } else
      assert(false);
  }
  return {true, new_ones};
}
static pair<spline_node, spline_node> lane_riesenfeld_split_node(
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const spline_node& leaf, bool& max_depth_reached, strip_arena& arena) {
  auto curr_t     = (leaf.t.x + leaf.t.y) / 2;
  int  curr_entry = (int)(pow((float)2, (float)leaf.depth + 1) * curr_t);
  curr_entry += 3;
  if (curr_entry % 2) {
    max_depth_reached = true;
    return {leaf, {}};
  }
  auto curr_entries               = vector<int>{curr_entry - 4, curr_entry - 3,
      curr_entry - 2, curr_entry - 1, curr_entry};
  auto [are_boundaries, new_ones] = handle_boundary_node(
      solver, triangles, positions, adjacencies, leaf, curr_entries, arena);
  if (!are_boundaries) {
    new_ones[0] = eval_path_point(
        solver, triangles, positions, adjacencies, leaf.lines[0], 0.5);
    new_ones[1] = lane_riesenfeld_regular(solver, triangles, positions,
        adjacencies, leaf.lines[0], leaf.lines[1], arena);
    new_ones[2] = eval_path_point(
        solver, triangles, positions, adjacencies, leaf.lines[1], 0.5);
    new_ones[3] = lane_riesenfeld_regular(solver, triangles, positions,
        adjacencies, leaf.lines[1], leaf.lines[2], arena);
    new_ones[4] = eval_path_point(
        solver, triangles, positions, adjacencies, leaf.lines[2], 0.5);
  }
  auto L01 = compute_shortest_path(
      solver, triangles, positions, adjacencies, new_ones[0], new_ones[1]);
  auto L12 = compute_shortest_path(
      solver, triangles, positions, adjacencies, new_ones[1], new_ones[2]);
  auto L23 = compute_shortest_path(
      solver, triangles, positions, adjacencies, new_ones[2], new_ones[3]);
  spline_node P0 = {{new_ones[0], new_ones[1], new_ones[2], new_ones[3]},
      {L01, L12, L23}, {leaf.t.x, curr_t}, leaf.depth + 1};
  L01            = compute_shortest_path(
                 solver, triangles, positions, adjacencies, new_ones[3], new_ones[4]);
  spline_node P1 = {{new_ones[1], new_ones[2], new_ones[3], new_ones[4]},
      {L12, L23, L01}, {curr_t, leaf.t.y}, leaf.depth + 1};
  return {P0, P1};
}


vector<mesh_point> lane_riesenfeld_adaptive(
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const array<mesh_point, 4>& polygon, const spline_params& params, strip_arena& arena) {
  auto q = vector<mesh_point>(7);

  auto& p = polygon;
  q[0]    = p[0];
  q[1]    = geodesic_lerp(
         solver, triangles, positions, adjacencies, p[0], p[1], 1 / 4.f, arena);
  auto p0p1 = geodesic_lerp(
      solver, triangles, positions, adjacencies, p[0], p[1], 1 / 2.f, arena);
  auto p1p2 = geodesic_lerp(
      solver, triangles, positions, adjacencies, p[1], p[2], 1 / 2.f, arena);
  q[2] = geodesic_lerp(
      solver, triangles, positions, adjacencies, p0p1, p1p2, 1 / 4.f, arena);
  auto p2p3 = geodesic_lerp(
      solver, triangles, positions, adjacencies, p[2], p[3], 1 / 2.f, arena);
  q[3] = lane_riesenfeld_init(
      solver, triangles, positions, adjacencies, p0p1, p1p2, p2p3, arena);
  q[4] = geodesic_lerp(
      solver, triangles, positions, adjacencies, p1p2, p2p3, 3 / 4.f, arena);
  q[5] = geodesic_lerp(
      solver, triangles, positions, adjacencies, p2p3, p[3], 1 / 2.f, arena);
  q[6]     = p[3];
  auto L01 = compute_shortest_path(
      solver, triangles, positions, adjacencies, q[0], q[1]);
  auto L12 = compute_shortest_path(
      solver, triangles, positions, adjacencies, q[1], q[2]);
  auto L23 = compute_shortest_path(
      solver, triangles, positions, adjacencies, q[2], q[3]);
  spline_node P0 = {{q[0], q[1], q[2], q[3]}, {L01, L12, L23}, {0, 0.25}, 2};
  L01            = compute_shortest_path(
                 solver, triangles, positions, adjacencies, q[3], q[4]);
  spline_node P1 = {{q[1], q[2], q[3], q[4]}, {L12, L23, L01}, {0.25, 0.5}, 2};
  L12            = compute_shortest_path(
                 solver, triangles, positions, adjacencies, q[4], q[5]);
  spline_node P2 = {{q[2], q[3], q[4], q[5]}, {L23, L01, L12}, {0.5, 0.75}, 2};
  L23            = compute_shortest_path(
                 solver, triangles, positions, adjacencies, q[5], q[6]);
  spline_node P3 = {{q[3], q[4], q[5], q[6]}, {L01, L12, L23}, {0.75, 1}, 2};

  P0.is_good = is_bezier_straight_enough(P0.lines[0], P0.lines[1], P0.lines[2],
      solver, triangles, positions, adjacencies, params);
  P1.is_good = is_bezier_straight_enough(P1.lines[0], P1.lines[1], P1.lines[2],
      solver, triangles, positions, adjacencies, params);
  P2.is_good = is_bezier_straight_enough(P2.lines[0], P2.lines[1], P2.lines[2],
      solver, triangles, positions, adjacencies, params);
  P3.is_good = is_bezier_straight_enough(P3.lines[0], P3.lines[1], P3.lines[2],
      solver, triangles, positions, adjacencies, params);
  // auto                    count_path = 16;
  // auto                    count_eval = 8;
  std::deque<spline_node> Q;
  Q.push_back(P3);
  Q.push_back(P2);
  Q.push_back(P1);
  Q.push_back(P0);
  auto P                 = vector<spline_node>{};
  bool max_depth_reached = false;
  while (!Q.empty()) {
    auto curr = Q.back();
    Q.pop_back();
    if (P.size() > 0 && P.back().depth == curr.depth && curr.is_good) {
      P.push_back(curr);

    } else {
      auto [left, right] = lane_riesenfeld_split_node(
          solver, triangles, positions, adjacencies, curr, max_depth_reached, arena);
      if (max_depth_reached) {
        curr.is_good = true;
        if (P.size() > 0) {
          auto last = P.back();
          auto L    = compute_shortest_path(solver, triangles, positions,
                 adjacencies, last.points[0], curr.points[0]);
          if (is_bezier_straight_enough(L, curr.lines[0], curr.lines[1], solver,
                  triangles, positions, adjacencies, params))
            P.push_back(curr);
          else {
            P.pop_back();
            last = {{last.points[0], curr.points[0], curr.points[1],
                        curr.points[2]},
                {L, curr.lines[0], curr.lines[1]}, last.t, last.depth, false};
            Q.push_back(curr);
            Q.push_back(last);
          }
        } else
          P.push_back(curr);

        max_depth_reached = false;
      } else {
        left.is_good  = is_bezier_straight_enough(left.lines[0], left.lines[1],
             left.lines[2], solver, triangles, positions, adjacencies, params);
        right.is_good = is_bezier_straight_enough(right.lines[0],
            right.lines[1], right.lines[2], solver, triangles, positions,
            adjacencies, params);
        if (left.is_good && right.is_good) {
          if (P.size() == 0) {
            P.push_back(left);
            P.push_back(right);
          } else {
            auto last = P.back();
            auto L    = compute_shortest_path(solver, triangles, positions,
                   adjacencies, last.points[0], left.points[0]);
            if (is_bezier_straight_enough(L, left.lines[0], left.lines[1],
                    solver, triangles, positions, adjacencies, params)) {
              last     = {{last.points[0], left.points[0], left.points[1],
                          left.points[2]},
                  {L, left.lines[0], left.lines[1]}, last.t, last.depth, true};
              P.back() = last;
              P.push_back(left);
              P.push_back(right);
            } else if (left.depth < last.depth) {
              left.is_good  = false;
              right.is_good = false;
              Q.push_back(right);
              Q.push_back(left);
            } else {
              P.pop_back();
              last = {{last.points[0], left.points[0], left.points[1],
                          left.points[2]},
                  {L, left.lines[0], left.lines[1]}, last.t, last.depth, false};
              Q.push_back(right);
              Q.push_back(left);
              Q.push_back(last);
            }
          }
        } else {
          Q.push_back(right);
          Q.push_back(left);
        }
      }
    }
  }
  // std::cout << "Number of paths";
  // std::cout << count_path << std::endl;
  // std::cout << "Number of eval";
  // std::cout << count_eval << std::endl;
  // std::cout << "Precision";
  // std::cout << yocto::pow(2, -params.precision) / pif * 180 << std::endl;
  auto polyline = vector<mesh_point>{};
  for (auto i = 0; i < (int)P.size(); ++i) {
    if (i == 0) {
      for (auto j = 0; j < 4; ++j) {
        polyline.push_back(P[i].points[j]);
      }
    } else
      polyline.push_back(P[i].points.back());
  }

  return polyline;
}

} // namespace yocto
