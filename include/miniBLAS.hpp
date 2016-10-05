#pragma once

#include <cstdint>
#include <cmath>

#if defined(__GNUC__)
#   include <x86intrin.h>
#   include <malloc.h>
#   define ALIGN32 __attribute__((aligned(32)))
#   define ALIGN16 __attribute__((aligned(16)))
#   define malloc_align(size, align) memalign((align), (size))
#   define free_align(ptr) free(ptr)
#else
#   include <intrin.h>
#   define ALIGN32 __declspec(align(32))
#   define ALIGN16 __declspec(align(16))
#   define malloc_align(size, align) _aligned_malloc((size), (align))
#   define free_align(ptr) _aligned_free(ptr)
#endif

#define USE_SSE2
#define USE_SSE3
#define USE_SSE4

namespace miniBLAS
{

class ALIGN16 Vertex;
class ALIGN16 VertexI;

template<typename T>
class ALIGN16 Vec4Base
{
	static_assert(sizeof(T) == 4, "only 4-byte length type allowed");
protected:
	union
	{
		__m128 float_dat;
		__m128i int_dat;
		T data[4];
		struct  
		{
			T x, y, z, w;
		};
		struct
		{
			float float_x, float_y, float_z, float_w;
		};
		struct
		{
			int32_t int_x, int_y, int_z, int_w;
		};
	};

	void* operator new(size_t size)
	{
		return malloc_align(size, 16);
	};
	void operator delete(void *p)
	{
		free_align(p);
	}
	Vec4Base() { };
	Vec4Base(const T x_) :x(x_) { };
	Vec4Base(const T x_, const T y_) :x(x_), y(y_) { };
	Vec4Base(const T x_, const T y_, const T z_) :x(x_), y(y_), z(z_) { };
	Vec4Base(const T x_, const T y_, const T z_, const T w_) :x(x_), y(y_), z(z_), w(w_) { };
};

class ALIGN16 Vertex :public Vec4Base<float>
{
public:
	using Vec4Base::x; using Vec4Base::y; using Vec4Base::z; using Vec4Base::w;
	using Vec4Base::int_x; using Vec4Base::int_y; using Vec4Base::int_z; using Vec4Base::int_w;
	Vertex() { }
	Vertex(const bool setZero)
	{
	#ifdef USE_SSE2
		float_dat = _mm_setzero_ps();
	#else
		x = y = z = 0;
	#endif
	}
	Vertex(const float x_, const float y_, const float z_, const float w_ = 0) :Vec4Base(x_, y_, z_, w_) { };
	Vertex(const float x_, const float y_, const float z_, const int32_t w_) :Vec4Base(x_, y_, z_) { int_w = w_; };
	Vertex(const __m128& dat) { float_dat = dat; };
	operator float*() const { return (float *)&x; };
	operator const __m128&() const { return float_dat; };
	operator VertexI&() const { return *(VertexI*)this; };
	float& operator[](uint32_t idx)
	{
		return ((float*)&x)[idx];
	}

	float length() const
	{
	#ifdef USE_SSE4
		return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(float_dat, float_dat, 0b01110001)));
	#else
		return sqrt(length_sqr());
	#endif
	}
	float length_sqr() const
	{
		return operator%(*this);
	}
	Vertex &norm()
	{
	#ifdef USE_SSE4
		const __m128 ans = _mm_sqrt_ps(_mm_dp_ps(float_dat, float_dat, 0b01110111));
		return *this = _mm_div_ps(float_dat, ans);
	#else
		return operator/=(length());
	#endif
	}
	Vertex &do_sqrt()
	{
	#ifdef USE_SSE2
		float_dat = _mm_sqrt_ps(float_dat);
	#else
		x = sqrt(x), y = sqrt(y), z = sqrt(z), w = sqrt(w);
	#endif
		return *this;
	}
	Vertex muladd(const float &n, const Vertex &v) const;
	Vertex mixmul(const Vertex &v) const;

	Vertex operator+(const Vertex &v) const
	{
	#ifdef USE_SSE2
		return _mm_add_ps(float_dat, v);
	#else
		return Vertex(x + v.x, y + v.y, z + v.z);
	#endif
	}
	Vertex &operator+=(const Vertex &right)
	{
	#ifdef USE_SSE2
		return *this = _mm_add_ps(float_dat, right);
	#else
		x += right.x, y += right.y, z += right.z;
		return *this;
	#endif
	}
	Vertex operator-(const Vertex &v) const
	{
	#ifdef USE_SSE2
		return _mm_sub_ps(float_dat, v);
	#else
		return Vertex(x - v.x, y - v.y, z - v.z);
	#endif
	}
	Vertex &operator-=(const Vertex &right)
	{
	#ifdef USE_SSE2
		return *this = _mm_sub_ps(float_dat, right);
	#else
		x += right.x, y += right.y, z += right.z;
		return *this;
	#endif
	}
	Vertex operator*(const float &n) const
	{
	#ifdef USE_SSE2
		return _mm_mul_ps(float_dat, _mm_set1_ps(n));
	#else
		return Vertex(x * n, y * n, z * n);
	#endif
	}
	Vertex &operator*=(const float &right)
	{
	#ifdef USE_SSE2
		return *this = _mm_mul_ps(float_dat, _mm_set1_ps(right));
	#else
		x *= right, y *= right, z *= right;
		return *this;
	#endif
	}
	Vertex operator/(const float &n) const
	{
		return operator*(1 / n);
	}
	Vertex &operator/=(const float &right)
	{
		return operator*=(1 / right);
	}
	Vertex operator*(const Vertex &v) const
	{
	#ifdef USE_SSE2
		const __m128 t1 = _mm_shuffle_ps(float_dat, float_dat, _MM_SHUFFLE(3, 0, 2, 1))/*y,z,x,w*/,
			t2 = _mm_shuffle_ps(v.float_dat, v.float_dat, _MM_SHUFFLE(3, 1, 0, 2))/*v.z,v.x,v.y,v.w*/,
			t3 = _mm_shuffle_ps(float_dat, float_dat, _MM_SHUFFLE(3, 1, 0, 2))/*z,x,y,w*/,
			t4 = _mm_shuffle_ps(v.float_dat, v.float_dat, _MM_SHUFFLE(3, 0, 2, 1))/*v.y,v.z,v.x,v.w*/;
		return _mm_sub_ps(_mm_mul_ps(t1, t2), _mm_mul_ps(t3, t4));
	#else
		float a, b, c;
		a = y*v.z - z*v.y;
		b = z*v.x - x*v.z;
		c = x*v.y - y*v.x;
		return Vertex(a, b, c);
	#endif
	}
	float operator%(const Vertex &v) const//点积
	{
	#ifdef USE_SSE4
		return _mm_cvtss_f32(_mm_dp_ps(float_dat, v.float_dat, 0b01110001));
	#else
		return x*v.x + y*v.y + z*v.z;
	#endif
	}
};

class ALIGN16 VertexI :public Vec4Base<int>
{
public:
	using Vec4Base::x; using Vec4Base::y; using Vec4Base::z; using Vec4Base::w;
	using Vec4Base::float_x; using Vec4Base::float_y; using Vec4Base::float_z; using Vec4Base::float_w;
	VertexI()
	{
	#ifdef USE_SSE2
		int_dat = _mm_setzero_si128();
	#else
		x = y = z = 0;
	#endif
	}
	VertexI(const __m128i &dat) { int_dat = dat; };
	VertexI(const int x_, const int y_, const int z_, const int w_ = 0) :Vec4Base(x_, y_, z_, w_) { };
	operator int*() const { return (int *)&x; };
	operator const __m128i&() const { return int_dat; };
	operator Vertex&() const { return *(Vertex*)this; };
	int& operator[](uint32_t idx)
	{
		return data[idx];
	}
};

}