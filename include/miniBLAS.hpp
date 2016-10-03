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



namespace miniBLAS
{

class ALIGN16 Vertex
{
public:
	union
	{
		__m128 dat;
		struct
		{
			float x, y, z, w;
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

	Vertex()
	{
	#ifdef USE_SSE2
		dat = _mm_setzero_ps();
	#else
		x = y = z = 0;
	#endif
	}
	Vertex(const __m128 &idat) :dat(idat) { };
	Vertex(const float ix, const float iy, const float iz, const float iw = 0) :x(ix), y(iy), z(iz), w(iw) { };
	Vertex(const float ix, const float iy, const float iz, const int32_t iw) :x(ix), y(iy), z(iz) { int_w = iw; };
	operator float*() const { return (float *)&x; };
	operator __m128() const { return dat; };
	float& operator[](uint32_t idx)
	{
		return ((float*)&x)[idx];
	}

	float length() const
	{
	#ifdef USE_SSE4
		return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(dat, dat, 0b01110001)));
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
		const __m128 ans = _mm_sqrt_ps(_mm_dp_ps(dat, dat, 0b01110111));
		return *this = _mm_div_ps(dat, ans);
	#else
		return operator/=(length());
	#endif
	}
	Vertex &do_sqrt()
	{
	#ifdef USE_SSE2
		dat = _mm_sqrt_ps(dat);
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
		return _mm_add_ps(dat, v);
	#else
		return Vertex(x + v.x, y + v.y, z + v.z);
	#endif
	}
	Vertex &operator+=(const Vertex &right)
	{
	#ifdef USE_SSE2
		return *this = _mm_add_ps(dat, right);
	#else
		x += right.x, y += right.y, z += right.z;
		return *this;
	#endif
	}
	Vertex operator-(const Vertex &v) const
	{
	#ifdef USE_SSE2
		return _mm_sub_ps(dat, v);
	#else
		return Vertex(x - v.x, y - v.y, z - v.z);
	#endif
	}
	Vertex &operator-=(const Vertex &right)
	{
	#ifdef USE_SSE2
		return *this = _mm_sub_ps(dat, right);
	#else
		x += right.x, y += right.y, z += right.z;
		return *this;
	#endif
	}
	Vertex operator*(const float &n) const
	{
	#ifdef USE_SSE2
		return _mm_mul_ps(dat, _mm_set1_ps(n));
	#else
		return Vertex(x * n, y * n, z * n);
	#endif
	}
	Vertex &operator*=(const float &right)
	{
	#ifdef USE_SSE2
		return *this = _mm_mul_ps(dat, _mm_set1_ps(right));
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
		const __m128 t1 = _mm_shuffle_ps(dat, dat, _MM_SHUFFLE(3, 0, 2, 1))/*y,z,x,w*/,
			t2 = _mm_shuffle_ps(v.dat, v.dat, _MM_SHUFFLE(3, 1, 0, 2))/*v.z,v.x,v.y,v.w*/,
			t3 = _mm_shuffle_ps(dat, dat, _MM_SHUFFLE(3, 1, 0, 2))/*z,x,y,w*/,
			t4 = _mm_shuffle_ps(v.dat, v.dat, _MM_SHUFFLE(3, 0, 2, 1))/*v.y,v.z,v.x,v.w*/;
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
		return _mm_cvtss_f32(_mm_dp_ps(dat, v.dat, 0b01110001));
	#else
		return x*v.x + y*v.y + z*v.z;
	#endif
	}
};

class ALIGN16 VertexI
{
public:
	union
	{
		__m128i dat;
		struct
		{
			int x, y, z, w;
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

	VertexI()
	{
	#ifdef USE_SSE2
		dat = _mm_setzero_si128();
	#else
		x = y = z = 0;
	#endif
	}
	VertexI(const __m128i &idat) :dat(idat) { };
	VertexI(const int ix, const int iy, const int iz, const int iw = 0) :x(ix), y(iy), z(iz), w(iw) { };
	operator int*() const { return (int *)&x; };
	operator __m128i() const { return dat; };
	int& operator[](uint32_t idx)
	{
		return ((int*)&x)[idx];
	}
};

}