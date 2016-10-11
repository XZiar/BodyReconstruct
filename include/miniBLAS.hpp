#pragma once

#include <cstdint>
#include <cmath>
#include <vector>

#if defined(__GNUC__)
#   include <x86intrin.h>
#   define _mm256_set_m128(/* __m128 */ hi, /* __m128 */ lo)  _mm256_insertf128_ps(_mm256_castps128_ps256(lo), (hi), 0x1)
#   define _mm256_set_m128d(/* __m128d */ hi, /* __m128d */ lo)  _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), (hi), 0x1)
#   define _mm256_set_m128i(/* __m128i */ hi, /* __m128i */ lo)  _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 0x1)
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

inline void MatrixTranspose4x4(const __m256 l1, const __m256 l2, __m256& o1, __m256& o2)
{
	const __m256 n1 = _mm256_permute_ps(l1, _MM_SHUFFLE(3, 1, 2, 0))/*x1,z1,y1,w1;x2,z2,y2,w2*/;
	const __m256 n2 = _mm256_permute_ps(l2, _MM_SHUFFLE(3, 1, 2, 0))/*x3,z3,y3,w3;x4,z4,y4,w4*/;
	const __m256 t1 = _mm256_unpacklo_ps(n1, n2)/*x1,x3,z1,z3;x2,x4,z2,z4*/;
	const __m256 t2 = _mm256_unpackhi_ps(n1, n2)/*y1,y3,w1,w3;y2,y4,w2,w4*/;
	const __m256 t3 = _mm256_permute2f128_ps(t1, t2, 0b00100000)/*x1,x3,z1,z3;y1,y3,w1,w3*/;
	const __m256 t4 = _mm256_permute2f128_ps(t1, t2, 0b00110001)/*x2,x4,z2,z4;y2,y4,w2,w4*/;
	o1 = _mm256_unpacklo_ps(t3, t4)/*x1,x2,x3,x4;y1,y2,y3,y4*/;
	o2 = _mm256_unpackhi_ps(t3, t4)/*z1,z2,z3,z4lw1,w2,w3,w4*/;
}

inline __m128 cross_product(const __m128 a, const __m128 b)
{
	const __m128 t1 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1))/*y,z,x,w*/,
		t2 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2))/*v.z,v.x,v.y,v.w*/,
		t3 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2))/*z,x,y,w*/,
		t4 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1))/*v.y,v.z,v.x,v.w*/;
	return _mm_sub_ps(_mm_mul_ps(t1, t2), _mm_mul_ps(t3, t4));
}

inline __m128 Mat3x3_Mul_Vec3(const __m128 m0, const __m128 m1, const __m128 m2, const __m128 v)
{
	return _mm_blend_ps(_mm_blend_ps(_mm_dp_ps(m0, v, 0x77), _mm_dp_ps(m1, v, 0x77), 0b010), _mm_dp_ps(m2, v, 0x77), 0b100);
}

class ALIGN16 Vertex;
class ALIGN16 VertexI;

static const uint32_t Vec4Align = 32;
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

	Vec4Base() { };
	Vec4Base(const T x_) :x(x_) { };
	Vec4Base(const T x_, const T y_) :x(x_), y(y_) { };
	Vec4Base(const T x_, const T y_, const T z_) :x(x_), y(y_), z(z_) { };
	Vec4Base(const T x_, const T y_, const T z_, const T w_) :x(x_), y(y_), z(z_), w(w_) { };

public:
	void* operator new(size_t size)
	{
		return malloc_align(size, Vec4Align);
	};
	void operator delete(void *p)
	{
		free_align(p);
	}
	void* operator new[](size_t size)
	{
		return malloc_align(size, Vec4Align);
	};
	void operator delete[](void *p)
	{
		free_align(p);
	}
};

template<class T>
struct AlignAllocator : std::allocator<T>
{
public:
	typedef T value_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T& reference;
	typedef const T& const_reference;

	template<class U>
	struct rebind
	{
		typedef AlignAllocator<U> other;
	};

	T* allocate(size_t n, const T *hint = 0)
	{
		T* ptr = new T[n];
		//printf("allocate at %llx\n", ptr);
		return ptr;
	}
	void deallocate(T *p, size_t n)
	{
		delete[] p;
	}
	size_t max_size() const noexcept
	{
		return ((size_t)(-1) / sizeof(T));
	}
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
	Vertex(const float *ptr) :Vec4Base(ptr[0], ptr[1], ptr[2]) { };
	Vertex(const float x_, const float y_, const float z_, const int32_t w_) :Vec4Base(x_, y_, z_) { int_w = w_; };
	Vertex(const __m128& dat) { float_dat = dat; };
	operator float*() const { return (float *)&x; };
	operator const __m128&() const { return float_dat; };
	operator VertexI&() const { return *(VertexI*)this; };
	float& operator[](uint32_t idx)
	{
		return ((float*)&x)[idx];
	};
	void assign(const float x_, const float y_, const float z_, const float w_ = 0) 
	{
		x = x_, y = y_, z = z_, w = w_;
	};
	void assign(const float *ptr)
	{
		float_dat = _mm_loadu_ps(ptr);
	};
	void assign(const __m128& dat)
	{
		_mm_store_ps(&x, dat);
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
using VertexVec = std::vector<miniBLAS::Vertex, AlignAllocator<miniBLAS::Vertex>>;

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
