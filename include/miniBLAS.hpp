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

inline void sumVec(__m128 *ptr, const int32_t cnt, const uint32_t step = 1)
{
	int32_t a = 0;
	const int32_t off1 = 1 << step, off2 = 2 << step;
	const int32_t r0 = 1 << (step - 1), r1 = off1 + r0, r2 = off2 + r0;
	const int32_t adder = 6 * r0;
	for (int32_t times = (cnt + r0 - 1) / adder; times--; a += adder)
	{
		ptr[a + 0] = _mm_add_ps(ptr[a + 0], ptr[a + r0]);
		ptr[a + off1] = _mm_add_ps(ptr[a + off1], ptr[a + r1]);
		ptr[a + off2] = _mm_add_ps(ptr[a + off2], ptr[a + r2]);
	}
	switch (((cnt - a) + r0 - 1) / r0)
	{
	case 5:
		ptr[a + 0] = _mm_add_ps(ptr[a + 0], ptr[a + off2]);
	case 4:
		ptr[a + off1] = _mm_add_ps(ptr[a + off1], ptr[a + r1]);
	case 3:
		ptr[a + 0] = _mm_add_ps(ptr[a + 0], ptr[a + off1]);
	case 2:
		ptr[a + 0] = _mm_add_ps(ptr[a + 0], ptr[a + r0]);
	case 1:
		a++;
		break;
	case 0:
		a = cnt - step;
	default://wrong
		printf("seems wrong here");
		break;
	}
	if (a <= 1)
		return;
	sumVec(ptr, a, step + 1);
}

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

inline __m256 Mat3x3_Mul2_Vec3(const __m128 ma0, const __m128 ma1, const __m128 ma2, const __m128 va,
	const __m128 mb0, const __m128 mb1, const __m128 mb2, const __m128 vb)
{
	const __m256 v256 = _mm256_set_m128(va, vb);
	return _mm256_blend_ps
	(
		_mm256_blend_ps(
			_mm256_dp_ps(_mm256_set_m128(ma0, mb0), v256, 0x77),
			_mm256_dp_ps(_mm256_set_m128(ma1, mb1), v256, 0x77),
			0b00100010),
		_mm256_dp_ps(_mm256_set_m128(ma2, mb2), v256, 0x77),
		0b01000100
	);
}


/*make struct's heap allocation align to N bytes boundary*/
template<uint32_t N>
struct AlignBase
{
	void* operator new(size_t size)
	{
		return malloc_align(size, N);
	};
	void operator delete(void *p)
	{
		free_align(p);
	}
	void* operator new[](size_t size)
	{
		return malloc_align(size, N);
	};
	void operator delete[](void *p)
	{
		free_align(p);
	}
};

/*minimun allocator for vector, use T's operator new to alloc space. Should be used with AlignBase*/
/*
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
		T *ptr = (T*)T::operator new(n * sizeof(T));
		return ptr;
	}
	void deallocate(T *p, size_t n)
	{
		T::operator delete(p);
	}
	size_t max_size() const noexcept
	{
		return ((size_t)(-1) / sizeof(T));
	}
};*/

template <class T>
struct AlignAllocator
{
	typedef T value_type;
	AlignAllocator() noexcept { }
	template<class U> AlignAllocator(const AlignAllocator<U>&) noexcept { }

	template<class U> bool operator==(const AlignAllocator<U>&) const noexcept
	{
		return true;
	}
	template<class U> bool operator!=(const AlignAllocator<U>&) const noexcept
	{
		return false;
	}
	T* allocate(const size_t n) const
	{
		T *ptr = (T*)T::operator new(n * sizeof(T));
		return ptr;
	}
	void deallocate(T* const p, size_t) const noexcept
	{
		T::operator delete(p);
	}
};

class ALIGN16 Vertex;
class ALIGN16 VertexI;

/*a vector contains 4 element(int32 or float), align to 32 boundary for AVX*/
static constexpr uint32_t Vec4Align = 32;
template<typename T>
class ALIGN16 Vec4Base : public AlignBase<Vec4Align>
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
	template<uint8_t N, typename T2>
	void save(T2 *ptr) const
	{
		static_assert(N <= 4, "there is only 4 elements");
		for (uint8_t a = 0; a < N; ++a)
			ptr[a] = data[a];
	};
	template<uint8_t N, typename T2>
	void load(const T2 *ptr)
	{
		static_assert(N <= 4, "there is only 4 elements");
		for (uint8_t a = 0; a < N; ++a)
			data[a] = ptr[a];
	};
};

/*vecot contains 4 float, while only xyz are considered in some calculation*/
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
	Vertex &do_norm()
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

class ALIGN16 VertexI :public Vec4Base<int32_t>
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
	VertexI(const int32_t x_, const int32_t y_, const int32_t z_, const int32_t w_ = 0) :Vec4Base(x_, y_, z_, w_) { };
	operator int32_t*() const { return (int32_t *)&x; };
	operator const __m128i&() const { return int_dat; };
	operator Vertex&() const { return *(Vertex*)this; };
	int& operator[](uint32_t idx)
	{
		return data[idx];
	}

	void assign(const int32_t x_, const int32_t y_, const int32_t z_, const int32_t w_ = 0)
	{
		x = x_, y = y_, z = z_, w = w_;
	};
	void assign(const int32_t *ptr)
	{
		int_dat = _mm_loadu_si128((__m128i*)ptr);
	};
};
using VertexIVec = std::vector<miniBLAS::VertexI, AlignAllocator<miniBLAS::VertexI>>;

/*4x4matrix contains 4 row of 4elements-vector, align to 32bytes for AVX*/
static constexpr uint32_t SQMat4Align = 32;
template<typename T, typename T2>
class ALIGN32 SQMat4Base : public AlignBase<SQMat4Align>
{
	static_assert(sizeof(T) == 16, "only 16-byte length type(for a row) allowed");
	static_assert(sizeof(T2) == 32, "T2 should be twice of T");
protected:
	union
	{
		__m128 float_row[4];
		__m128i int_row[4];
		T row[4];
		__m256 float_row2[2];
		__m256i int_row2[2];
		T2 row2[2];
		struct
		{
			T x, y, z, w;
		};
		struct
		{
			__m128 float_x, float_y, float_z, float_w;
		};
		struct
		{
			__m128i int_x, int_y, int_z, int_w;
		};
		struct
		{
			T2 xy, zw;
		};
		struct
		{
			__m256 float_xy, float_zw;
		};
		struct
		{
			__m256i int_xy, int_zw;
		};
	};

	SQMat4Base() { };
	SQMat4Base(const T x_) :x(x_) { };
	SQMat4Base(const T x_, const T y_) :x(x_), y(y_) { };
	SQMat4Base(const T x_, const T y_, const T z_) :x(x_), y(y_), z(z_) { };
	SQMat4Base(const T x_, const T y_, const T z_, const T w_) :x(x_), y(y_), z(z_), w(w_) { };
	SQMat4Base(const T2 xy_) :xy(xy_) { };
	SQMat4Base(const T2 xy_, const T z_) :xy(xy_), z(z_) { };
	SQMat4Base(const T2 xy_, const T2 zw_) :xy(xy_), zw(zw_) { };
};

using __m128x4 = __m128[4];
using __m256x2 = __m256[2];
class ALIGN32 SQMat3x3;
/*4x4 matrix*/
class ALIGN32 SQMat4x4 :public SQMat4Base<__m128, __m256>
{
public:
	using SQMat4Base::x; using SQMat4Base::y; using SQMat4Base::z; using SQMat4Base::w;
	SQMat4x4() { }
	SQMat4x4(const bool isIdentity)
	{
		if (isIdentity)
			float_row2[0] = _mm256_set_ps(0, 0, 1, 0, 0, 0, 0, 1), float_row2[1] = _mm256_set_ps(1, 0, 0, 0, 0, 1, 0, 0);
	}
	SQMat4x4(const __m128 x_, const __m128 y_, const __m128 z_, const __m128 w_) :SQMat4Base(x_, y_, z_, w_) { };
	SQMat4x4(const __m256 xy_, const __m256 zw_) :SQMat4Base(xy_, zw_) { };
	SQMat4x4(const __m128 *ptr) :SQMat4Base(ptr[0], ptr[1], ptr[2], ptr[3]) { };
	SQMat4x4(const __m128x4& dat)
	{
		row[0] = dat[0]; row[1] = dat[1]; row[2] = dat[2]; row[3] = dat[3];
	};
	operator __m128*() const { return (__m128 *)&x; };
	operator SQMat3x3&() const { return *(SQMat3x3*)this; };
	__m128& operator[](const uint32_t rowidx)
	{
		return float_row[rowidx];
	};
	float operator()(const uint32_t rowidx, const uint32_t colidx)
	{
		return Vertex(float_row[rowidx])[colidx];
	};
	void assign(const __m128 x_, const __m128 y_, const __m128 z_, const __m128 w_ = _mm_setr_ps(1, 0, 0, 0))
	{
		x = x_, y = y_, z = z_, w = w_;
	};
	void assign(const float *ptr)
	{
		float_row2[0] = _mm256_loadu_ps(ptr);
		float_row2[1] = _mm256_loadu_ps(ptr + 8);
	};

	SQMat4x4 inv()
	{
		SQMat4x4 ret;
		MatrixTranspose4x4(row2[0], row2[1], ret.row2[0], ret.row2[1]);
		return ret;
	}
	SQMat4x4 &do_inv()
	{
		MatrixTranspose4x4(row2[0], row2[1], row2[0], row2[1]);
		return *this;
	}

	__m128 operator*(const __m128 v) const
	{
		const __m256 v256 = _mm256_set_m128(v, v);
		const __m256 ans = _mm256_blend_ps(
			_mm256_dp_ps(row2[0], v256, 0xff),
			_mm256_dp_ps(row2[1], v256, 0xff),
			0b11001100)/*xxzz,yyww*/;
		return _mm_blend_ps(_mm256_castps256_ps128(ans), _mm256_extractf128_ps(ans, 1), 0b1010)/*xyzw*/;
	}
	SQMat4x4 operator*(const float v) const
	{
		const __m256 v256 = _mm256_set1_ps(v);
		return SQMat4x4(_mm256_mul_ps(row2[0], v256), _mm256_mul_ps(row2[1], v256));
	}
	SQMat4x4 operator*(const SQMat4x4& m) const
	{
		__m256 c02, c13;
		{
			const __m256 t1 = _mm256_unpacklo_ps(m.row2[0], m.row2[1])/*x1,x3,y1,y3;x2,x4,y2,y4*/;
			const __m256 t2 = _mm256_unpackhi_ps(m.row2[0], m.row2[1])/*z1,z3,w1,w3;z2,z4,w2,w4*/;
			const __m256 t3 = _mm256_permute2f128_ps(t1, t2, 0b00100000)/*x1,x3,y1,y3;z1,z3,w1,w3*/;
			const __m256 t4 = _mm256_permute2f128_ps(t1, t2, 0b00110001)/*x2,x4,y2,y4;z2,z4,w2,w4*/;
			c02 = _mm256_unpacklo_ps(t3, t4)/*x1,x2,x3,x4;z1,z2,z3,z4*/;
			c13 = _mm256_unpackhi_ps(t3, t4)/*y1,y2,y3,y4;w1,w2,w3,w4*/;
		}
		const __m256 c0 = _mm256_permute2f128_ps(c02, c13, 0x00),
			c1 = _mm256_permute2f128_ps(c02, c13, 0x22),
			c2 = _mm256_permute2f128_ps(c02, c13, 0x11),
			c3 = _mm256_permute2f128_ps(c02, c13, 0x33);
		SQMat4x4 ret;
		const __m256 xy0 = _mm256_dp_ps(xy, c0, 0xf1), xy1 = _mm256_dp_ps(xy, c1, 0xf2), xy2 = _mm256_dp_ps(xy, c2, 0xf4), xy3 = _mm256_dp_ps(xy, c3, 0xf8);
		ret.xy = _mm256_blend_ps(_mm256_blend_ps(xy0, xy1, 0b00100010)/*x0,x1,,;y0,y1,,*/, _mm256_blend_ps(xy2, xy3, 0b10001000)/*,,x2,x3;,,y2,y3*/,
			0b11001100)/*x0,x1,x2,x3;y0,y1,y2,y3*/;
		const __m256 zw0 = _mm256_dp_ps(zw, c0, 0xf1), zw1 = _mm256_dp_ps(zw, c1, 0xf2), zw2 = _mm256_dp_ps(zw, c2, 0xf4), zw3 = _mm256_dp_ps(zw, c3, 0xf8);
		ret.zw = _mm256_blend_ps(_mm256_blend_ps(zw0, zw1, 0b00100010)/*z1,z1,,;w0,w1,,*/, _mm256_blend_ps(zw2, zw3, 0b10001000)/*,,z2,z3;,,w2,w3*/,
			0b11001100)/*z0,z1,z2,z3;w0,w1,w2,w3*/;
		return ret;
	}

	SQMat4x4 operator+(const SQMat4x4& m) const
	{
		return SQMat4x4(_mm256_add_ps(row2[0], m.row2[0]), _mm256_add_ps(row2[1], m.row2[1]));
	}
	SQMat4x4& operator+=(const SQMat4x4& m)
	{
		row2[0] = _mm256_add_ps(row2[0], m.row2[0]); row2[1] = _mm256_add_ps(row2[1], m.row2[1]);
		return *this;
	}
	SQMat4x4 operator-(const SQMat4x4& m) const
	{
		return SQMat4x4(_mm256_sub_ps(row2[0], m.row2[0]), _mm256_sub_ps(row2[1], m.row2[1]));
	}
	SQMat4x4& operator-=(const SQMat4x4& m)
	{
		row2[0] = _mm256_sub_ps(row2[0], m.row2[0]); row2[1] = _mm256_sub_ps(row2[1], m.row2[1]);
		return *this;
	}
};

/*3x3 matrix, row4's value is not promised while calculation. it would be convert into 4x4 matrix while calc with SQMat4x4*/
class ALIGN32 SQMat3x3 :public SQMat4x4
{
public:
	SQMat3x3() { };
	SQMat3x3(const __m128 x_, const __m128 y_, const __m128 z_) :SQMat4x4(x_, y_, z_, _mm_setr_ps(1, 0, 0, 0)) { };

	__m128 operator*(const __m128 v) const
	{
		const __m256 v256 = _mm256_set_m128(v, v);
		const __m256 ans = _mm256_blend_ps(
			_mm256_dp_ps(row2[0], v256, 0x7f),
			_mm256_dp_ps(row2[1], v256, 0x7f),
			0b11001100)/*xxzz,yy??*/;
		return _mm_blend_ps(_mm256_castps256_ps128(ans), _mm256_extractf128_ps(ans, 1), 0b1010)/*xyz?*/;
	}
	SQMat3x3 operator*(const float v) const
	{
		return ((SQMat4x4*)this)->operator*(v);
	}
};

inline __m256 Mat3x3_Mul2_Vec3(const SQMat3x3& ma, const __m128 va, const SQMat3x3& mb, const __m128 vb)
{
	return Mat3x3_Mul2_Vec3(ma[0], ma[1], ma[2], va, mb[0], mb[1], mb[2], vb);
}

}
