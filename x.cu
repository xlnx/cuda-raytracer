#include <iostream>
#include <cstdio>

struct X
{
	virtual ~X() = default;
	virtual void f( void * ) = 0;
};

template <typename T>
__global__ void glob( T *p )
{
	new ( p ) T();
}

template <typename T>
struct B: X
{
	void f(void *a) override
	{
		auto p = static_cast<T *>( a );
		glob<T><<<1, 1>>>( p );
		new (p) T();
	}
};

struct A: B<A>
{
	__host__ __device__ A() = default;
};

int main()
{
	A a;
	a.f(&a);
}
