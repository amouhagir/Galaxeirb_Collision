#include "cuda.h"
#include "header.h"


__global__ void kernel_update_acc( particule_t *p, vector_t *acc, int size ) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i=0;
	float dx,dy,dz,d,fact;
	if ( j < size ) {
		acc[i].x = 0.0f;
		acc[i].y = 0.0f;
		acc[i].z = 0.0f;

		for (i = 0; i < size; ++i)
		{

				dx = p[i].x-p[j].x;
				dy = p[i].y-p[j].y;
				dz = p[i].z-p[j].z;

				d  = dx*dx+dy*dy+dz*dz;
				if ( d < 1.0 ) d = 1.0;
				fact=p[i].m/(d*sqrtf(d));
				acc[i].x += dx*fact;
				acc[i].y += dy*fact;
				acc[i].z += dz*fact;

		}

	}
}

void update_acc( int nblocks, int nthreads, particule_t *p, vector_t *acc, int size) {
	kernel_update_acc<<<nblocks, nthreads>>>( p, acc, size);
}
