#include "cuda.h"
#include "header.h"


__global__ void kernel_update_pos( particule_t *p, vector_t *acc, int size ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i < size)
	{
		p[i].vx += acc[i].x*MASSFACTOR*DAMP;
		p[i].vy += acc[i].y*MASSFACTOR*DAMP;
		p[i].vz += acc[i].z*MASSFACTOR*DAMP;
		p[i].x += (p[i].vx)*DT;
		p[i].y += (p[i].vy)*DT;
		p[i].z += (p[i].vz)*DT;
	}

}

// 1. utilisation des builtins -> 600 fps
// float3 (x, y, z)
// float4 (x, y, z, w)
// x, y, z, vx, vy, vz, m
// float3, float4

// 2. utilisation de la mémoire partagée -> 1200 fps
// mémoire partagée entre thread d'un bloc
// (NVIDIA N-BODY GPU GEM)

// 3. utilisation de la pinned memory -> 1400-1500 fps

__global__ void kernel_update_acc( particule_t *p, vector_t *acc, int size ) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	//int index = blockIdx.x * blockDim.x + threadIdx.x;
	//int stride = gridDim.x * blockDim.x;
	int i=0;

	float dx,dy,dz,d,fact;
	if ( j < size ) {
	//for(j=index;j<size; j+= stride){
		acc[j].x = 0.0f;
		acc[j].y = 0.0f;
		acc[j].z = 0.0f;
		float xj = p[j].x;
		float yj = p[j].y;
		float zj = p[j].z;


		for (i = 0; i < size; ++i)
		{

			dx = p[i].x-xj;
			dy = p[i].y-yj;
			dz = p[i].z-zj;

			d  = dx*dx+dy*dy+dz*dz;
			if ( d < 1.0 ) d = 1.0;
			fact=p[i].m/(d*sqrtf(d));
			acc[j].x += dx*fact;
			acc[j].y += dy*fact;
			acc[j].z += dz*fact;

		}

	}
}

void update_acc( int nblocks, int nthreads, particule_t *p, vector_t *acc, int size) {
	kernel_update_acc<<<nblocks, nthreads>>>( p, acc, size);

}



void update_position( int nblocks, int nthreads, particule_t *p, vector_t *acc, int size) {
	kernel_update_acc<<<nblocks, nthreads>>>( p, acc, size);
	//cudaDeviceSynchronize();
	kernel_update_pos<<<nblocks, nthreads>>>( p, acc, size);
}
