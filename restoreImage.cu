__global__ void restoreImage(const float *orderedImage, float *restoredImage, int m, int n, int x, int y){
	float value=0.0;
	int label=0;
	int pbr=0;
	int posI=0;
	int posJ=0;
	for(int globalTidY=threadIdx.y+blockIdx.y*blockDim.y;globalTidY<y;globalTidY+=blockDim.y*gridDim.y){
		for(int globalTidX=threadIdx.x+blockIdx.x*blockDim.x;globalTidX<x;globalTidX+=blockDim.x*gridDim.x){
			value=orderedImage[x*globalTidY+globalTidX];

			if(!isnan(value)){
				label=globalTidY+1;
				pbr=(label<=n)?0: (label-n)/2 + (label-n)%2;
				posI=globalTidX+pbr;
				posJ=label-1-2*posI;
				restoredImage[posI*n+posJ]=value;
			}			
		}
	}
}
