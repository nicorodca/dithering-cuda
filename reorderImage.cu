__global__ void reorderImage(const float *originalImage, float *orderedImage, int m, int n){
	
	float value=0.0;
	int label=0;
	int pbr=0;
	int posInGroup=0;
	for(int globalTidY=threadIdx.y+blockIdx.y*blockDim.y;globalTidY<m;globalTidY+=blockDim.y*gridDim.y){
		for(int globalTidX=threadIdx.x+blockIdx.x*blockDim.x;globalTidX<n;globalTidX+=blockDim.x*gridDim.x){
			value=originalImage[n*globalTidY+globalTidX];
			label=globalTidX+2*globalTidY+1;
			pbr=(label<=n)?0: (label-n)/2 + (label-n)%2;
		
			posInGroup=globalTidY-pbr;
			
			
			orderedImage[min(m,n/2 + n%2)*(label-1)+posInGroup]=value;//globalTidX+2*globalTidY;
			
		}
	}

}
