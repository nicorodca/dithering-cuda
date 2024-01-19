__global__ void dither(const float* input,float *error,float *output, int m, int n, int iter,int offset){
	float value=0.0f;
	float e=0.0f;
	float eAux=0.0f;
	int label=0;
	for(int globalTidX=threadIdx.x+blockIdx.x*blockDim.x;globalTidX<offset;globalTidX+=blockDim.x*gridDim.x){
		value=input[iter*offset+globalTidX];
		e=0.0f;
		label=iter+1;
		if (!isnan(value)){
			if(label<=n){
				if((iter-1)>-1){
					eAux=error[(iter-1)*offset+globalTidX];
					(!isnan(eAux))?e+=eAux*0.4375:e+=0.0f;		
					if((globalTidX-1)>-1){
						eAux=error[(iter-1)*offset+(globalTidX-1)];
						(!isnan(eAux))?e+=eAux*0.1875:e+=0.0f;		
					}
				}
				if((iter-2)>-1){
					if((globalTidX-1)>-1){
						eAux=error[(iter-2)*offset+(globalTidX-1)];
						(!isnan(eAux))?e+=eAux*0.3125:e+=0.0f;		
					}
				}
				if((iter-3)>-1){
					if((globalTidX-1)>-1){
						eAux=error[(iter-3)*offset+(globalTidX-1)];
						(!isnan(eAux))?e+=eAux*0.0625:e+=0.0f;		
					}
				}
			}
			else{
				if((n-label)%2){
					if((iter-1)>-1){
						eAux=error[(iter-1)*offset+globalTidX];
						(!isnan(eAux))?e+=eAux*0.1875:e+=0.0f;		
						if((globalTidX+1)<offset){
							eAux=error[(iter-1)*offset+(globalTidX+1)];
							(!isnan(eAux))?e+=eAux*0.4375:e+=0.0f;		
						}
					}
				}
				else{
					if((iter-1)>-1){
						eAux=error[(iter-1)*offset+globalTidX];
						(!isnan(eAux))?e+=eAux*0.4375:e+=0.0f;		
						if((globalTidX-1)>-1){
							eAux=error[(iter-1)*offset+(globalTidX-1)];
							(!isnan(eAux))?e+=eAux*0.1875:e+=0.0f;		
						}
					}
				}
				if((iter-2)>-1){
					eAux=error[(iter-2)*offset+(globalTidX)];
						(!isnan(eAux))?e+=eAux*0.3125:e+=0.0f;		
				}
				if((iter-3)>-1){
					eAux=error[(iter-3)*offset+(globalTidX)];
					(!isnan(eAux))?e+=eAux*0.0625:e+=0.0f;		
				}	

			}
			value+=e;
			error[iter*offset+globalTidX]=value-round(value);
			output[iter*offset+globalTidX]=round(value);
		}
	}
}
