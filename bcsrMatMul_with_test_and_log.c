#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include<stdbool.h>
#include <assert.h> 

#define dim 60
#define blocksize 5
#define num_block_val blocksize*blocksize
#define random_dividend 1000
#define uniq_val_in_half 10
#define uniq_val_in_symm 15


struct keyParameters {
	int num_block_row;
  	int num_block_row_except_root;
  	int excess;
  	int per_process_num_block_total;
  	int per_process_num_block_col_prev;
  	int per_process_num_block_col_only;
  	int per_process_num_block_col_later;
  	int per_process_num_block_prev;
	int per_process_num_block_only;
	int per_process_num_block_later;
  	int per_process_num_block_row;
  	int per_process_A_size;
};


bool checkRestrictions(int root, int rank, int size);
void SpMVinBCSR(int per_process_num_block_row, double* A_vals, double* vec, double* res, int* bcsr_rows_idx, int* bcsr_cols );
void NineBandSymmBCSR(int rank, int size, int root, double *A, int per_process_num_block_row, int A_length);
void generateVector(int root, int rank, int size, double* vec, struct keyParameters keys);



int main(int argc, char *argv[])
{
	int size, rank, i, j, k, root = 0, ierr, res_length, res_total_length;
	double start, stop, wtime;
	double *vec, *res, *res_total;
	double *A_vals;
	int *bcsr_rows_idx, *bcsr_cols;
	struct keyParameters keys;
	keys.num_block_row = dim / blocksize;
	keys.per_process_num_block_col_prev = 2;
  	keys.per_process_num_block_col_only = 0;
  	keys.per_process_num_block_col_later = 2;
  	keys.per_process_num_block_prev = 1;
	keys.per_process_num_block_later = 1;
  	
	
	bool test = true;
	bool log = true;
	int check = 0;


	/* Step 0: MPI Initialization */
	ierr = MPI_Init(&argc,&argv);
	if(ierr !=0){
		exit(1);
	}

  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  	MPI_Comm_size(MPI_COMM_WORLD, &size);

  	/* Step 1: Restrictions */
  	bool flag = checkRestrictions(root, rank, size);
  	if (flag == false)
  	{
  		MPI_Finalize();
		return EXIT_SUCCESS;
  	}
  	

  	

  	/* Step 2: Initialization (4 Substeps) */  	 	
 	// Step 2a: num_block_row distribution
  	keys.num_block_row_except_root = keys.num_block_row - 2;
  	// root will handle first and last blocks and gatherv at the end of the process. so size-1 processors will work
  	keys.excess = keys.num_block_row_except_root % (size-1);  
  	keys.per_process_num_block_row = (keys.num_block_row_except_root - keys.excess) / (size-1);
  	if (keys.excess == 0){
  		if(log && rank ==root){
  			printf("BALANCED, Each take %d\n", keys.per_process_num_block_row);
  		}
  	}
  	else{
  		if(rank<=keys.excess && 0 < rank){
			keys.per_process_num_block_row++;
		}
  		if(log && rank ==root){
  			printf("UNBALANCED, if 0<rank<=%d, then Each take  %d; ow %d\n",keys.excess,keys.per_process_num_block_row+1, keys.per_process_num_block_row);
  		}	
  	}
  	// root's per_process_num_block_row value is 2 and fixed!
  	if(rank == root){
  		keys.per_process_num_block_row = 2;
  	}

  	MPI_Reduce(&keys.per_process_num_block_row, &check, 1, MPI_INT,  MPI_SUM, 0, MPI_COMM_WORLD);
  	if(rank == root){
  		assert(check == keys.num_block_row);
  		if(log){
  			printf("1-SUCCESS\n");
  		}
  	} 
	// Step 2b: per_process_num_block_col calculations
	if (rank != root){
		keys.per_process_num_block_col_only = keys.per_process_num_block_row -2;
		assert(keys.per_process_num_block_col_only >= 0 );
	}
	// Step 2c: per_process_num_block calculations
	if(rank == root){
		keys.per_process_num_block_total = 4;} 
	else{
		keys.per_process_num_block_total = keys.per_process_num_block_row*3;}
	keys.per_process_num_block_only = keys.per_process_num_block_total - (keys.per_process_num_block_prev + keys.per_process_num_block_later);
	// Step 2d: total number of values in A array
	keys.per_process_A_size =keys.per_process_num_block_total * num_block_val;
	if(log & rank ==root){
		printf("2-SUCCESS\n");
	}


  	/* Step 3: Vector generation */
  	generateVector(root, rank, size, vec, keys);
  	if(log & rank ==root){
		printf("3-SUCCESS\n");
	}
  

  	// Step 4: BSCR form (3 Substeps)
  	// Step 4a: A values
  	A_vals = calloc(keys.per_process_A_size, sizeof(double));
  	NineBandSymmBCSR(rank, size, root, A_vals, keys.per_process_num_block_row, keys.per_process_A_size);
  	if (test && rank == 1 ){
	    FILE *fptr = fopen("A1.txt", "w");
		if (fptr == NULL)
		{
		    printf("Could not open file");
		    return 0;
		}
		else{
		  	for (i = 0; i<keys.per_process_A_size; i++){
		  		if (i % 5 == 0){fprintf(fptr,"\n");}
				fprintf(fptr,"%f ",A_vals[i]);
			}
		} 
	}
	if (test && rank == 1 ){
	    FILE *fptr = fopen("vec1.txt", "w");
		if (fptr == NULL)
		{
		    printf("Could not open file");
		    return 0;
		}
		else{
			int vec_length = (keys.per_process_num_block_col_prev + keys.per_process_num_block_col_only +keys.per_process_num_block_col_later ) * blocksize;
		  	for (i = 0; i<vec_length; i++){
		  		if (i % 5 == 0){fprintf(fptr,"\n");}
				fprintf(fptr,"%f ",vec[i]);
			}
		} 		
  	}
  	// Step 4b: Initialization for cols_idx and rows_ptr
  	int row_idx_length = (keys.per_process_num_block_row + 1);
  	int col_lenght = keys.per_process_num_block_total;
  	bcsr_rows_idx = calloc(row_idx_length, sizeof(int));  	
  	bcsr_cols = calloc(col_lenght, sizeof(int));
  	// Step 4c: Generate other two arrays for BCSR
  	int out = 0;
  	if (rank == root){ 		
  		for (i = 0; i<row_idx_length; i++, out+=2){
  			bcsr_rows_idx[i] = out;
  		}
  		for(i = 0; i<col_lenght; i++){
  			bcsr_cols[i] =i;
  		}	
  	}
   	else{
  		for (i = 0; i<row_idx_length; i++, out+=3){
  			bcsr_rows_idx[i] = out;
  		}
  		bcsr_rows_idx[row_idx_length-1] = keys.per_process_num_block_total-1;
  		//for (i = 0; i<row_idx_length; i++){
  		//	if(rank == 1){printf("Check bcsr_rows_idx  %d\n", bcsr_rows_idx[i]);}
  		//}
   		j = 0;
  		for(i = 0; i<col_lenght; i+=3, j++){
  			bcsr_cols[i]   =j;
  			bcsr_cols[i+1] =j+1;
  			bcsr_cols[i+2] =j+2;
  		}
  	}
	if(log & rank ==root){
		printf("4-SUCCESS\n");
	}

  	// Step 5: Allocate memory for result vector
  	res_length = (keys.per_process_num_block_row) * blocksize;
  	res_total_length = dim;
  	
  	res = calloc(res_length, sizeof(double));
  	double *res_ptr = res;
  	if (rank == root){
  		res_total = calloc(dim, sizeof(double));
  	}
	if(log & rank ==root){
		printf("5-SUCCESS\n");
	}

	// Step 6: SpMV by using BCSR arrays
	if(rank == root){
		start = MPI_Wtime();
	}	
	SpMVinBCSR(keys.per_process_num_block_row, A_vals, vec, res, bcsr_rows_idx, bcsr_cols);
	if(log & rank ==root){
		printf("6-SUCCESS(only root)\n");
	}

	// A_vals and res pointers are meaningless now!!!!!
	if (test && rank == 1 ){
	    FILE *fptr = fopen("res1.txt", "w");
		if (fptr == NULL)
		{
		    printf("Could not open file");
		    return 0;
		}
		else{
		  	for (i = 0; i<res_length; i++){
		  		if (i % 5 == 0){fprintf(fptr,"\n");}
				fprintf(fptr,"%f ",res_ptr[i]);
			}
		} 		
  	}

  	// Step 7: MPI Gatherv
  	// Step 7a: Initialize
  	int *receive_cnt, *receive_plc;
  	receive_cnt = calloc(size,sizeof(int));
  	receive_plc = calloc(size,sizeof(int));
  	// Step 7b: Fill receive_cnt
  	receive_cnt[0] = 10; // from root
  	int global_info = (keys.num_block_row_except_root - keys.excess) / (size-1);
  	for(i = 1; i< size; i++){
  		if(i<=keys.excess){
  			receive_cnt[i] = (global_info+1) * blocksize;
  		}
  		else{
  			receive_cnt[i] = global_info * blocksize;
  		}
  		if(rank == 9){printf("receive_cnt[%d] = %d\n",i,receive_cnt[i] );}		
  	}

  	// Step 7c: Fill receive_plc
  	receive_plc[0] = 0; // callocdan dolayı zaten 0 olmasını bekleriz ama yine de tanımlayalım
  	for(i = 1 ; i<size; i++){
  		receive_plc[i] = receive_plc[i-1] + receive_cnt[i-1];
  		if(rank == 9){printf("receive_plc[%d] = %d\n",i,receive_plc[i] );}
  	}
  	assert(receive_plc[size-1] + receive_cnt[size -1 ] == dim );

  	// Step 7d: Gatherv
  	MPI_Gatherv(res_ptr,res_length,MPI_DOUBLE,res_total,receive_cnt, receive_plc, MPI_DOUBLE, root, MPI_COMM_WORLD);

	if(log & rank ==root){		
		printf("7-SUCCESS(only root)\n");
	}
	if(log & rank ==root){
		printf("7-SUCCESS\n");
	}
  	if(rank == root){
  		// calculate the wtime
		stop = MPI_Wtime();
		wtime = stop - start;		
		printf("#P\tDim\tWtime\n");
		printf("%d\t%d\t%f\n", size, dim, wtime);
	}

	if (test && rank == root){
	    FILE *fptr = fopen("res_total.txt", "w");
		if (fptr == NULL)
		{
		    printf("Could not open file");
		    return 0;
		}
		else{
		  	for (i = 0; i<res_total_length; i++){
		  		if (i % 5 == 0){fprintf(fptr,"\n");}
				fprintf(fptr,"%f ",res_total[i]);
			}
		} 		
  	}

	// Step 0: MPI Finalize
  	MPI_Finalize();
  	return EXIT_SUCCESS;
}

void SpMVinBCSR(int per_process_num_block_row, double* A_vals, double* vec, double* res, int* bcsr_rows_idx, int* bcsr_cols ){
	double y0,y1,y2,y3,y4,x0,x1,x2,x3,x4;	
	int i,j,k;  	
	for (i = 0; i < per_process_num_block_row; i++, res += blocksize)
	{
		// Each block has impact on 5-row b
		y0 = res[0]; 
		y1 = res[1];
		y2 = res[2];
		y3 = res[3];
		y4 = res[4]; 
		for (j = bcsr_rows_idx[i]; j < bcsr_rows_idx[i+1]; j++, A_vals += num_block_val)
		{
			k = bcsr_cols[j] * blocksize;
			// Each block needs 5-row x 
			x0 = vec[k]; 
			x1 = vec[k+1]; 
			x2 = vec[k+2];
			x3 = vec[k+3];
			x4 = vec[k+4];
			// Multiplication
			y0 += A_vals[0]*x0;
			y0 += A_vals[1]*x1;	
			y0 += A_vals[2]*x2; 
			y0 += A_vals[3]*x3;	
			y0 += A_vals[4]*x4;
			y1 += A_vals[5]*x0; 
			y1 += A_vals[6]*x1; 
			y1 += A_vals[7]*x2; 
			y1 += A_vals[8]*x3; 
			y1 += A_vals[9]*x4; 
			y2 += A_vals[10]*x0; 
			y2 += A_vals[11]*x1; 
			y2 += A_vals[12]*x2; 
			y2 += A_vals[13]*x3; 
			y2 += A_vals[14]*x4; 
			y3 += A_vals[15]*x0;
			y3 += A_vals[16]*x1;
			y3 += A_vals[17]*x2;
			y3 += A_vals[18]*x3;
			y3 += A_vals[19]*x4;
			y4 += A_vals[20]*x0;
			y4 += A_vals[21]*x1;
			y4 += A_vals[22]*x2;
			y4 += A_vals[23]*x3;
			y4 += A_vals[24]*x4;
		}
		res[0]=y0;
		res[1]=y1;
		res[2]=y2;
		res[3]=y3;
		res[4]=y4;			
	}}


void NineBandSymmBCSR(int rank, int size, int root, double *A, int per_process_num_block_row, int A_length){
	/* Pipeline:
	1. Calculate unique values for each processor
	2. Generate these number of unique values
	3. Put unique values into A
	*/
	double *A_uniq_prev, *A_uniq_only, *A_uniq_later;
	int i, idx=0;
	int per_process_num_uniq_total;
	int per_process_num_uniq_prev = uniq_val_in_half;
	int per_process_num_uniq_only;
	int per_process_num_uniq_later = uniq_val_in_half;
	if(rank==root){
		per_process_num_uniq_total = 2*(uniq_val_in_half +uniq_val_in_symm);}
	else{
		per_process_num_uniq_total = per_process_num_block_row * (uniq_val_in_half+uniq_val_in_symm) + per_process_num_uniq_prev;}
	per_process_num_uniq_only = per_process_num_uniq_total - (per_process_num_uniq_prev + per_process_num_uniq_later);
	A_uniq_prev = calloc(per_process_num_uniq_prev, sizeof(double)); // Upper Triang
  	A_uniq_only = calloc(per_process_num_uniq_only, sizeof(double)); // symmetric
  	A_uniq_later = calloc(per_process_num_uniq_later, sizeof(double)); // Lower Triang


	// cyclic relations in previous
	if (rank == root){
		srand(size-1);
	  	for (i = 0; i<per_process_num_uniq_prev; i++){
			A_uniq_prev[i] = rand()%random_dividend;
		}		
	}
	else{
		srand(rank-1);
	  	for (i = 0; i<per_process_num_uniq_prev; i++){
			A_uniq_prev[i] = rand()%random_dividend;
		}
	}

	// only
  	srand(rank*rank);
  	for (i = 0; i<per_process_num_uniq_only; i++){
		A_uniq_only[i] = rand()%random_dividend;
	}

	// later
	srand(rank);
  	for (i = 0; i<per_process_num_uniq_later; i++){
		A_uniq_later[i] = rand()%random_dividend;		
	}

	i =0;
	int off_idx, iter, symmetric_idx;
	if(rank !=root){
		// once prev değerlerini yerleştir
		for(i = 0; i < per_process_num_uniq_prev; i++){
			if(i == 0 || i ==1) {A[i+1] = A_uniq_prev[i];}
			else if(i == 2) {A[7] = A_uniq_prev[i];}
			else if(i == 3) {A[3] = A_uniq_prev[i];}
			else if(i == 4) {A[8] = A_uniq_prev[i];}
			else if(i == 5) {A[13] = A_uniq_prev[i];}
			else if(i == 6) {A[4] = A_uniq_prev[i];}
			else if(i == 7) {A[9] = A_uniq_prev[i];}
			else if(i == 8) {A[14] = A_uniq_prev[i];}
			else if(i == 9) {A[19] = A_uniq_prev[i];}
		}
		assert(i == per_process_num_uniq_prev);	
		// sonra only values
		i =0;
		for(idx = num_block_val; idx < A_length-num_block_val; idx++){
			off_idx = (idx - num_block_val) % 75;
			iter =  (idx - num_block_val) / 75;

			if(off_idx == 0 || off_idx == 6  || off_idx == 12 || off_idx == 18 || off_idx == 24){
				A[idx] = A_uniq_only[i];
				i++;

			}
			else if(off_idx == 1 || off_idx == 2 || off_idx == 3 || off_idx == 4){
				A[idx] = A_uniq_only[i];
				symmetric_idx = (off_idx*5) + num_block_val + (75 * iter);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}	
				i++;
			}			
			else if (off_idx == 7 || off_idx == 13 || off_idx == 19){
				A[idx] = A_uniq_only[i];
				symmetric_idx= (off_idx + 4) + num_block_val + (75 * iter);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}
				i++;
			}
			else if(off_idx == 8 || off_idx == 14){
				A[idx] = A_uniq_only[i];
				symmetric_idx = (off_idx+8)+ num_block_val + (75 * iter);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}		
				i++;
			}
			else if(off_idx == 45){
				A[idx] = A_uniq_only[i];
				symmetric_idx = (off_idx+9)+ num_block_val + (75 * iter);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}	
				i++;
			}
			else if(off_idx == 9){
				A[idx] = A_uniq_only[i];
				symmetric_idx = (off_idx+12)+ num_block_val + (75 * iter);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}	
				i++;
			}
			else if (off_idx == 40 || off_idx == 46){
				A[idx] = A_uniq_only[i];
				symmetric_idx = (off_idx + 13) + num_block_val + (75 * iter);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}	
				i++;
			}
			else if (off_idx == 35 || off_idx == 41 || off_idx == 47){
				A[idx] = A_uniq_only[i];
				symmetric_idx = (off_idx + 17) + num_block_val + (75 * iter);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}	
				i++;
			}
			else if (off_idx == 30 || off_idx == 36 || off_idx == 42 || off_idx == 48){
				A[idx] = A_uniq_only[i];
				symmetric_idx = (off_idx + 21) + num_block_val + (75 * iter);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}	
				i++;
			}
			
		}		
		assert(i == per_process_num_uniq_only);

		// put later values		
		for(i=0; i < per_process_num_uniq_later; i++){
			if(i == 0)      {A[A_length-20] = A_uniq_later[i];}
			else if (i ==1) {A[A_length-15] = A_uniq_later[i];}
			else if(i == 2) {A[A_length-14] = A_uniq_later[i];}
			else if(i == 3) {A[A_length-10] = A_uniq_later[i];}
			else if(i == 4) {A[A_length-9] = A_uniq_later[i];}
			else if(i == 5) {A[A_length-8] = A_uniq_later[i];}
			else if(i == 6) {A[A_length-5] = A_uniq_later[i];}
			else if(i == 7) {A[A_length-4] = A_uniq_later[i];}
			else if(i == 8) {A[A_length-3] = A_uniq_later[i];}
			else if(i == 9) {A[A_length-2] = A_uniq_later[i];}
		}
	}
	else{
		// root
		// only
		for(idx = 0; idx < num_block_val; idx++){
			if(idx == 0 || idx == 6  || idx == 12 || idx == 18 || idx == 24){
				A[idx] = A_uniq_only[i];
				i++;
			}
			else if(idx == 1 || idx == 2 || idx == 3 || idx == 4){
				A[idx] = A_uniq_only[i];
				symmetric_idx = (idx*5);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}	
				i++;
			}			
			else if (idx == 7 || idx == 13 || idx == 19){
				A[idx] = A_uniq_only[i];
				symmetric_idx= (idx + 4);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}	
				i++;
			}
			else if(idx == 8 || idx == 14){
				A[idx] = A_uniq_only[i];
				symmetric_idx = (idx+8);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}				
				i++;
			}	
			else if(idx == 9){
				A[idx] = A_uniq_only[i];
				symmetric_idx = (idx+12);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}	
				i++;
			}
		}
		assert(i == uniq_val_in_symm);

		// put later values
		i = 0;
		for(idx = num_block_val; idx<2*num_block_val ;idx++){
			if(idx == 30||idx == 35||idx == 36||idx == 40||idx == 41||idx == 42||idx == 45||idx == 46||idx == 47||idx == 48){
				A[idx] = A_uniq_later[i];
				i++;
			}
		}
		assert(i == uniq_val_in_half);

		// put previous values
		for(i = 0; i <per_process_num_uniq_prev ;i++){
			if(i == 0) {A[51] = A_uniq_prev[i];}
			else if(i == 1) {A[52] = A_uniq_prev[i];}
			else if(i == 2) {A[57] = A_uniq_prev[i];}
			else if(i == 3) {A[53] = A_uniq_prev[i];}
			else if(i == 4) {A[58] = A_uniq_prev[i];}
			else if(i == 5) {A[63] = A_uniq_prev[i];}
			else if(i == 6) {A[54] = A_uniq_prev[i];}
			else if(i == 7) {A[59] = A_uniq_prev[i];}
			else if(i == 8) {A[64] = A_uniq_prev[i];}
			else if(i == 9) {A[69] = A_uniq_prev[i];}
		}
		assert(i == uniq_val_in_half);

		// put only
		i=uniq_val_in_symm;
		for(idx = 3*num_block_val; idx<4*num_block_val ;idx++){
			iter = 1;
			off_idx = idx % 75;

			if(off_idx == 0 || off_idx == 6  || off_idx == 12 || off_idx == 18 || off_idx == 24){
				A[idx] = A_uniq_only[i];
				i++;
			}
			else if(off_idx == 1 || off_idx == 2 || off_idx == 3 || off_idx == 4){
				A[idx] = A_uniq_only[i];
				symmetric_idx = (off_idx*5)+ (75 * iter);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}	
				i++;
			}			
			else if (off_idx == 7 || off_idx == 13 || off_idx == 19){
				A[idx] = A_uniq_only[i];
				symmetric_idx= (off_idx + 4)+(75 * iter);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}	
				i++;
			}
			else if(off_idx == 8 || off_idx == 14){
				A[idx] = A_uniq_only[i];
				symmetric_idx = (off_idx+8)+ (75 * iter);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}				
				i++;
			}
			else if(off_idx == 9){
				A[idx] = A_uniq_only[i];
				symmetric_idx = (off_idx+12) + (75 * iter);
				if(symmetric_idx<A_length){A[symmetric_idx] = A_uniq_only[i];}	
				i++;
			}

		}
		assert(i == per_process_num_uniq_only);
	}}


bool checkRestrictions(int root, int rank, int size){
	int mustBeZero, num_block_row;
  	mustBeZero = dim % blocksize;
  	num_block_row = dim / blocksize;
  	if(mustBeZero != 0){
  		// Mention about the situation
  		if(rank == root){
  			printf("Matrix dimension should be multiple of blocksize: Matrix dim %d and BlockSize %d\n", dim, blocksize);
  		}
  		return false;

  	}  	
  	else if(num_block_row < 4 || (num_block_row - 2) < (size-1)){
  		// Mention about the situation
  		if(rank == root){
  			printf("Matrix dimension is too small or Num of processor is not enough for parallelization. One/Some of the processor will not do any thing! num_block_row per process should be at least 1\n");
  		}
  		return false;		
  	}
  	else
  		return true;}


void generateVector(int root, int rank, int size, double* vec, struct keyParameters keys)
{
  	// Step 3a: Allocate memory for vector partitions and initialize them with 0
  	double *vec_only, *vec_prev, *vec_later;
  	int i;

  	vec_prev  = calloc(keys.per_process_num_block_col_prev * blocksize, sizeof(double)); // actually it is fixed but for convenience, make them parametrized
  	vec_only  = calloc(keys.per_process_num_block_col_only * blocksize, sizeof(double));
  	vec_later = calloc(keys.per_process_num_block_col_later * blocksize, sizeof(double));
  	int vec_length = (keys.per_process_num_block_col_prev + keys.per_process_num_block_col_only +keys.per_process_num_block_col_later ) * blocksize;	
  	
  	// Step 3b: Initialize with rank-related values 
  	srand(rank*rank);
  	for (i = 0; i<(keys.per_process_num_block_col_only * blocksize); i++){
		vec_only[i] = rand()%random_dividend;

	}
	srand(rank);
  	for (i = 0; i<(keys.per_process_num_block_col_later * blocksize); i++){
		vec_later[i] = rand()%random_dividend;
	}
	// cyclic relations
	if (rank == root){
		srand(size-1);
	  	for (i = 0; i<(keys.per_process_num_block_col_prev * blocksize); i++){
			vec_prev[i] = rand()%random_dividend;
		}

	}
	else{
		srand(rank-1);
	  	for (i = 0; i<(keys.per_process_num_block_col_prev * blocksize); i++){
			vec_prev[i] = rand()%random_dividend;
		}
	}
	// Step 3c: Combine them and generate only one vec
	int  updated_idx;
	vec = calloc(vec_length, sizeof(double));
	if (rank == root){
		for(i = 0; i<vec_length; i++){  		
	  		if(i<keys.per_process_num_block_col_later*blocksize){
	  			vec[i] = vec_later[i];
	  		}
	  		else{
				updated_idx = i - (keys.per_process_num_block_col_later*blocksize);
	  			vec[i] = vec_prev[updated_idx];
	  		}  		
		}
	}
	else{
		int offset;
		for(i = 0; i<vec_length; i++){			
	  		if(i<keys.per_process_num_block_col_prev*blocksize){
	  			vec[i] = vec_prev[i];
	  		}
	  		else if(i>=keys.per_process_num_block_col_prev*blocksize && 
	  					i<(vec_length - (keys.per_process_num_block_col_later*blocksize)))
	  		{
				updated_idx = i - (keys.per_process_num_block_col_prev*blocksize);
	  			vec[i] = vec_only[updated_idx];
	  			//if(rank == 1) {printf(" 1.updated_idx %d\n", updated_idx);}
	  		}
	  		else{
				updated_idx = i - ((keys.per_process_num_block_col_prev+keys.per_process_num_block_col_only)*blocksize);
	  			vec[i] = vec_later[updated_idx];
	  			//if(rank == 1) {printf("2. updated_idx %d\n", updated_idx);}
	  		}	
		}
	}
	free(vec_prev);
	free(vec_only);
	free(vec_later);
}

