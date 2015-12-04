#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

typedef  int ElemType;

int getRandom(int min, int max){
	int sub = max - min;
	printf("sub:%d\n", sub);
	srand((int)time(0));
	int j = 0 +(int)(sub * rand()/(RAND_MAX + 1.0));
	printf("j :%d\n", j);
	return j;
}
void swap(ElemType* M, int i, int j){
	int temp;
	temp = M[i];
	M[i] = M[j];
	M[j] = temp;
}

void show(ElemType* M, int length){
	int i;

	for(i = 0; i < length; ++i){
		printf("%d ", M[i]);
	}
	printf("\n");
}

int partitioning(ElemType* M, int p, int r){
	int x = M[r];
	int i = p -1;
	int j;
	for(j = p; j < r;++j){
		if(M[j] < x){
			++i;
			swap(M, j, i);
		} 
	}
	++i;
	swap(M, i, j);
	return i;
}

void quickSort(ElemType* M, int p, int r){
	int q;
	if( p <= r){
		q = partitioning(M, p, r);
		quickSort(M, p, q-1);
		quickSort(M, q+1, r);
	}
}

int randomizedPartition(ElemType* M, int p, int r){
	int i = getRandom(0, r-p-1);
	swap(M, i, r);
	printf("%d\n", i );
	return partitioning(M, p, r);
}

void randomizedQuicksort(ElemType* M, int p, int r){
	int q;
	if(p <= r){
		q = randomizedPartition(M, p, r);
		randomizedQuicksort(M, p, q-1);
		randomizedQuicksort(M, q+1, r);
	}
}
int main(){
	/*ElemType A[10] = {4, 2, 7, 1, 2 , 54, 5, 14, 2, 7};
	printf("Hello\n");
	int length = sizeof(A) / sizeof(A[0]);
	printf("%d\n", length );
	show(A, length);

	printf("\n");
	//quickSort(A, 0, length-1);
	//show(A, length);

	randomizedQuicksort(A, 0, length-1);
	show(A, length);*/
	int i = getRandom(0, 10);
	printf("%d\n", i);
	return 0;
}