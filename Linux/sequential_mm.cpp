#include<iostream>
#include<chrono>
using namespace std;

int main(){
cout << "Starting Matrix Multiplication Test... " << endl;

int N= 256;
float*A= new float[N*N];
float*B= new float[N*N];
float*C= new float[N*N];

//initialize matrices
for(int i=0; i<N*N; i++){
A[i]= 1.0f;
B[i]= 2.0f;
C[i]= 0.0f;
}

cout<< "Matrices initialzed. Starting multiplication..." << endl;

auto start = chrono::high_resolution_clock::now();

//Matrix multiplication
for(int i=0; i<N; i++){
for(int j=0; j<N; j++){
float sum= 0.0f;
for(int k=0; k<N; k++){
sum+= A[i*N+k] *B[k*N+j];
}
C[i*N+j]=sum;
}
}

auto end= chrono::high_resolution_clock::now();
auto duration=chrono::duration_cast<chrono::milliseconds>(end-start);

cout<< "Time taken:" << duration.count()<< "milliseconds"<< endl;
cout<< "C[0]="<< C[0] << "( should be  " << 1.0f*2.0f*N <<")" <<endl;

delete[] A;
delete[] B;
delete[] C;
return 0;
}

