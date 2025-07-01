// based on https://www.usna.edu/Users/cs/fknoll/shmem/Lesson0_HelloWorld.html
//
// compile with
//   oshc++ -O3 hello-shmem.cpp -o hello-shmem
// run with
//   oshrun -np 4 ./hello-shmem

#include <cstdio>
#include <shmem.h>

int main(){
  int my_pe, num_pe;         //declare variables for both pe id of processor and the number of pes
  
  shmem_init();
  num_pe = shmem_n_pes();    //obtain the number of pes that can be used
  my_pe  = shmem_my_pe();    //obtain the pe id number
  
  printf("Hello from %d of %d\n", my_pe, num_pe);
  shmem_finalize();
  return 0;
}
