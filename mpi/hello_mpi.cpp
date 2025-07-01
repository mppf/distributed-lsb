#include <cstdio>

#include <mpi.h>

int main (int argc, char** argv)
{
  int rank, size, length;
  char name[BUFSIZ];


  MPI_Init (&argc, &argv);      /* starts MPI */
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);        /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);        /* get number of processes */
  MPI_Get_processor_name(name, &length);

  printf( "%s: hello world from process %d of %d arg:%s\n", name, rank, size, argv[1]);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}
