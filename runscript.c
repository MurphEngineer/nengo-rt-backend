#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char *argv[]){
  /*
    char *envp[2];
    envp[0] = "PYTHONPATH=/home/mtrberzi/nengo";
    envp[1] = (char*)NULL;
  */

  setuid(0);
  execlp("python3", "", argv[1], (char*)NULL);
  // does not return
}
