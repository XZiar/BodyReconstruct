#include "fitMesh.h"

int main(int argc, char *argv[])
{
	printf("%s\n", argv[0]);
    fitMesh meshFittor;
    meshFittor.mainProcess();
	printf("Finished!\n");
	getchar();
    return 0;
}
