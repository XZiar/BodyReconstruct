#include "main.h"
#include "fitMesh.h"

uint64_t getCurTime()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
uint64_t getCurTimeNS()
{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char *argv[])
{
	printf("%s\n", argv[0]);
    fitMesh meshFittor;
    meshFittor.mainProcess();
	printf("Finished!\n");
	getchar();
    return 0;
}
