#include "main.h"
#include "fitMesh.h"

using std::string;
uint64_t getCurTime()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
uint64_t getCurTimeNS()
{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
bool yesORno(const char *str)
{
	if (isVtune)
		return true;
	printf("%s(y/n): ", str);
	int key = getchar();
	getchar();
	return key == 'y';
}
bool isVtune = false;

int main(int argc, char *argv[])
{
	printf("%s\n", argv[0]);
	for (int a = 1; a < argc; ++a)
	{
		const string para(argv[a]);
		if (para == "--vtune")
			isVtune = true;
	}
    fitMesh meshFittor;
    meshFittor.mainProcess();
	printf("Finished!\n");
	if(!isVtune)
		getchar();
    return 0;
}
