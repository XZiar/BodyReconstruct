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
void sleepMS(uint32_t ms)
{
	std::this_thread::sleep_for(std::chrono::milliseconds(ms));
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
    fitMesh meshFittor("../BodyReconstruct/data/");
	//input object scan file
	{
		printf("FileName(number for one scan, letter for sequence): ");
		int ret = getchar();
		if (!(ret == 13 || ret == 10) && !isVtune)
		{
			getchar();
			if (ret >= '0' && ret <= '9')
				meshFittor.init(string("scan") + (char)ret, true);
			else
				meshFittor.init(string("/clips/clouds") + (char)ret, false);
		}
		else
		{
			meshFittor.init("scan", true);
		}
	}
    meshFittor.mainProcess();
	if (!isVtune)
	{
		if (yesORno("load previous param to watch?"))
			meshFittor.watch("params_n_f70");
		else
			meshFittor.watch();
	}
    return 0;
}
