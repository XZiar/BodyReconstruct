#include "main.h"
#include "fitMesh.h"

using std::string;


uint64_t hash_(const string& str)
{
	uint64_t hash = 0;
	for (size_t a = 0; a < str.length(); ++a)
		hash = hash * 33 + str[a];
	return hash;
}



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
bool yesORno(const char *str, const bool expect)
{
	if (isVtune)
		return expect;
	while (true)
	{
		printf("%s(y/n): ", str);
		const int key = getchar();
		if (key == 'y' || key == 'n')
		{
			getchar();
			return key == 'y';
		}
	}
}
int32_t inputNumber(const char *str, const int32_t expect)
{
	if (isVtune)
		return expect;
	printf("%s : ", str);
	int32_t num = expect;
	scanf("%d", &num);
	getchar();
	return num;
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
		string ifn;
		cin >> ifn;
		getchar();
		bool isSingle = true;
		for(const auto& ch : ifn)
			if (ch < '0' || ch > '9')
			{
				isSingle = false;
				break;
			}
		if(isSingle)
			meshFittor.init(ifn, "scan" + ifn, true);
		else
			meshFittor.init(ifn, "/clips/clouds" + ifn, false);
	}
    meshFittor.mainProcess();
	if (!isVtune)
	{
		if (yesORno("load previous param to watch?"))
		{
			const auto fcnt = inputNumber("up to how many frames?", 70);
			meshFittor.watch(fcnt);
		}
		else
			meshFittor.watch();
		printf("\nFinished!\n");
		getchar();
	}
    return 0;
}
