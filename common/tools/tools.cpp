#include "ArgParser.h"
#include "Utils.h"

int main(int argc, char **argv){
	ArgParser ap;

	ap.parseArgs(argc,argv);
	if(ap.exists(HELP) || ap.count() == 0){
		ap.menu();
		return 0;
	}

	int c=0;
	if(!ap.exists("-c")){
		printf("Missing Number of random numbers!!!\n");
		return 1;
	}else{
		c = ap.getInt("-c");
		if (c <= 0 ||  c >=10){
			printf("give a number inclusive [1,10]: example -c=5\n");
			return 1;
		}
	}

	Utils<double> u;
	for(int i = 0; i< c;i++){
		printf("%f\n", u.uni(1.0));
	}

	return 0;

}
