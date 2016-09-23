#include "Time.h"


int main(){

	Time<millis> t;

	t.start();
	int k=0;
	for (int i =0;i<100000;i++){
		k++;
	}
	t.lap("Elapsed Time in ms");


	return 0;
}
