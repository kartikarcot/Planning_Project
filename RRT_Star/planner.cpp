/*=================================================================
*
* planner.c
*
*=================================================================*/
#include <math.h>
#include "mex.h"
// #include "rrt.hpp"
// #include "rrt_connect.hpp"
#include "prm.hpp"
#include "rrt_star.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
/* Input Arguments */
#define	MAP_IN      prhs[0]
#define	ARMSTART_IN	prhs[1]
#define	ARMGOAL_IN     prhs[2]
#define	PLANNER_ID_IN     prhs[3]

/* Planner Ids */
#define RRT_ID         0
#define RRTCONNECT_ID  1
#define RRTSTAR_ID     2
#define PRM_ID         3

/* Output Arguments */
#define	PLAN_OUT	plhs[0]
#define	PLANLENGTH_OUT	plhs[1]

#if !defined(GETMAPINDEX)
#define GETMAPINDEX(X, Y, XSIZE, YSIZE) (Y*XSIZE + X)
#endif

#if !defined(MAX)
#define	MAX(A, B)	((A) > (B) ? (A) : (B))
#endif

#if !defined(MIN)
#define	MIN(A, B)	((A) < (B) ? (A) : (B))
#endif

#define PI 3.141592654
#define RADIUS 2
//the length of each link in the arm (should be the same as the one used in runtest.m)
#define LINKLENGTH_CELLS 10

typedef struct {
	int X1, Y1;
	int X2, Y2;
	int Increment;
	int UsingYIndex;
	int DeltaX, DeltaY;
	int DTerm;
	int IncrE, IncrNE;
	int XIndex, YIndex;
	int Flipped;
} bresenham_param_t;

template<typename T>
std::ostream& operator << (std::ostream& os, std::vector<T> list){
    for(int i=0; i<list.size(); i++){
        os<<list[i]<<" ";
    }
    return os;
}

void ContXY2Cell(double x, double y, short unsigned int* pX, short unsigned int *pY, int x_size, int y_size)
{
	double cellsize = 1.0;
	//take the nearest cell
	*pX = (int)(x/(double)(cellsize));
	if( x < 0) *pX = 0;
	if( *pX >= x_size) *pX = x_size-1;

	*pY = (int)(y/(double)(cellsize));
	if( y < 0) *pY = 0;
	if( *pY >= y_size) *pY = y_size-1;
}


int IsValid(std::vector<double> angles, int numofDOFs, double*	map,
	int x_size, int y_size)
{
    //std::cout<<std::endl;
    int x0 = MAX(0,int(x_size*angles[0])-RADIUS), y0 = MAX(0,int(y_size*angles[1])-RADIUS);
//    std::cout<<x0<<" "<<y0<<" "<<x0+10<<" "<<y0+10<<std::endl;
    //std::cout<<"201 56 "<<map[GETMAPINDEX(201, 56, x_size, y_size)]<<std::endl;
    for(int i=0; i<2*RADIUS+1; i++){
        for(int j=0; j<2*RADIUS+1; j++){
            int a = MIN(x0+i,x_size-1), b=MIN(y0+j,y_size-1);
            //std::cout<<a<<" "<<b<<" ";
            //std::cout<<map[GETMAPINDEX(a, b, x_size, y_size)]<<" ";
            if(map[GETMAPINDEX(a, b, x_size, y_size)]==1){
                //std::cout<<a<<" "<<b<<" ";
                //std::cout<<"NOT VALID\n";
                return 0;
            }
        }
    }
    return 1;

}

int IsValid(double* angles, int numofDOFs, double* map,
    int x_size, int y_size)
{
    int x0 = MAX(0,int(x_size*angles[0])-RADIUS), y0 = MAX(0,int(y_size*angles[1])-RADIUS);
    for(int i=0; i<2*RADIUS+1; i++){
        for(int j=0; j<2*RADIUS+1; j++){
            int a = MIN(x0+i,x_size-1), b=MIN(y0+j,y_size-1);
            if(map[GETMAPINDEX(a, b, x_size, y_size)]==1){
                std::cout<<"NOT VALID\n";
                return 0;
            }
        }
    }
    return 1;
}

static void planner(
	double*	map,
	int x_size,
	int y_size,
		double* armstart_anglesV_rad,
		double* armgoal_anglesV_rad,
	int numofDOFs,
	double*** plan,
	int* planlength)
{
//no plan by default
*plan = NULL;
*planlength = 0;
	
	//for now just do straight interpolation between start and goal checking for the validity of samples
	double distance = 0;
	int i,j;
	for (j = 0; j < numofDOFs; j++){
		if(distance < fabs(armstart_anglesV_rad[j] - armgoal_anglesV_rad[j]))
			distance = fabs(armstart_anglesV_rad[j] - armgoal_anglesV_rad[j]);
	}
	int numofsamples = (int)(distance/(PI/20));
	if(numofsamples < 2){
		printf("the arm is already at the goal\n");
		return;
	}
	*plan = (double**) malloc(numofsamples*sizeof(double*));
	int firstinvalidconf = 1;
	for (i = 0; i < numofsamples; i++){
		(*plan)[i] = (double*) malloc(numofDOFs*sizeof(double)); 
		for(j = 0; j < numofDOFs; j++){
			(*plan)[i][j] = armstart_anglesV_rad[j] + ((double)(i)/(numofsamples-1))*(armgoal_anglesV_rad[j] - armstart_anglesV_rad[j]);
		}
		if(!IsValid((*plan)[i], numofDOFs, map, x_size, y_size) && firstinvalidconf)
		{
			firstinvalidconf = 1;
			printf("ERROR: Invalid arm configuration!!!\n");
		}
	}    
	*planlength = numofsamples;
	
	return;
}


//prhs contains input parameters (3): 
//1st is matrix with all the obstacles
//2nd is a row vector of start angles for the arm 
//3nd is a row vector of goal angles for the arm 
//plhs should contain output parameters (2): 
//1st is a 2D matrix plan when each plan[i][j] is the value of jth angle at the ith step of the plan
//(there are D DoF of the arm (that is, D angles). So, j can take values from 0 to D-1
//2nd is planlength (int)
void mexFunction( int nlhs, mxArray *plhs[], 
	int nrhs, const mxArray*prhs[])
	
{ 
	
	/* Check for proper number of arguments */    
	if (nrhs != 4) { 
	mexErrMsgIdAndTxt( "MATLAB:planner:invalidNumInputs",
				"Four input arguments required."); 
	} else if (nlhs != 2) {
	mexErrMsgIdAndTxt( "MATLAB:planner:maxlhs",
				"One output argument required."); 
	} 
		
	/* get the dimensions of the map and the map matrix itself*/     
	int x_size = (int) mxGetM(MAP_IN);
	int y_size = (int) mxGetN(MAP_IN);
	double* map = mxGetPr(MAP_IN);
	
	/* get the start and goal angles*/     
	int numofDOFs = (int) (MAX(mxGetM(ARMSTART_IN), mxGetN(ARMSTART_IN)));
	if(numofDOFs <= 1){
	mexErrMsgIdAndTxt( "MATLAB:planner:invalidnumofdofs",
				"it should be at least 2");         
	}
	double* armstart_anglesV_rad = mxGetPr(ARMSTART_IN);
    std::cout<<"Start "<<armstart_anglesV_rad[0]<<" "<<armstart_anglesV_rad[1]<<std::endl;
	if (numofDOFs != MAX(mxGetM(ARMGOAL_IN), mxGetN(ARMGOAL_IN))){
			mexErrMsgIdAndTxt( "MATLAB:planner:invalidnumofdofs",
				"numofDOFs in startangles is different from goalangles");         
	}
	double* armgoal_anglesV_rad = mxGetPr(ARMGOAL_IN);
    std::cout<<"End "<<armgoal_anglesV_rad[0]<<" "<<armgoal_anglesV_rad[1]<<std::endl;
	//get the planner id
	int planner_id = (int)*mxGetPr(PLANNER_ID_IN);
	if(planner_id < 0 || planner_id > 3){
	mexErrMsgIdAndTxt( "MATLAB:planner:invalidplanner_id",
				"planner id should be between 0 and 3 inclusive");         
	}
	
	//generate random start positions
	/*
	unsigned seed=0;
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<double> u_dist(0,1);
	Point q_rand(numofDOFs);
	int cnt=0;
	std::ofstream start=std::ofstream("start.txt", std::ios::app);
	while(cnt<1000){
		for(int j=0; j<numofDOFs; j++)q_rand[j]=u_dist(eng);
		if(IsValid(q_rand, numofDOFs, map,
                x_size, y_size)){
			cnt++;
			start<<q_rand<<std::endl;
		}
	}
	cnt=0;
	std::ofstream end=std::ofstream("end.txt", std::ios::app);
	while(cnt<1000){
		for(int j=0; j<numofDOFs; j++)q_rand[j]=u_dist(eng);
		if(IsValid(q_rand, numofDOFs, map, 
                x_size, y_size)){
			cnt++;
			end<<q_rand<<std::endl;
		}
    }
    */
	//call the planner
	
	double** plan = NULL;
	int planlength = 0;
	auto begin = std::chrono::high_resolution_clock::now();
	//you can may be call the corresponding planner function here
	if (planner_id == RRT_ID)
	{
        // RRT * planner = new RRT(numofDOFs,map,x_size,y_size);
        // bool found = planner->plan(armstart_anglesV_rad,armgoal_anglesV_rad,&plan, &planlength);
		// if(!found){
		// 	std::cout<<"No path found trying reverse planning \n";
		// 	delete planner;
		// 	for(int i=0; i<(planlength); i++){
		// 		delete [] (plan)[i];
		// 	}
		// 	RRT * planner = new RRT(numofDOFs,map,x_size,y_size);
        // 	found = planner->plan(armgoal_anglesV_rad,armstart_anglesV_rad,&plan, &planlength, true);
		// }
		// if(found){
		// 	auto now = std::chrono::high_resolution_clock::now();
		// 	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - begin);
		// 	planner->ofs<<"Time taken "<<duration.count()<<std::endl;
		// }
	}
	else if (planner_id == RRTCONNECT_ID)
	{
        // RRTConnect planner(numofDOFs,map,x_size,y_size);
        // bool found=planner.plan(armstart_anglesV_rad,armgoal_anglesV_rad,&plan, &planlength);
		// if(found){
		// 	auto now = std::chrono::high_resolution_clock::now();
		// 	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - begin);
		// 	planner.ofs<<"Time taken "<<duration.count()<<std::endl;
		// }
	}
    else if (planner_id == PRM_ID)
    {
         PRM planner(numofDOFs,map,x_size,y_size);
         bool found = planner.plan(armstart_anglesV_rad,armgoal_anglesV_rad,&plan, &planlength);
		 if(found){
		 	auto now = std::chrono::high_resolution_clock::now();
		 	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - begin);
		 	planner.ofs<<"Time taken "<<duration.count()<<std::endl;
		 }
    }
	else if (planner_id == RRTSTAR_ID)
    {
        RRTStar planner(numofDOFs,map,x_size,y_size);
        bool found=planner.plan(armstart_anglesV_rad,armgoal_anglesV_rad,&plan, &planlength);
		if(found){
			auto now = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - begin);
			planner.ofs<<"Time taken "<<duration.count()<<std::endl;
		}
    }
	//dummy planner which only computes interpolated path
	// planner(map,x_size,y_size, armstart_anglesV_rad, armgoal_anglesV_rad, numofDOFs, &plan, &planlength); 
	
	printf("planner returned plan of length=%d\n", planlength); 
	
	/* Create return values */
	if(planlength > 0)
	{
		PLAN_OUT = mxCreateNumericMatrix( (mwSize)planlength, (mwSize)numofDOFs, mxDOUBLE_CLASS, mxREAL); 
		double* plan_out = mxGetPr(PLAN_OUT);        
		//copy the values
		int i,j;
		for(i = 0; i < planlength; i++)
		{
			for (j = 0; j < numofDOFs; j++)
			{
				plan_out[j*planlength + i] = plan[i][j];
			}
		}
	}
	else
	{
		PLAN_OUT = mxCreateNumericMatrix( (mwSize)2, (mwSize)numofDOFs, mxDOUBLE_CLASS, mxREAL);
		double* plan_out = mxGetPr(PLAN_OUT);
		//copy the values
		int j;
        std::vector<int> d_len = {x_size,y_size};
		for(j = 0; j < numofDOFs; j++)
		{
				plan_out[j] = d_len[j]*armstart_anglesV_rad[j];
		}
        for(j = 0; j < numofDOFs; j++)
        {
                plan_out[j+numofDOFs] = d_len[j]*armgoal_anglesV_rad[j];
        }
	}
	PLANLENGTH_OUT = mxCreateNumericMatrix( (mwSize)1, (mwSize)1, mxINT8_CLASS, mxREAL); 
	int* planlength_out = (int*) mxGetPr(PLANLENGTH_OUT);
	*planlength_out = planlength;

	
	return;
	
}





