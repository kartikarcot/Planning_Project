function[numofmoves, caught] = runtest(mapfile, armstart, armgoal, planner_id)

LINKLENGTH_CELLS=10;
envmap = load(mapfile);

close all;

%draw the environment
figure('units','normalized','outerposition',[0 0 1 1]);
imagesc(envmap'); axis square; colorbar; colormap jet; hold on;

%armplan should be a matrix of D by N 
%where D is the number of DOFs in the arm (length of armstart) and
%N is the number of steps in the plan 
armplan = armplanner(envmap, armstart, armgoal, planner_id); 

fprintf(1, 'plan of length %d was found\n', size(armplan,1));

%draw the plan
midx = size(envmap,2)/2;
x = zeros(length(armstart)+1,1);
x(1) = midx;
y = zeros(length(armstart)+1,1);
for i = 1:size(armplan)-1
    r = 2;
    xc = armplan(i,1);
    yc = armplan(i,2);
    xn = armplan(i+1,1);
    yn = armplan(i+1,2);
    theta = linspace(0,2*pi);
    x = r*sin(theta) + xc;
    y = r*cos(theta) + yc;
    plot(x,y)
    fill(x, y, 'g')
    plot([xc,xn],[yc,yn])
    pause(0.01);
end
armplan(end,:)
xc = armplan(end,1);
yc = armplan(end,2);
theta = linspace(0,2*pi);
x = r*sin(theta) + xc;
y = r*cos(theta) + yc;
plot(x,y)
fill(x, y, 'g')
%armplan
