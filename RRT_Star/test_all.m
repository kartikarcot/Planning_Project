s = load('start.txt');
e = load('end.txt');
planner_id = [0,1,2,3];
for i = 141:300
    startQ = [s(40*(0)+i) s(40*(1)+i)];
    goalQ  = [e(40*(0)+i) e(40*(1)+i)];
    i
    startQ
    goalQ
    runtest('../Maps/map3.txt', startQ, goalQ, 3);
end
