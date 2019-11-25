

function[numofmoves, caught] = runtest()
% add current folder to the python search path
% if count(py.sys.path,'') == 0
%     insert(py.sys.path,int32(0),'');
% end

mapfile = "map2.txt"; % define the txt file
% wrapped_mapfile = py.textwrap.wrap(mapfile);
% wrapped_mapfile = "map2.txt";
armstart = py.list({0,0}); % define start as a python list
armgoal = py.list({2,1}); % define goal as a python list

LINKLENGTH_CELLS=10;
envmap = load(mapfile);

close all;

%draw the environment
figure('units','normalized','outerposition',[0 0 1 1]);
imagesc(envmap'); axis square; colorbar; colormap jet; hold on;

%armplan should be a matrix of D by N 
%where D is the number of DOFs in the arm (length of armstart) and
%N is the number of steps in the plan 
armplan_python = armplanner(mapfile, armstart, armgoal); 
arm_plan = cellfun(@char,cell(armplan_python),'UniformOutput',false);
fprintf(1, 'plan of length %d was found\n', size(armplan,1));

%draw the plan
midx = size(envmap,2)/2;
x = zeros(length(armstart)+1,1);
x(1) = midx;
y = zeros(length(armstart)+1,1);
for i = 1:size(armplan)
    for j = 1:length(armstart)
        x(j+1) = x(j) + LINKLENGTH_CELLS*cos(armplan(i,j));
        y(j+1) = y(j) + LINKLENGTH_CELLS*sin(armplan(i,j));
    end
    plot(x,y, 'c-');
    pause(0.1);
end

end

function[armplan] = armplanner(envmap, armstart, armgoal)
%call the planner in C
armplan = py.rrt_planner.planner(envmap, armstart, armgoal);
end