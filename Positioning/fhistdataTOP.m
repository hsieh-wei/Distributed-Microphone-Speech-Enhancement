clc;
clear;
close all;
load('fhistRT320.mat')


Ydata=[];
FrameNum=1;
i=1;
ii=1;
while i<=length(fHist)
    if ~isempty(fHist(i).pointCloud)
        true_data{ii,1}=FrameNum;  
        %true_data{ii,2}=fHist(i).TIME;
        true_data{ii,3}=fHist(i).pointCloud;
        FrameNum=FrameNum+1;
        ii=ii+1;
    end
    i=i+1;
end

high=2.90;

for i=1:length(true_data)
    for a=1:length(true_data{i,3})
        r = true_data{i,3}(1, :);
        true_data{i,4}(3, :) = r.*sin(true_data{i,3}(3,:));%y
        true_data{i,4}(2, :) =high- r.*cos(true_data{i,3}(2,:)).*cos(true_data{i,3}(3,:));%����
        true_data{i,4}(1, :) = r.*sin(true_data{i,3}(2,:)).*cos(true_data{i,3}(3,:));%x
    end
       Ydata=cat(2,Ydata,true_data{i,4}(2, :));
end
min=min(Ydata);
max=max(Ydata);
save( 'position320', 'true_data');
% for i=1:length(true_data)
%     for a=1:length(true_data{i,3})
%         true_data{i,4}(2, a) =true_data{i,4}(2, a)-min;
%     end
% end
%% DBSCAN
dbdata=true_data;
for i=1:length(true_data)
    
end 
%% Draw213
%for i=1:length(true_data)
%    x=[true_data{i,4}(1,1:end)];
%    y=[true_data{i,4}(2,1:end)];
%    z=[true_data{i,4}(3,1:end)];
%    average=sum(y)/length(y);
%    axis([-4,4,-4,4,0,3]);
%    
%    scatter3(x,z,y,400,'.','blue');
%    grid on;
%    hold on;
%    title(true_data{i,1})
%
%    xlabel('X');
%    ylabel('Z');
%    zlabel('����');
%1     set(gca,'xtick',-3:0.5:3);
%1     set(gca,'yTick',-3:0.5:3);
%1     set(gca,'zTick',-3:0.5:3);

%    drawnow;
%    pause(0.005) 

%     cla;
%    view(0,90); %XOZ����(����)   
%1     view(-90,360);%YOZ����(�e��)
%1     view(0,0);%YOX����(���k)
%end

%% ��V���k

 %count=310;
 %x=[true_data{count,4}(1,1:end)];
 %y=[true_data{count,4}(2,1:end)];
 %z=[true_data{count,4}(3,1:end)];
 %figure(1)
 %axis([-5,5,-5,5,0,3]);
 %scatter3(x,z,y,400,'.','blue');
 %title(count)
 %grid on;
 %hold on;

%xlabel('X');
%ylabel('Z');
%zlabel('Y');
%view(0,0);%YOX����(���k)

%% ��V ����
  count=519;
  x=[true_data{count,4}(1,1:end)];
  y=[true_data{count,4}(2,1:end)];
  z=[true_data{count,4}(3,1:end)];
  figure(2)
  axis([-5,5,-5,5,0,3]);
  scatter3(x,z,y,400,'.','blue');
  title(count)
  grid on;
  hold on;
     
 xlabel('X');
 ylabel('Y');
 zlabel('Y');
 view(0,90); %XOZ����(����)  

%% ��V �e��
%  x=[true_data{count,4}(1,1:end)];
%  y=[true_data{count,4}(2,1:end)];
%  z=[true_data{count,4}(3,1:end)];
%   
%  figure(3)
%  axis([-5,5,-5,5,0,3]);
%  scatter3(x,z,y,400,'.','blue');
%     
%  grid on;
%  hold on;
%  title(count)
% xlabel('X');
% ylabel('Z');
% zlabel('Y');
% set(gca,'xtick',-3:0.5:3);
% set(gca,'yTick',-3:0.5:3);
% set(gca,'zTick',-3:0.5:3);
% view(-90,360);%YOZ����(�e��) 
