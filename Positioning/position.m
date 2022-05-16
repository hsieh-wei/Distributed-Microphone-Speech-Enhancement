%%
%要取位置的frame
frame =[125
186
276
360
461
525
634
701
809
884
953
1034
1109
1237
1332
1401
1468
1581
1629
1719
];
%%
%1是X軸 2是高度 3是Y軸
load('position320.mat');%rue_data的資料
frame_num = length(true_data);%所有frame數
num = size(frame,1);%要取的位置數
frame_count=1;%取到第幾個位置
x_single_reg=[];%暫存filter後的x座標
y_single_reg=[];%暫存filter後的y座標
coordinate = zeros(num,3);%位置資訊
for i=1:1:frame_num
    if i==frame(frame_count,1)
        x_reg = true_data{i,4}(1,:);%一個frame裡面所有x座標
        y_reg = true_data{i,4}(3,:);%一個frame裡面所有y座標
        point_cloud_num = size(x_reg,2);%一個frame裡面的點雲數
        for j=1:1:point_cloud_num
            if x_reg(1,j)<1 && x_reg(1,j)>-1 && y_reg(1,j)>0 && y_reg(1,j)<2%filter
                x_single_reg = [x_single_reg x_reg(1,j)];
                y_single_reg = [y_single_reg y_reg(1,j)];
            end
        end
        x_coordinate = mean(x_single_reg,'all');
        y_coordinate = mean(y_single_reg,'all');
        
        coordinate(frame_count,1) = x_coordinate;%位置資訊x
        coordinate(frame_count,2) = y_coordinate-1;%位置資訊y
        coordinate(frame_count,3) = 1.72;%位置資訊z
        
        x_single_reg=[];
        y_single_reg=[];
        
        
        frame_count=frame_count+1;
        
        
        if frame_count > num
            break;
        end
    end  
end

save( 'coordinate320', 'coordinate');
%% 檢查取的點對不對
 frame_check = 1;
 x= coordinate(frame_check,1);
 y= coordinate(frame_check,2);
 figure(1)
 axis([-1,1,0,2]);
 plot(x,y,'r.');
 grid on;
 hold on;
     
 xlabel('X');
 ylabel('Y');
