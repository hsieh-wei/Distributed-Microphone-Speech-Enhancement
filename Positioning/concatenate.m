
c1=load('coordinate46.mat');
c2=load('coordinate100.mat');
c3=load('coordinate150.mat');
c4=load('coordinate200.mat');
c5=load('coordinate250.mat');
c6=load('coordinate300.mat');
c7=load('coordinate320.mat');
coordinate = [c1.coordinate ; c2.coordinate ; c3.coordinate ; c4.coordinate ; c5.coordinate ; c6.coordinate ; c7.coordinate];
save( 'coordinate', 'coordinate');