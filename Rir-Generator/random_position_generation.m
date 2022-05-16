c = 340;                     % Sound velocity (m/s)
fs = 16000;                  % Sample frequency (samples/s)
nsample = 4096;              % Number of samples
beta = 0.0;                  % Reverberationtime (s)
hp_filter= 0;
mtype = 'omnidirectional';
order = -1; 
dim = 3;
orientation = [0,0];            
L = [ 8.475 10.8 3 ];                            % Room dimensions [ x y z ] (m)
r1 = [4.2375+1*cosd(90)  5.4+1*sind(90)  1.5 ];  % Receiver1 position [ x y z ] (m) //room
r2 = [4.2375+1*cosd(210) 5.4+1*sind(210) 1.5];   % Receiver2 position [ x y z ] (m) //room
r3 = [4.2375+1*cosd(330) 5.4+1*sind(330) 1.5];   % Receiver3 position [ x y z ] (m) //room
r4 = [4.2375 5.4 1.5];                           % Receiver4 position [ x y z ] (m) //room
s = zeros(320,3);                         % Source position [ x y z ] (m) //room
rader_source = zeros(320,3);              % Source position [ x y z ] (m) //rader

for i=1:1:320
    position_degree_generator =  rand*(360);
    position_radius_generator =  rand*(1);
    position_z = 1.5;
    
    s(i,1) = 4.2375+position_radius_generator*cosd(position_degree_generator);  % Source position x //room
    s(i,2) = 5.4+position_radius_generator*sind(position_degree_generator);     % Source position y //room
    s(i,3) = position_z;                                                        % Source position z //room
    
    rader_source(i,1) = position_radius_generator*cosd(position_degree_generator); % Source position x //rader
    rader_source(i,2) = position_radius_generator*sind(position_degree_generator); % Source position y //rader
    rader_source(i,3) = position_z;                                                % Source position z //rader
    
    h1 = rir_generator(c, fs, r1, s(i,:), L, beta, nsample, mtype, order, dim, orientation, hp_filter);
    h2 = rir_generator(c, fs, r2, s(i,:), L, beta, nsample, mtype, order, dim, orientation, hp_filter);
    h3 = rir_generator(c, fs, r3, s(i,:), L, beta, nsample, mtype, order, dim, orientation, hp_filter);
    h4 = rir_generator(c, fs, r4, s(i,:), L, beta, nsample, mtype, order, dim, orientation, hp_filter);
    
    wav = int2str(i);
    test_wav_dir = strcat('D:\107 專題實驗\107專題資料\107專題原始音檔\EX4\target\',wav,'.wav');
    test_wav = audioread(test_wav_dir);
    
    mix_wav1 = fftfilt(h1,test_wav);
    mix_wav2 = fftfilt(h2,test_wav);
    mix_wav3 = fftfilt(h3,test_wav);
    mix_wav4 = fftfilt(h4,test_wav);
    
    audiowrite_dir1= strcat('D:\107 專題實驗\107專題資料\107專題原始音檔\EX4\ch1\',wav,'.wav');
    audiowrite_dir2= strcat('D:\107 專題實驗\107專題資料\107專題原始音檔\EX4\ch2\',wav,'.wav');
    audiowrite_dir3= strcat('D:\107 專題實驗\107專題資料\107專題原始音檔\EX4\ch3\',wav,'.wav');
    audiowrite_dir4= strcat('D:\107 專題實驗\107專題資料\107專題原始音檔\EX4\ch4\',wav,'.wav');
    audiowrite(audiowrite_dir1,mix_wav1,16000);
    audiowrite(audiowrite_dir2,mix_wav2,16000);
    audiowrite(audiowrite_dir3,mix_wav3,16000);
    audiowrite(audiowrite_dir4,mix_wav4,16000);
end
save( 'source_position_room.mat', 's');
save( 'source_position_rader.mat', 'rader_source');






