noise_list=["battle016","buccaneer1","cafeteria_babble","n36","n84","PC Fan Noise","pinknoise","發動機噪聲"];
for i=-5:3:16
  a = int2str(i);%db
  for j=1:3
     b = int2str(j);%ch
     for k=1:8
         for l=1:300
               c = int2str(l);%wav
               InpPath = strcat('D:\107 專題實驗\107專題資料\107專題原始音檔\EX6\ch',b,'\',c,'.wav');
               NoiseInpPath = strcat('D:\107 專題實驗\107專題資料\純噪聲\噪聲tr\',noise_list(k),'.wav');
               OutDir = strcat('D:\107 專題實驗\107專題資料\EX6\train noise\',a,'db\ch',b,'\',noise_list(k),'\',c,'.wav');
               SNR=i; % dB
               INDICATOR='tr';% for training : 'tr'; for testing: 'ts'
               add_noise_v3_1(InpPath,NoiseInpPath,OutDir,SNR,INDICATOR);   
         end
     end
  end
end