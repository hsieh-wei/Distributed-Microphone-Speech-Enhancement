noise_list=["Car_Noise_Idle 60mph","SIREN3_WAIL_FAST","Street_Noise_downtown","Water_Cooler"];
for i=-10:5:15
  a = int2str(i);%db
  for j=1:3
     b = int2str(j);%ch
     for k=1:4
         for l=301:320
               c = int2str(l);%wav
               InpPath = strcat('D:\107 專題實驗\107專題資料\107專題原始音檔\EX6\ch',b,'\',c,'.wav');
               NoiseInpPath = strcat('D:\107 專題實驗\107專題資料\純噪聲\噪聲ts\',noise_list(k),'.wav');
               OutDir = strcat('D:\107 專題實驗\107專題資料\EX6\test noise\',a,'db\ch',b,'\',noise_list(k),'\',c,'.wav');
               SNR=i; % dB
               INDICATOR='ts';% for training : 'tr'; for testing: 'ts'
               add_noise_v3_1(InpPath,NoiseInpPath,OutDir,SNR,INDICATOR);     
         end
     end
  end
end 