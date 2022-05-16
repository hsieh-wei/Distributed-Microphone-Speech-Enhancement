noise_list=["Car_Noise_Idle 60mph","SIREN3_WAIL_FAST","Street_Noise_downtown","Water_Cooler"];
n=0;
for i=-10:5:15
    a = int2str(i);%db
    for k=1:4
        n=n+1;
        indir1 = 'D:\107 專題實驗\評分\評分indir1\EX3\indir1_target';
        indir2 = "D:\107 專題實驗\107專題資料\EX3\test noise\"+a+"db\target\"+noise_list(k);
        indir3 = "D:\107 專題實驗\107專題資料\EX3\test noise\"+a+"db\target\"+noise_list(k);
        outdir = 'D:\107 專題實驗\評分\評分outdir\EX3\target_noisy_score';
        outfilename = "target_noisy_score_"+a+"db_"+noise_list(k);
        
        indir2 = convertStringsToChars(indir2);
        indir3 = convertStringsToChars(indir3);
        outfilename = convertStringsToChars(outfilename);
        EVALUATION_MAIN_V1_V1(indir1,indir2,indir3,outdir,outfilename);
    end
end