function EVALUATION_MAIN_V1_V2_V1(indir1,indir2,indir3,outdir,outfilename)

% indir1: document path for clean data
% indir2: document path for noisy data
% indir3: document path for enhanced data
% outdir: document path for evaluated results

title_flag=1;

if  indir1(end) == filesep
    indir1=indir1(1:(end-1));
end
if  indir2(end) == filesep
    indir2=indir2(1:(end-1));
end
if  indir3(end) == filesep
    indir3=indir3(1:(end-1));
end

if  strcmp(outdir(end),'\') || strcmp(outdir(end),'/')
    outdir=outdir(1:(end-1));
end

if exist(outdir) ~=7
    mkdir(outdir);
end

sdi=[];HASQI=[];HASPI=[];
pesq_mos=[];stoi_scor=[];
ssnr_dB=[];fw=-1;

filelist_1=dir(indir1);
filelist_2=dir(indir2);
filelist_3=dir(indir3);
filelist_len=length(filelist_3);

for k=3:filelist_len
    [pathstr_1,filenamek_1,ext_1] = fileparts(filelist_1(k).name);
    [pathstr_2,filenamek_2,ext_1] = fileparts(filelist_2(k).name);
    [pathstr_3,filenamek_3,ext_1] = fileparts(filelist_3(k).name);
    if filelist_1(k).isdir
       [title_flag,sdi,HASQI,HASPI,pesq_mos,stoi_scor,ssnr_dB,fw]=EVALUATION_Sub_1([indir1 filesep filenamek_1],[indir2 filesep filenamek_2],[indir3 filesep filenamek_3],outdir,outfilename,title_flag,sdi,HASQI,HASPI,pesq_mos,stoi_scor,ssnr_dB,fw);
    else
       [title_flag,sdi,HASQI,HASPI,pesq_mos,stoi_scor,ssnr_dB,fw]=EVALUATION_Sub_1(indir1,indir2,indir3,outdir,outfilename,title_flag,sdi,HASQI,HASPI,pesq_mos,stoi_scor,ssnr_dB,fw);
    end
end

EVALUATION_Sub_2(sdi,HASQI,HASPI,pesq_mos,stoi_scor,ssnr_dB,fw);


end

function [title_flag,sdi,HASQI,HASPI,pesq_mos,stoi_scor,ssnr_dB,fw]=EVALUATION_Sub_1(indir1,indir2,indir3,outdir,outfilename,title_flag,sdi,HASQI,HASPI,pesq_mos,stoi_scor,ssnr_dB,fw)

if  indir1(end) == filesep
    indir1=indir1(1:(end-1));
end
if  indir2(end) == filesep
    indir2=indir2(1:(end-1));
end
if  indir3(end) == filesep
    indir3=indir3(1:(end-1));
end

if  strcmp(outdir(end),'\') || strcmp(outdir(end),'/')
    outdir=outdir(1:(end-1));
end

filelist_1=dir(indir1);
filelist_2=dir(indir2);
filelist_3=dir(indir3);
filelist_len=length(filelist_3);

for k=3:filelist_len
    [pathstr_1,filenamek_1,ext_1] = fileparts(filelist_1(k).name);
    [pathstr_2,filenamek_2,ext_1] = fileparts(filelist_2(k).name);
    [pathstr_3,filenamek_3,ext_1] = fileparts(filelist_3(k).name);
    if filelist_1(k).isdir
        [title_flag,sdi,HASQI,HASPI,pesq_mos,stoi_scor,fw]=EVALUATION_Sub_1([indir1 filesep filenamek_1],[indir2 filesep filenamek_2],[indir3 filesep filenamek_3],outdir,outfilename,title_flag,sdi,HASQI,HASPI,pesq_mos,stoi_scor,ssnr_dB,fw);
    else
        
        % tp=1:10:41;iii=1;
        
        CleanDataFile=fullfile(indir1, filelist_1(k).name);
        NoisyDataFile=fullfile(indir2, filelist_2(k).name);
        EnhadDataFile=fullfile(indir3, filelist_3(k).name);
        
        [TCleanData,fc]=audioread(CleanDataFile);
        [TNoisyData,fn]=audioread(NoisyDataFile);
        [TEnhadData,fe]=audioread(EnhadDataFile);
        
        minimum_points=min([length(TCleanData),length(TNoisyData),length(TEnhadData)]);
        
        TCleanData=TCleanData/std(TCleanData(1:minimum_points));
        TNoisyData=TNoisyData/std(TNoisyData(1:minimum_points));
        TEnhadData=(TEnhadData)/std(TEnhadData(1:minimum_points));
        
%         Idx=find(TCleanData(1:minimum_points).^2>0.005);  % Simple VAD
        Idx=1:minimum_points;
        
        CleanData=TCleanData(Idx)/std(TCleanData(Idx));
        NoisyData=TNoisyData(Idx)/std(TNoisyData(Idx));
        EnhadData=TEnhadData(Idx)/std(TEnhadData(Idx));
        
        % for HASQI used!
        HL = [0, 0, 0, 0, 0, 0];
        eq = 2;
        Level1 = 65;
        % for SSNR
        len=256; % frame length
        
        %STOI
        stoi_scor(end+1) = stoi(CleanData, EnhadData, fn);
        
        % SDI test
        sdi(end+1)=compute_sdi(CleanData,EnhadData);
        % SSNR
        if((length(NoisyData) == length(EnhadData)))
            ssnr_dB(end+1)=ssnr(EnhadData,NoisyData,CleanData,len);
            %             ssnr_dB(1)=0;
        end
        %HASQI
        [Combined,Nonlin,Linear,raw]=HASQI_v2(CleanData,fc,EnhadData,fe,HL,eq,Level1);
        HASQI(end+1)=Combined;
        %HASPI
        [Intel,raw] = HASPI_v1(CleanData,fc,EnhadData,fe,HL,Level1);
        HASPI(end+1)=Intel;
        %PESQ
        pesq_mos(end+1)=pesq(CleanDataFile, EnhadDataFile);
        
        
        %Writing process
        if title_flag == 1
            fw=fopen(sprintf('%s%s%s.txt',outdir,filesep,outfilename),'wb');
            if fw ~= -1
                fprintf(fw,'%20s:\t%7s\t%8s\t%8s\t%8s\t%8s\t%9s\n','EVALUATED METHODS','PESQ','HASQI','HASPI','SDI','STOI','SSNR');
                title_flag=0;
            else
                disp('Error: Cannot open the text file. Stop.');
                break;
            end
        end
        
        %'PESQ','HASQI','HASPI','SDI','STOI','SSNR'
        if((length(NoisyData) ~= length(EnhadData)) || (sum(NoisyData-EnhadData) ~= 0))
            fprintf(fw,'%20s:\t%f\t%f\t%f\t%f\t%f\t%f\n',filelist_1(k).name,pesq_mos(end),HASQI(end),HASPI(end),sdi(end),stoi_scor(end),ssnr_dB(end));
        else
            fprintf(fw,'%20s:\t%f\t%f\t%f\t%f\t%f\t%f\n',filelist_1(k).name,pesq_mos(end),HASQI(end),HASPI(end),sdi(end),stoi_scor(end),0);
        end
        
    end
end
end

function EVALUATION_Sub_2(sdi,HASQI,HASPI,pesq_mos,stoi_scor,ssnr_dB,fw)

if ~isempty(sdi)
    
    % mean and variance calcualtion
    mean_sdi=mean(sdi);
    std_sdi=std(sdi);
    mean_ssnr=mean(ssnr_dB);
    std_ssnr=std(ssnr_dB);
    mean_hasqi=mean(HASQI);
    std_hasqi=std(HASQI);
    mean_haspi=mean(HASPI);
    std_haspi=std(HASPI);
    mean_pesq=mean(pesq_mos);
    std_pesq=std(pesq_mos);
    mean_stoi=mean(stoi_scor);
    std_stoi=std(stoi_scor);
    
    %Writing process
    %'PESQ','HASQI','HASPI','SDI','STOI','SSNR'
    fprintf(fw,'%20s:\t%f\t%f\t%f\t%f\t%f\t%f\n','Mean',mean_pesq,mean_hasqi,mean_haspi,mean_sdi,mean_stoi,mean_ssnr);
    fprintf(fw,'%20s:\t%f\t%f\t%f\t%f\t%f\t%f\n','Stad',std_pesq,std_hasqi,std_haspi,std_sdi,std_stoi,std_ssnr);
    fclose(fw);
end
end