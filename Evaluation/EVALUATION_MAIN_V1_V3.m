function EVALUATION_MAIN_V1_V3(inlist1,inlist2,inlist3,outdir,outfilename)

% indir1: document path for clean data
% indir2: document path for noisy data
% indir3: document path for enhanced data
% outdir: document path for evaluated results

title_flag=1;

for ref_ind=1:length(inlist3)

        filelist=inlist3{ref_ind}(end-35:end-4);
        
        [TCleanData,fc]=audioread([inlist1{ref_ind}(1:end-4),'.wav']);
        [TNoisyData,fn]=audioread([inlist2{ref_ind}(1:end-4),'.wav']);
        [TEnhadData,fe]=audioread([inlist3{ref_ind}(1:end-4),'.wav']);
        
        %         if ((1) ~= tp(iii))
        %             Tpp=[];Tpp=TEnhadData(257:end);
        %             TEnhadData=[];TEnhadData=Tpp;
        %         else
        %             iii=iii+1;
        %             if iii > length(tp)
        %                 iii=length(tp)
        %             end
        %         end
        
        minimum_points=min([length(TCleanData),length(TNoisyData),length(TEnhadData)]);
        
        TCleanData=TCleanData/std(TCleanData(1:minimum_points));
        TNoisyData=TNoisyData/std(TNoisyData(1:minimum_points));
        TEnhadData=TEnhadData/std(TEnhadData(1:minimum_points));
        
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
        stoi_scor(ref_ind) = stoi(CleanData, EnhadData, fn);
        
        % SDI test
        sdi(ref_ind)=compute_sdi(CleanData,EnhadData);
        % SSNR
        if((length(NoisyData) == length(EnhadData)))
            ssnr_dB(ref_ind)=ssnr(EnhadData,NoisyData,CleanData,len);
            %             ssnr_dB(1)=0;
        end
        %HASQI
        [Combined,Nonlin,Linear,raw]=HASQI_v2(CleanData,fc,EnhadData,fe,HL,eq,Level1);
        HASQI(ref_ind)=Combined;
        %HASPI
        [Intel,raw] = HASPI_v1(CleanData,fc,EnhadData,fe,HL,Level1);
        HASPI(ref_ind)=Intel;
        %PESQ
        pesq_mos(ref_ind)=pesq([inlist1{ref_ind}(1:end-4),'.wav'], [inlist3{ref_ind}(1:end-4),'.wav']);
        
        
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
            fprintf(fw,'%20s:\t%f\t%f\t%f\t%f\t%f\t%f\n',filelist,pesq_mos(end),HASQI(end),HASPI(end),sdi(end),stoi_scor(end),ssnr_dB(end));
        else
            fprintf(fw,'%20s:\t%f\t%f\t%f\t%f\t%f\t%f\n',filelist,pesq_mos(end),HASQI(end),HASPI(end),sdi(end),stoi_scor(end),0);
        end
        
end
    

EVALUATION_Sub_2(sdi,HASQI,HASPI,pesq_mos,stoi_scor,ssnr_dB,fw);


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