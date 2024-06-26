function TV=temporalvariability(MWNs)
timeframe=size(MWNs,3);
channel_num=size(MWNs,1);
for i=1:channel_num
    x(:,:)=MWNs(i,:,:);
    corr=corrcoef(x); 
    tr(i)=1-sum(sum(corr))/(timeframe*(timeframe-1)+0.00001);
    x=[];
end
