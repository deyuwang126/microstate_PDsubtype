function SV=spatialvariability(MWNs)
window_num=MWNs(net,3);
channel_num=MWNs(net,1);
for i=1:channel_num
    N=[];
    for j=1:channel_num
        if i~=j
            a=[a,MWNs(i,j,:)];              
        end
    end
    x(:,:)=N;
    corr=corrcoef(x');
    sr(i)=1-(sum(sum(corr))-channel_num)/((channel_num-1)*(channel_num-2)+0.00001);
end
