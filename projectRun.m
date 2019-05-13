%% Main run 
function[accuracy] =projectRun(classifier,TrainValue,TestValue,K,distanceFunction ,numOfNodesForEachLayer,epochs,eta,gradiantType)
%% Read Data, Split Data and Prepare Data 
[F,N,O,S,Z] = ReadData( 'F','N','O','S','z');
[Normal,Upnormal]=SplittingData(F,N,O,S,Z);
if classifier == 1 && classifier == 3
Normal=Normal ./max(Normal(:));
Upnormal(:,1:20)=Upnormal(:,1:20)./max(Upnormal(:,1:20));
end
[XTrain, YTrain, XTest, YTest] = PrepareTheData(TrainValue,TestValue,Normal,Upnormal); 
Train=[XTrain YTrain];
XTrainTemp = Train(randperm(size(Train, 1)), :);
XTrain=XTrainTemp(:,1:20);
YTrain=XTrainTemp(:,21);

%% Test KNN Classifier 
if classifier == 2 
YPredict=KnnClassifier(XTrain,YTrain,XTest,K,distanceFunction);
accuracy= CalculateAccuracy(YTest,YPredict)
end

%% Test Bayes Classifier 
if classifier == 1
[Mus,sigmas]=GetMusSigmas(TrainValue,XTrain);
YPredict = GeneralBayesClassifier(XTest,Mus,sigmas,TestValue);
accuracy=CalculateAccuracy(YTest,YPredict)
end

%% Test Neural Network Classifier 
if classifier == 3
    if gradiantType == 1
    accuracy= fullBatchMain(TrainValue,TestValue,numOfNodesForEachLayer,eta,epochs);
    else
    accuracy = stochasticMain(TrainValue,TestValue,eta,epochs,numOfNodesForEachLayer);
    end
end

end

%% Data Preparetion Functions 
function [ F,N,O,S,Z] = ReadData( FileNameF,FileNameN,FileNameO,FileNameS,FileNameZ )
txt='.txt';
fileID = fopen(strcat(FileNameF,txt),'r');
formatSpec = '%g';
sizeA = [20 inf];
F = fscanf(fileID,formatSpec,sizeA);
F=F.';


fileID = fopen(strcat(FileNameN,txt),'r');
formatSpec = '%g';
sizeA = [20 inf];
N = fscanf(fileID,formatSpec,sizeA);
N=N.';


fileID = fopen(strcat(FileNameO,txt),'r');
formatSpec = '%g';
sizeA = [20 inf];
O = fscanf(fileID,formatSpec,sizeA);
O=O.';

fileID = fopen(strcat(FileNameS,txt),'r');
formatSpec = '%g';
sizeA = [20 inf];
S = fscanf(fileID,formatSpec,sizeA);
S=S.';

fileID = fopen(strcat(FileNameZ,txt),'r');
formatSpec = '%g';
sizeA = [20 inf];
Z = fscanf(fileID,formatSpec,sizeA);
Z=Z.';
end
function [Normal,Upnormal] =SplittingData(F,N,O,S,Z)
Normal=cat(1,Z,O);
Upnormal=cat(1,F,N,S);
Normal(1:800,21)=0;
Upnormal(1:1200,21)=1;
end
function [XTrain, YTrain, XTest, YTest] = PrepareTheData(SplitTrain,SplitTest,Normal,Upnormal)
    
    [row,col]=size(Normal);
    [row,col]=size(Upnormal);
    TotalSample=SplitTrain+SplitTest;
    col=col-1;
    %SplitTrain=350
    %SplitTest=50
    
    %XTrain =Size(1750,20)
    %YTrain =Size(1750,1)
    %XTest =Size(250,20)
    %YTest =Size(250,1)
    
    XTrain=zeros(SplitTrain*5,col);
    YTrain=zeros(SplitTrain*5,1);
    
    XTest=zeros(SplitTest*5,col);
    YTest=zeros(SplitTest*5,1);
    
    %XTrain(1:350,1:20) class1norm
    XTrain(1:SplitTrain*1,:)=Normal(1:SplitTrain,1:col);
    %XTrain(351:700)  = Normal(401:750)class2norm
    XTrain(SplitTrain+1:SplitTrain*2,:)=Normal(TotalSample+1:TotalSample+SplitTrain,1:col);
    %XTrain(701:1050) class 1 upnorm
    XTrain(SplitTrain*2+1:SplitTrain*3,:)=Upnormal(1:SplitTrain,1:col);
    %XTrain(1051:1400) =Upnormal(401:750) class 2 upnorm
    XTrain(SplitTrain*3+1:SplitTrain*4,:)=Upnormal(TotalSample+1:TotalSample+SplitTrain,1:col);
    %XTrain(1401:1750)=Upnormal(801:1150) class 3 upnorm
    XTrain(SplitTrain*4+1:SplitTrain*5,:)=Upnormal(2*TotalSample+1:2*TotalSample+SplitTrain,1:col);
   
    %YTrain(1:350,1:20) class1norm
    YTrain(1:SplitTrain*1,:)=Normal(1:SplitTrain,col+1);
    %YTrain(351:700)  = Normal(401:750)class2norm
    YTrain(SplitTrain+1:SplitTrain*2,:)=Normal(TotalSample+1:TotalSample+SplitTrain,col+1);
    %YTrain(701:1050) class 1 upnorm
    YTrain(SplitTrain*2+1:SplitTrain*3,:)=Upnormal(1:SplitTrain,col+1);
    %YTrain(1051:1400) =Upnormal(401:750) class 2 upnorm
    YTrain(SplitTrain*3+1:SplitTrain*4,:)=Upnormal(TotalSample+1:TotalSample+SplitTrain,col+1);
    %YTrain(1401:1750)=Upnormal(801:1150) class 3 upnorm
    YTrain(SplitTrain*4+1:SplitTrain*5,:)=Upnormal(2*TotalSample+1:2*TotalSample+SplitTrain,col+1);
    
    
    
    %XTest(1:50) =Normal(351,400) class 1norm
    XTest(1:SplitTest*1,:)=Normal(SplitTrain+1:TotalSample,1:col);
    %XTest(51:100) =Normal(750,800)class 2norm
    XTest(SplitTest+1:SplitTest*2,:)=Normal(TotalSample+SplitTrain+1:TotalSample*2,1:col);
    %XTest(101:150)=Upnormal(350:400)class 1upnorm
    XTest(SplitTest*2+1:SplitTest*3,:)=Upnormal(SplitTrain+1:TotalSample,1:col);
    %XTest(151:200)=Upnormal(751:800)class 2upnorm
    XTest(SplitTest*3+1:SplitTest*4,:)=Upnormal(TotalSample+SplitTrain+1:TotalSample*2,1:col);
    %XTest(201:250)upnormal(1150:1200) class 3upnorm
    XTest(SplitTest*4+1:SplitTest*5,:)=Upnormal(2*TotalSample+SplitTrain+1:TotalSample*3,1:col);

    %YTest(1:50) =Normal(351,400) class 1norm
    YTest(1:SplitTest*1,:)=Normal(SplitTrain+1:TotalSample,col+1);
    %YTest(51:100) =Normal(750,800)class 2norm
    YTest(SplitTest+1:SplitTest*2,:)=Normal(TotalSample+SplitTrain+1:TotalSample*2,col+1);
    %YTest(101:150)=Upnormal(350:400)class 1upnorm
    YTest(SplitTest*2+1:SplitTest*3,:)=Upnormal(SplitTrain+1:TotalSample,col+1);
    %YTest(151:200)=Upnormal(751:800)class 2upnorm
    YTest(SplitTest*3+1:SplitTest*4,:)=Upnormal(TotalSample+SplitTrain+1:TotalSample*2,col+1);
    %YTest(201:250)upnormal(1150:1200) class 3upnorm
    YTest(SplitTest*4+1:SplitTest*5,:)=Upnormal(2*TotalSample+SplitTrain+1:TotalSample*3,col+1);
    
end

%% KNN Classifier 
function dist = get_euclidean_distance(X, sample)
    m = size(X,1);
    dist=zeros(m,1);
    summation=0;
    for i=1:m
       f1=sum(( X(i,:) - sample(:,:) ).^2);
        summation=sqrt(f1);
        dist(i,1)=summation;
    end
    
    
end
function dist = get_cosinesimilarty_distance(X, sample)

    m = size(X,1);
    dist=zeros(m,1);
    summation=0;
    for i=1:m
       record=X(i,:);
       num=dot(record,sample);
       recorddot=sqrt(dot(record,record));
       sampledot=sqrt(dot(sample,sample));
       dum=recorddot * sampledot;
       summation=num/dum;
       dist(i,1)=summation;
    end
end
function YPredict=KnnClassifier(XTrain,YTrain,XTest,K,Method)
    TestValue=size(XTest,1);
    YPredict=ones(TestValue,1);  
    norm=0;
    up=0;
    for i=1:TestValue
        if Method == 1
            distance=get_cosinesimilarty_distance(XTrain,XTest(i,:));
        else
            distance=get_euclidean_distance(XTrain,XTest(i,:));
        end
    size(distance);
    distance = [distance YTrain];
    distance=sortrows(distance);
        for j=1:K
            if distance(j,2)==0
                norm=norm+1;
            else
                up=up+1;
            end
        end
        if norm > up
           YPredict(i)=0;
        else
           YPredict(i)=1;
        end
    end
end
function accuracy = CalculateAccuracy(YTrue, YPredict)
    
    %Hint:: you can get the 1 dimenssion of the vector by size(vec,1)
    sz = size(YTrue, 1);
    
    %Hint:: you can know the true predictions by creating a boolean vector 
    %using YTrue == YPredict;
    truePredictions = YTrue == YPredict;
    %Hint:: accuracy = summation(truePredictions) / #samples
    accuracy = (sum(truePredictions)/sz)*100;
end

%% Bayes Classifier 
function mus = EstimateMus(X,AllTrain)   
   %Hint:: you can validate the output by using "mean(X)"
   %Your code goes here ...
   AllTrain=AllTrain*5;
   sumFeat = sum(X);
   mus = sumFeat/AllTrain;
   
end
function sigmas = EstimateSigmas(X, Mus,AllTrain)
    %Hint:: you can validate the output by using "std(X)"
    AllTrain=AllTrain*5;
    %Your code goes here ...
    diff = 0;
    for i = 1:size(X)
        diff = diff + (X(i)- Mus)^2;
    end
    
    sigmas = sqrt(diff/AllTrain);
    
end
function [YPredict] = GeneralBayesClassifier(X, Mus, Sigmas,TestRecords)

    C = 2; % number of classes
    F = 20; %number of features
    m = size(X,1); %number of samples
    
    Mus=Mus.';
    Sigmas=Sigmas.';
    YPredict=zeros(m,1);
    %Your code goes here....   
    %Priors
    Pw = [2/5 3/5];
  for K=1:m        
      %1) Compute the likelihood values for the given features
        Pxjwi = zeros(C,F); %likelihoods
        for i=1:C
            for j=1:F
            Pxjwi(i,j) = mynormalfn(X(K,j), Mus(i,j), Sigmas(i,j));
            end
        end
        PXw = zeros(C,1);
        PXw = prod(Pxjwi,2);
    %2) Compute the posterior probabilities P(wi|x1,x2,x3) by which we decide
    % that the given features belongs to which class!
        PwX = zeros(C,1); %Posteriors
        sum = 0;
        for i=1:C
        PwX(i) = PXw(i) * Pw(i);
        sum = sum + PwX(i);
        end
        PX = sum;

        %Normalize the posteriors
        for i=1:C
        PwX(i) = PwX(i) / PX;
        end
        
        %Now make a decision using the posteriors; 
        if (PwX(1) >= PwX(2)) 
           YPredict(K)=0;
           
        elseif (PwX(2) > PwX(1))
            YPredict(K)=1;
       
        end
  end
end
function p = mynormalfn(x, mu, sigma)
p = (1 / sqrt(2 * pi) * sigma) * exp(-(x - mu).^2/(2 * sigma^2));
end
function [Mus,sigmas]=GetMusSigmas(TrainValue,XTrain)
for j=1:20
        Mus(j,1)=EstimateMus(XTrain(1:TrainValue*2,j),TrainValue);
        Mus(j,2)=EstimateMus(XTrain(TrainValue*2+1:TrainValue*5,j),TrainValue);
        
        sigmas(j,1)=EstimateSigmas(XTrain(1:TrainValue*2,j),Mus(j,1),TrainValue);
        sigmas(j,2)=EstimateSigmas(XTrain(TrainValue*2+1:TrainValue*5,j),Mus(j,2),TrainValue);
        
        Temp(j,1)=std(XTrain(1:TrainValue*2,j));
        Temp(j,2)=std(XTrain(TrainValue*2+1:TrainValue*5,j));
    end
end

%% Nuerel Network full batch gradient descent 

function[acc]= fullBatchMain(X,Y,Network,lr,epochs)

[XTrain, YTrain, XTest, YTest]=data(X,Y);

Network=Network.';

X=X*5;
Y=Y*5;

%Step 1 : Intialaize Network
[NetworkLayer]=IntializeNetwork(Network,X);
NumberLayers=size(Network,2);
Bias=zeros(X,NumberLayers-1);
% %Step 2: Go for FeedForward 
% disp("TRAINING ACCURACY");
J=zeros(epochs,1);
cost1=0;
cost2=0;
for epoch=1:epochs
   [Z,NetworkLayer]=FeedForward(NetworkLayer,XTrain,Bias);
   [Totalcost,cost1,cost2]= CalculateCost(YTrain,Z,cost1,cost2);
   J(epoch,1)=Totalcost;
   [NetworkLayer]=BackPropagation(NetworkLayer,Z,YTrain);
   [NetworkLayer]=updateweights(NetworkLayer,lr);
   accuracy(epoch,1) = CalculateAccuracy(YTrain, Z);
disp("epoch"+ epoch+": Accuracy------------"+accuracy(epoch,1))
end

figure,plot(J)



 Bias=zeros(Y,NumberLayers-1);
 [Z,NetworkLayer]=FeedForward(NetworkLayer,XTest,Bias);
 accuracy = CalculateAccuracy(YTest, Z);
 acc =accuracy;
 disp("Accuracy OF Test is --> " + accuracy);

end

function [Totalcost,cost1,cost2]=CalculateCost(YTrain,Predicted,cost1,cost2)
for i=1:size(YTrain)
    if YTrain(i)==1
        if Predicted(i)==0
            cost1=cost1+0;
        else
            cost1=cost1+(-YTrain(i)*log(Predicted(i)));
        end
    else
        if Predicted(i)==1
            cost2=cost2+0;
        else
            cost2=cost2+(-(1-YTrain(i)*log(1-Predicted(i))));
        end
    end
end
Totalcost=(1/length(Predicted))*(cost1+cost2);
end

%THIS FUNCTION WILL RECIEVE 1D ARRAY WITH NUMBER OF NEURONS
%[ INCLUDED INPUT + HIDDENS + OUTPUT ] 
% WILL RETURN NETWORK WITH ITS WEIGHTS
function [NetworkLayer]=IntializeNetwork(Network,Trainvalue)
NumberLayers=size(Network,2);

for i=1:NumberLayers-1  % LOOP ON EACH LAYER
  
    NetworkLayer(i).Number=i; % Represent #Layer
    NetworkLayer(i).NeuronValue=zeros(Trainvalue,Network(i)); % Represent Value of each neuron each layer
    NetworkLayer(i).LayerWeight=randn(Network(i),Network(i+1))*0.1; % Weight of each Layer
    NetworkLayer(i).LayerSOP=zeros(Trainvalue,Network(i)); % CASH FOR BACK PROPAGATION
    NetworkLayer(i).DELTA=zeros(Network(i),Network(i+1)); % WILL required in back propagation Triangle
    NetworkLayer(i).deltaa=zeros(Trainvalue,Network(i));
end
    NetworkLayer(NumberLayers).Number=NumberLayers;
    NetworkLayer(NumberLayers).NeuronValue=zeros(Trainvalue,Network(NumberLayers));
    NetworkLayer(NumberLayers).LayerWeight=0;
    NetworkLayer(NumberLayers).LayerSOP=zeros(Trainvalue,Network(NumberLayers));
    NetworkLayer(NumberLayers).DELTA=0;
    NetworkLayer(NumberLayers).deltaa=zeros(Trainvalue,Network(NumberLayers));

end

% HERE We multiply Weightof currentlayer with the previous layer
% SOP= WX+B
function [SOP]=SumOfProduct(A,W,Bais)
NeuronNumbersOfLayer=size(W,2);
sample=size(A,1);
SOP=zeros(sample,NeuronNumbersOfLayer);
%Bais Intialize
    for i=1:NeuronNumbersOfLayer
        SOP(:,i)= (A*W(:,i));
        SOP(:,i)=SOP(:,i)+Bais;   
    end
end

% Sigmoid Function Will Take input value of each neuron that represent SOP
% of the NEURON of each layer
function [SOPseg] = Sigmoid(SOP)
SOPseg=1./(1 + exp(-1.*(SOP)));
end

%Dervative Sigmoid Will recieve *****SUM OF PRODUCT*****
function [DervativeSigmoid]=DervativeSigmoid(SOP)

 DervativeSigmoid = Sigmoid(SOP) .* (1 - Sigmoid(SOP));
end


% Function Feed Forward will take each time new record (iX20) and fill
% values of neurons in each layer
function [Z,NetworkLayer]=FeedForward(NetworkLayer,row,Bias)
NetworkLayer(1).NeuronValue=row;
NumberLayers=size(NetworkLayer,2);
for i=2:NumberLayers
    NetworkLayer(i).LayerSOP=SumOfProduct(NetworkLayer(i-1).NeuronValue,NetworkLayer(i-1).LayerWeight,Bias(:,i-1));
    SOPSeg=Sigmoid(NetworkLayer(i).LayerSOP);
    NetworkLayer(i).NeuronValue=SOPSeg;
end
Z= NetworkLayer(NumberLayers).NeuronValue;

% for i=1:size(Z)
% if(Z(i) >= 0.5)
%         Z(i)=1;
%     else
%         Z(i)=0;
% end
Z(Z>=0.5)=1;
Z(Z<0.5)=0;

%end
end

%Function Back Propagation will update the weights
function [NetworkLayer]=BackPropagation(NetworkLayer,Z,Y)
NumberOfSamples=size(Z,1);
NumberLayers=size(NetworkLayer,2);
NetworkLayer(NumberLayers).deltaa=Z-Y;
NextDelta=NetworkLayer(NumberLayers).deltaa;

for i=NumberLayers-1:-1:1
    
    temp=NextDelta*(NetworkLayer(i).LayerWeight.');
    NetworkLayer(i).deltaa=temp.*(DervativeSigmoid( NetworkLayer(i).LayerSOP));
    NextDelta=NetworkLayer(i).deltaa;
end

for i=1:NumberLayers-1
NetworkLayer(i).DELTA=(NetworkLayer(i).NeuronValue.'*(NetworkLayer(i+1).deltaa))./NumberOfSamples;   
end
end
function[NetworkLayer]=updateweights(NetworkLayer,lr)
NumberLayers=size(NetworkLayer,2);
    for i=2:NumberLayers-1
        NetworkLayer(i).LayerWeight=NetworkLayer(i).LayerWeight-lr.*(NetworkLayer(i).DELTA);
    end
end
function [ normalized ] = NormalizeData(M)
mn = min(M);
mx = max(M);
Col=size(M,2);
    for i = 1 : Col
        meancol = mean(M(:,i));
        standcol=std(M(:,i));
        mn=min(M(:,i));
        mx=min(M(:,i));
        normalized(:,i) = (M(:,i)- meancol)/mx-mn;
    end
end
function[XTrain, YTrain, XTest, YTest]=data(X,Y)
[F,N,O,S,Z] = ReadData( 'F','N','O','S','z');
[Normal,Upnormal]=SplittingData(F,N,O,S,Z);
Normal(:,1:20)=NormalizeData(Normal(:,1:20));
Upnormal(:,1:20)=NormalizeData(Upnormal(:,1:20));
% Normal=NormalizeData(Normal);
% Upnormal=NormalizeData(Upnormal);
Normal=Normal ./max(Normal(:));
Upnormal(:,1:20)=Upnormal(:,1:20)./max(Upnormal(:,1:20));
TrainValue=X;
TestValue=Y;
[XTrain, YTrain, XTest, YTest] = PrepareTheData(TrainValue,TestValue,Normal,Upnormal); 
Train=[XTrain YTrain];
XTrainTemp = Train(randperm(size(Train, 1)),:);
XTrain=XTrainTemp(:,1:20);
YTrain=XTrainTemp(:,21);



% XTrain=NormalizeData(XTrain);
end

%% Nuerel Network stochastic gradient descent 
function [acc]=stochasticMain(X,Y,lr,epochs,Network)

%Input From User 

[XTrain, YTrain, XTest, YTest]=data_(X,Y);

Network=Network.';
%Step 1 : Intialaize Network

 [NetworkLayer]=IntializeNetwork_(Network);
 NumberLayers=size(Network,2);
 Bias=zeros(NumberLayers-1,1);
 
% %Step 2: Go for FeedForward 
X=X*5;
Y=Y*5;
cost1=0;
cost2=0;
% Z=zeros(X,1);
accuracy=zeros(epochs,1);
disp("TRAINING ACCURACY");
for epoch=1:epochs
    for row=1:X
     [Z(row,1),NetworkLayer]=FeedForward_(NetworkLayer,XTrain(row,:),Bias);
     [NetworkLayer]=BackPropagation_(NetworkLayer,Z(row,1),YTrain(row,:));
     [NetworkLayer]=updateweights_(NetworkLayer,lr);
    end
[Totalcost,cost1,cost2]= CalculateCost_(YTrain,Z,cost1,cost2);
J(epoch,1)=Totalcost;
accuracy(epoch,1) = CalculateAccuracy(YTrain, Z);
disp("epoch"+ epoch+": Accuracy of Training------------"+accuracy(epoch,1))
end
figure,plot(J)

 Z=zeros(Y,1);
  for row=1:Y
     [Z(row,1),NetworkLayer]=FeedForward_(NetworkLayer,XTest(row,:),Bias);
  end
  accuracy = CalculateAccuracy(YTest, Z);
  acc=accuracy;
disp("Accuracy OF Test is --> " + accuracy);

end


function [Totalcost,cost1,cost2]=CalculateCost_(YTrain,Predicted,cost1,cost2)
for i=1:size(YTrain)
    if YTrain(i)==1
        if Predicted(i)==0
            cost1=cost1+0;
        else
            cost1=cost1+(-YTrain(i)*log(Predicted(i)));
        end
    else
        if Predicted(i)==1
            cost2=cost2+0;
        else
            cost2=cost2+(-(1-YTrain(i)*log(1-Predicted(i))));
        end
    end
end
Totalcost=(1/length(Predicted))*(cost1+cost2);
end

%THIS FUNCTION WILL RECIEVE 1D ARRAY WITH NUMBER OF NEURONS
%[ INCLUDED INPUT + HIDDENS + OUTPUT ] 
% WILL RETURN NETWORK WITH ITS WEIGHTS
function [NetworkLayer]=IntializeNetwork_(Network)
NumberLayers=size(Network,2);

for i=1:NumberLayers-1  % LOOP ON EACH LAYER
  
    NetworkLayer(i).Number=i; % Represent #Layer
    
    NetworkLayer(i).NeuronValue=zeros(Network(i),1); % Represent Value of each neuron each layer

    NetworkLayer(i).LayerWeight=randn(Network(i),Network(i+1)); % Weight of each Layer
   
    NetworkLayer(i).LayerSOP=zeros(Network(i),1); % CASH FOR BACK PROPAGATION
   
    NetworkLayer(i).DELTA=zeros(Network(i),Network(i+1)); % WILL required in back propagation Triangle
    
    NetworkLayer(i).deltaa=zeros(Network(i),1);
end
    NetworkLayer(NumberLayers).Number=NumberLayers;
    NetworkLayer(NumberLayers).NeuronValue=zeros(Network(NumberLayers),1);
    NetworkLayer(NumberLayers).LayerWeight=0;
    NetworkLayer(NumberLayers).LayerSOP=0;
    NetworkLayer(NumberLayers).DELTA=0;
    NetworkLayer(NumberLayers).deltaa=0;
     
    
end
% HERE We multiply Weightof currentlayer with the previous layer
% SOP= WX+B
function [SOP]=SumOfProduct_(A,W,Bais)
NeuronNumbersOfLayer=size(W,2);
%size(A)
%size(W)
SOP=zeros(NeuronNumbersOfLayer,1);
%Bais Intialize
    for i=1:NeuronNumbersOfLayer
        SOP(i)= sum(A*W(:,i));
        SOP(i)=SOP(i)+Bais;
        
    end
    
end
% Sigmoid Function Will Take input value of each neuron that represent SOP
% of the NEURON of each layer
function [SOPseg] = Sigmoid_(SOP)
SOPseg=1 /(1 + exp(-SOP));
end

%Dervative Sigmoid Will recieve *****SUM OF PRODUCT*****
function [DervativeSigmoid]=DervativeSigmoid_(SOP)
SOPseg=1 /(1 + exp(-SOP));
SOPseg2= SOP-SOPseg;
DervativeSigmoid=SOPseg*SOPseg2;
end
% Function Feed Forward will take each time new record (iX20) and fill
% values of neurons in each layer
function [Z,NetworkLayer]=FeedForward_(NetworkLayer,row,Bias)
NetworkLayer(1).NeuronValue=row;

NumberLayers=size(NetworkLayer,2);
for i=2:NumberLayers
%     size(NetworkLayer(i-1).NeuronValue)
%     size(NetworkLayer(i-1).LayerWeight)
    NetworkLayer(i).LayerSOP=SumOfProduct_(NetworkLayer(i-1).NeuronValue,NetworkLayer(i-1).LayerWeight,Bias(i-1));
    
    SOPSeg=Sigmoid_(NetworkLayer(i).LayerSOP);
    
    NetworkLayer(i).NeuronValue=SOPSeg;
end
Z= NetworkLayer(NumberLayers).NeuronValue;
if(Z >= 0.5)
        Z=1;
    else
        Z=0;
end
end

%Function Back Propagation will update the weights
function [NetworkLayer]=BackPropagation_(NetworkLayer,Z,Y)
NumberOfSamples=size(Z,1);
NumberLayers=size(NetworkLayer,2);
NetworkLayer(NumberLayers).deltaa=Z-Y;
NextDelta=NetworkLayer(NumberLayers).deltaa;
for i=NumberLayers-1:-1:1
  
    NetworkLayer(i).LayerWeight;
    temp=NetworkLayer(i).LayerWeight*NextDelta;
    NetworkLayer(i).deltaa=temp.*(DervativeSigmoid_( NetworkLayer(i).LayerSOP).');
    NextDelta=NetworkLayer(i).deltaa;
   
end

for i=1:NumberLayers-1
NetworkLayer(i).DELTA=(NetworkLayer(i).NeuronValue.'*(NetworkLayer(i+1).deltaa).')./NumberOfSamples;   
end
end
function[NetworkLayer]=updateweights_(NetworkLayer,lr)
NumberLayers=size(NetworkLayer,2);
    for i=2:NumberLayers-1
        NetworkLayer(i).LayerWeight=NetworkLayer(i).LayerWeight-lr.*(NetworkLayer(i).DELTA);
    end
end
function[XTrain, YTrain, XTest, YTest]=data_(X,Y)
[F,N,O,S,Z] = ReadData( 'F','N','O','S','z');
[Normal,Upnormal]=SplittingData(F,N,O,S,Z);

Normal=Normal ./max(Normal(:));
Upnormal(:,1:20)=Upnormal(:,1:20)./max(Upnormal(:,1:20));

TrainValue=X;
TestValue=Y;
[XTrain, YTrain, XTest, YTest] = PrepareTheData(TrainValue,TestValue,Normal,Upnormal); 
Train=[XTrain YTrain];
XTrainTemp = Train(randperm(size(Train, 1)),:);
XTrain=XTrainTemp(:,1:20);
YTrain=XTrainTemp(:,21);
end

