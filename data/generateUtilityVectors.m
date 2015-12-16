% Script for generating vectors with random utlity weights/parameters
% We consider sampled random vector with no correlation between features

% parameters for uniform distribution
loBoUtil=1; % lower bound on utility parameters
upBoUtil=100; % lower bound on utility parameters

% parameters for normal distribution
muUtil = 25; % mean of utility parameters
stdUtil = 25/3; % std of utility parameters

% n.of parameter vectors to generate (= n. of runs of the experiments)
nVectors = 100; 

% a cell array, in each of them number of domains for each
featuresSettings = { [2,2], [3,3,3], 4*ones(1,4), 5*ones(1,5), 6*ones(1,6)...
    , 7*ones(1,7)};
labelSettings = {'2','3','4','5','6','7'};

% Inline functions for random generation (r=n.rows, c=n.cols)
myRandUniform = @(r,c)(loBoUtil + (upBoUtil-loBoUtil) * rand(r,c));
myRandNormal = @(r,c)(muUtil+stdUtil*randn(r,c));

% declare the random generators that we want to use
randGeneratorHandlers = {myRandUniform, myRandNormal}; % functions handlers
randGeneratorLabels = {'uniform', 'normal'};

% output: cell array containing utility vectors as a matrix
utilVectors=cell(numel(randGeneratorHandlers),numel(featuresSettings)); 
% output: cell array containining utility vecors as cell array
utilVectorCellArray=cell(numel(randGeneratorHandlers),numel(featuresSettings));

% file output
baseDir = './randomUtility/';
baseName = 'utilityParams_synthetic';
GuoLabel = 'GuoFormat';
extensionTXT = '.txt';
extensionMAT = '.mat';

if not(exist(baseDir,'dir'))
    fprintf('Directory %s does not exist. Do you want to create it?\n',baseDir);
    myAnswer = input('Y/N','s');
    if strcmp(myAnswer,'Y')
        mkdir(baseDir);
    else
        error('baseDir does not exist.');
    end
end

fprintf('I am writing the following files:\n');
% generate random utility vectors, in each setting and with each random
% generator
for j = 1:numel(randGeneratorHandlers) %iterate over [uniform, normal]
    thisRandomGenerator = randGeneratorHandlers{j};
    for i = 1:numel(featuresSettings) %iterate over all domain settings
        thisSetting = featuresSettings{i};
        thisNofBinaryFeatures = sum(thisSetting); % n. of equivalent binary features
        utilVectors{j,i} = thisRandomGenerator(nVectors,thisNofBinaryFeatures);
        
        % save to m file (binary feature format)
        suffixName = ['_',labelSettings{i},'_',randGeneratorLabels{j}];
        outputFileName_MAT = [baseDir, baseName, suffixName, extensionMAT];
        utilityWeights = utilVectors{j,i}; %we want to save the current one in its own file
        disp(outputFileName_MAT);
        save(outputFileName_MAT,'utilityWeights');
        
        % save to text file (binary feature format)
        outputFileName_TXT = [baseDir, baseName, suffixName, extensionTXT];
        disp(outputFileName_TXT);
        dlmwrite(outputFileName_TXT,utilVectors{j,i});

        % convert to Guo's format
        utilVectorCellArray{j,i} = convertToGuoFormat(utilVectors{j,i}, thisSetting);        
        % save to m-file (Guo's format)
        suffixName_Guo = [suffixName,'_',GuoLabel];
        outputFileName_Guo_MAT = [baseDir, baseName, suffixName_Guo, extensionMAT];
        utilityWeights = utilVectorCellArray{j,i}; %we want to save the current one in its own file
        disp(outputFileName_Guo_MAT);
        save(outputFileName_Guo_MAT,'utilityWeights');
    end
end