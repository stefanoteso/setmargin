% This script performs sparsification of utility vectors on several files
% at once; it make use of function 'sparsifyUtilityVecors'.
%
% Input files whose name contain the pattern 'spase' or 'sparsified' are 
% ignored and as well those containing 'Guo'. Output is stored on different
% files.
%
% We assume that vectors in input files are stored in utilityWeights.
%
% Use with caution, it may mess up your directory!
sparsificationCoeff = 0.8;

% define input directory
baseDir = './randomUtility/';
sparseLabel = 'sparse';
GuoLabel = 'GuoFormat';
extensionTXT = '.txt';
extensionMAT = '.mat';

% look at all MAT files in baseDir
MATfiles=dir([baseDir,'*.mat']);

fprintf('I am processing these files:\n');
for file=MATfiles'
    fullFileName = [baseDir, file.name];
    
    % avoid processing files that contain 'sparsification' in its name
    if not(isempty(strfind(file.name,'sparse')))...
            || not(isempty(strfind(file.name,'sparsified')))... 
            || not(isempty(strfind(file.name,sparseLabel)))
        continue
    end
    
    % avoid processing files that contain 'Guo' in its name
    if not(isempty(strfind(file.name,'Guo'))) 
        continue
    end
    
    % display the name of the file under consideration
    disp(fullFileName);
    load(fullFileName);
    
    % perform sparsification calling function 'sparsifyUtilityVectors'
    % NOTE: assume that vectors are stored in variable utilityWeights
    sparsifiedUtilityVectors=sparsifyUtilityVectors(utilityWeights, sparsificationCoeff);
    
    % define names of output files
    [pathstr,nameWithoutExt,ext] = fileparts(fullFileName);
    outputFileName_TXT = [baseDir, nameWithoutExt, '_',sparseLabel, extensionTXT];
    outputFileName_MAT = [baseDir, nameWithoutExt, '_',sparseLabel, extensionMAT];
    outputFileName_Guo_MAT = [baseDir, nameWithoutExt, '_',sparseLabel,'_',GuoLabel, extensionMAT];   
    
    % save to output files
    utilityWeights = sparsifiedUtilityVectors;
    save(outputFileName_MAT,'utilityWeights');
    dlmwrite(outputFileName_TXT,utilityWeights);
    thisDimDiscreteFeatures = sqrt(size(utilityWeights,2)); %hack
    utilityWeights = convertToGuoFormat(utilityWeights, thisDimDiscreteFeatures);
    save(outputFileName_Guo_MAT,'utilityWeights');
end
