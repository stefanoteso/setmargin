function utilVectorsCell = convertToGuoFormat(utilityVectors, featureDim)
% utilityVectorsCell = convertToGuoFormat(utilityVectors)
%
% Convert a number of utility vectors (given in the rows of the input
% matrix) into a cell array represenation according to the one used by Guo
% (code of the AISTAT'10 paper).
%
% The format in input considers all features to be binary (non-binary features
% are encoded using several binary features, one for each domain element), 
% while the format in output consider discrete features of varying domain
% size.
%
% INPUT
% - utilityVectors: matrix (n X m) where n is the number of vectors,
%       and m is the number of binary features
% - featureDim: array specifying the domain size for each feature
%
% OUTPUT
% utilityVectorsCellArray: cell-array of size (n.vectors,n.features)
nVectors = size(utilityVectors,1);
nFeatures = numel(featureDim);
utilVectorsCell = cell(nVectors,nFeatures);
for i=1:nVectors
    startI = 1;
    for j=1:nFeatures
        endI = startI + featureDim(j) - 1;
        utilVectorsCell{i,j}=utilityVectors(i,startI:endI);
        startI = endI + 1;
    end
end