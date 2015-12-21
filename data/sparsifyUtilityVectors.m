function sparsifiedUtilityVectors=sparsifyUtilityVectors(utilityVectors, sparsificationCoefficient)
% sparsifiedUtilityVectors=sparsifyUtilityVectors(utilityVectors, sparsificationCoefficint)
%
% INPUT
% - utilityVectors:  matrix (n X m), with n=n.of vectors m=n.of binary 
% featuresthis parameter the percentage of binary feature whose 
% weight will be zero
% - sparsificationCoefficint: fraction of features to be put to 0 in each
% row
%
% OUTPUT
% - sparsifiedUtilityVectors: matrix (n X m), with n=n.of vectors m=n.of 
% binary features (same size as utilityVectors given in input) giving the
% utility vectors obtained after sparsification
nVectors=size(utilityVectors,1);
nBinaryFeatures=size(utilityVectors,2);
nIrrelevantFeatures = floor(sparsificationCoefficient*nBinaryFeatures);
sparsifiedUtilityVectors = utilityVectors;
for i=1:nVectors
    % randomly sample nIrrelevantFeatures
    irrelevantFeatures = randsample(nBinaryFeatures,nIrrelevantFeatures);
    sparsifiedUtilityVectors(i,irrelevantFeatures)=0;
end