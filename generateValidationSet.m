% Code to generate validation set
load labeled_images;

validationSet = zeros(32, 32, 0);
validationIdentities = zeros(0, 1);

for i=1:(size(tr_identity, 1) -1)
    if (tr_identity(i, 1) == -1)
        validationSet = cat(3, validationSet, tr_images(:, :, 1));
        validationIdentities = [tr_identity(i, 1) ;validationIdentities];
    end
end
