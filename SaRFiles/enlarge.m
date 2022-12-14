function train_enlarge=enlarge(train_original,label_original,direction_len) 

% Input:       
%              train_original: the original training set
%              label_original: the groundtruth
%              direction_len:  the lengths of eight directions obtained by
%                              shape_adaptive.m
%                
% OUTPUT:      train_enlarge:  the enlarged training set

train_add=[];


row_max = size(label_original,1);
col_max = size(label_original,2);

for i=1:size(train_original,2)
    row = mod(train_original(1,i),row_max);
    if row == 0
        row = row_max;
    end
    col = ceil(train_original(1,i)/row_max);
    lens = direction_len(:,train_original(1,i));
   
    pixs_xy=PtsSaR(lens,row,col);
    in=sub2ind([row_max,col_max],pixs_xy(:,1),pixs_xy(:,2))';
    label=train_original(2,i)*ones(1,length(in));
    add=cat(1,in,label);
    train_add=[train_add add];

end
train_adds=[train_original train_add];
train_SR=train_adds';
train_SR=unique(train_SR,'rows');
train_enlarge=train_SR';
